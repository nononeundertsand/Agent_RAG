import re
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList

from .config import BODY_COLLECTION_NAME, INGEST_CHUNK_OVERLAP, INGEST_CHUNK_SIZE, TITLE_COLLECTION_NAME
from .session_state import create_chat_session, utc_now_iso


@dataclass
class ExtractedContent:
    title: str
    body: str
    image_notes: list[str]
    source: str
    content_kind: str


def _clean_lines(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        normalized = " ".join(line.split())
        if len(normalized) < 16:
            continue
        lower_line = normalized.lower()
        if any(keyword in lower_line for keyword in ["advertisement", "subscribe", "all rights reserved", "cookie"]):
            continue
        cleaned_lines.append(normalized)
    return "\n".join(cleaned_lines)


def _extract_images_from_soup(soup: BeautifulSoup, base_url: str) -> list[str]:
    notes = []
    for index, img in enumerate(soup.find_all("img"), start=1):
        alt = img.get("alt") or ""
        title = img.get("title") or ""
        src = img.get("src") or img.get("data-src") or ""
        caption = ""
        figure = img.find_parent("figure")
        if figure:
            figcaption = figure.find("figcaption")
            caption = figcaption.get_text(" ", strip=True) if figcaption else ""
        payload = " | ".join(part for part in [f"alt: {alt}", f"title: {title}", f"caption: {caption}", f"src: {src}"] if part and part.strip(": "))
        if payload.strip():
            notes.append(f"图片 {index}: {payload}")
    return notes


def extract_from_url(url: str) -> ExtractedContent:
    response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 AgenticRAG/1.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    image_notes = _extract_images_from_soup(soup, url)

    for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()
    for tag in soup.find_all(
        attrs={
            "class": lambda value: value
            and any(
                keyword in " ".join(value).lower()
                for keyword in ["ad", "ads", "banner", "popup", "subscribe", "cookie", "promo", "related", "sidebar"]
            )
        }
    ):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else url

    candidates = []
    for selector in ["article", "main", "[role='main']", ".post-content", ".entry-content", ".article-content"]:
        candidates.extend(soup.select(selector))
    target = max(candidates, key=lambda node: len(node.get_text(" ", strip=True)), default=soup.body or soup)
    return ExtractedContent(title=title, body=_clean_lines(target.get_text("\n", strip=True)), image_notes=image_notes, source=url, content_kind="url")


def extract_from_markdown(filename: str, content: str) -> ExtractedContent:
    title = ""
    for line in content.splitlines():
        if line.strip().startswith("#"):
            title = line.strip("# ").strip()
            break
    if not title:
        title = filename

    image_notes = []
    for index, match in enumerate(re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", content), start=1):
        alt, src = match.groups()
        image_notes.append(f"图片 {index}: alt: {alt or '无'} | src: {src}")

    soup = BeautifulSoup(content, "html.parser")
    html_image_count = len(image_notes)
    for index, img in enumerate(soup.find_all("img"), start=html_image_count + 1):
        alt = img.get("alt") or "无"
        src = img.get("src") or ""
        title_attr = img.get("title") or ""
        image_notes.append(f"图片 {index}: alt: {alt} | title: {title_attr} | src: {src}")

    text_without_images = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)
    return ExtractedContent(
        title=title,
        body=_clean_lines(text_without_images),
        image_notes=image_notes,
        source=filename,
        content_kind="markdown",
    )


def build_documents(
    extracted: ExtractedContent,
    blog_id: str,
    imported_at: str,
    document_id: Optional[str] = None,
    document_index: int = 1,
) -> tuple[Optional[Document], list[Document]]:
    if not extracted.body.strip() and not extracted.image_notes:
        return None, []

    document_id = document_id or str(uuid4())
    metadata = {
        "source": extracted.source,
        "blog_id": blog_id,
        "corpus_id": blog_id,
        "document_id": document_id,
        "document_index": document_index,
        "imported_at": imported_at,
        "content_kind": extracted.content_kind,
    }
    title_doc = Document(
        page_content=extracted.title,
        metadata={**metadata, "content_type": "title", "page_title": extracted.title},
    )

    sections = [extracted.body]
    if extracted.image_notes:
        sections.append("页面图片与多模态线索:\n" + "\n".join(extracted.image_notes))
    combined_body = "\n\n".join(part for part in sections if part.strip())

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=INGEST_CHUNK_SIZE, chunk_overlap=INGEST_CHUNK_OVERLAP)
    body_docs = splitter.create_documents(
        [combined_body],
        metadatas=[{**metadata, "content_type": "body", "page_title": extracted.title, "image_count": len(extracted.image_notes)}],
    )
    return title_doc, body_docs


def clear_documents_by_source(client, collection_name: str, source: str):
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]),
        with_payload=False,
        with_vectors=False,
        limit=1000,
    )
    point_ids = [point.id for point in points if point.id is not None]
    if point_ids:
        client.delete(collection_name=collection_name, points_selector=PointIdsList(points=point_ids))


def add_extracted_to_qdrant(extracted: ExtractedContent, client, title_db, body_db) -> bool:
    return add_extracted_batch_to_qdrant([extracted], client, title_db, body_db)


def add_extracted_batch_to_qdrant(extracted_items: list[ExtractedContent], client, title_db, body_db) -> bool:
    try:
        extracted_items = [item for item in extracted_items if item.body.strip() or item.image_notes]
        if not extracted_items:
            st.error("没有提取到可索引的正文或图片说明。")
            return False

        blog_id = str(uuid4())
        imported_at = utc_now_iso()
        title_docs = []
        body_docs = []
        sources = []
        titles = []
        image_count = 0

        for index, extracted in enumerate(extracted_items, start=1):
            title_doc, current_body_docs = build_documents(
                extracted,
                blog_id,
                imported_at,
                document_id=str(uuid4()),
                document_index=index,
            )
            if not title_doc or not current_body_docs:
                continue

            clear_documents_by_source(client, TITLE_COLLECTION_NAME, extracted.source)
            clear_documents_by_source(client, BODY_COLLECTION_NAME, extracted.source)
            title_docs.append(title_doc)
            body_docs.extend(current_body_docs)
            sources.append(extracted.source)
            titles.append(extracted.title or extracted.source)
            image_count += len(extracted.image_notes)

        if not title_docs or not body_docs:
            st.error("没有提取到可索引的正文或图片说明。")
            return False

        title_db.add_documents(title_docs, ids=[str(uuid4()) for _ in title_docs])
        body_db.add_documents(body_docs, ids=[str(uuid4()) for _ in body_docs])

        display_title = titles[0] if len(titles) == 1 else f"{titles[0]} 等 {len(titles)} 篇资料"
        source_text = sources[0] if len(sources) == 1 else "\n".join(sources)
        content_kind = extracted_items[0].content_kind if len({item.content_kind for item in extracted_items}) == 1 else "mixed"

        st.session_state.last_loaded_url = source_text
        st.session_state.current_sources = sources
        st.session_state.current_blog_id = blog_id
        st.session_state.current_blog_title = display_title
        st.session_state.current_blog_imported_at = imported_at
        st.session_state.current_content_kind = content_kind
        st.session_state.current_image_count = image_count
        st.session_state.current_document_count = len(title_docs)
        st.session_state.retrieval_debug = []
        st.session_state.retrieval_events = []
        st.session_state.chat_history = []
        st.session_state.conversation_memory = ""
        create_chat_session(
            source_text,
            blog_id,
            display_title,
            imported_at,
            content_kind,
            image_count,
            sources=sources,
            document_count=len(title_docs),
        )
        return True
    except Exception as exc:
        st.error(f"导入失败：{exc}")
        return False
