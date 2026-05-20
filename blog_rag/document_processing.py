import re
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import INGEST_CHUNK_OVERLAP, INGEST_CHUNK_SIZE


@dataclass
class ExtractedContent:
    title: str
    body: str
    image_notes: list[str]
    source: str
    content_kind: str


def clean_lines(text: str) -> str:
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


def extract_images_from_soup(soup: BeautifulSoup) -> list[str]:
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
        payload = " | ".join(
            part for part in [f"alt: {alt}", f"title: {title}", f"caption: {caption}", f"src: {src}"] if part and part.strip(": ")
        )
        if payload.strip():
            notes.append(f"image {index}: {payload}")
    return notes


def extract_from_url(url: str) -> ExtractedContent:
    response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 AgentRAG/2.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    image_notes = extract_images_from_soup(soup)

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
    return ExtractedContent(title=title, body=clean_lines(target.get_text("\n", strip=True)), image_notes=image_notes, source=url, content_kind="url")


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
        image_notes.append(f"image {index}: alt: {alt or 'empty'} | src: {src}")

    soup = BeautifulSoup(content, "html.parser")
    html_image_count = len(image_notes)
    for index, img in enumerate(soup.find_all("img"), start=html_image_count + 1):
        alt = img.get("alt") or "empty"
        src = img.get("src") or ""
        title_attr = img.get("title") or ""
        image_notes.append(f"image {index}: alt: {alt} | title: {title_attr} | src: {src}")

    text_without_images = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)
    return ExtractedContent(
        title=title,
        body=clean_lines(text_without_images),
        image_notes=image_notes,
        source=filename,
        content_kind="markdown",
    )


def build_documents(
    extracted: ExtractedContent,
    corpus_id: str,
    imported_at: str,
    document_id: Optional[str] = None,
    document_index: int = 1,
) -> tuple[Optional[Document], list[Document]]:
    if not extracted.body.strip() and not extracted.image_notes:
        return None, []

    document_id = document_id or str(uuid4())
    metadata = {
        "source": extracted.source,
        "blog_id": corpus_id,
        "corpus_id": corpus_id,
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
        sections.append("Image and multimodal clues:\n" + "\n".join(extracted.image_notes))
    combined_body = "\n\n".join(part for part in sections if part.strip())

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=INGEST_CHUNK_SIZE, chunk_overlap=INGEST_CHUNK_OVERLAP)
    body_docs = splitter.create_documents(
        [combined_body],
        metadatas=[{**metadata, "content_type": "body", "page_title": extracted.title, "image_count": len(extracted.image_notes)}],
    )
    return title_doc, body_docs
