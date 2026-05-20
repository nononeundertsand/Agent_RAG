from dataclasses import asdict
from typing import Optional
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList

from blog_rag.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    IngestedDocument,
    IngestRequest,
    IngestResponse,
    RetrievedChunk,
    RetrievalQualityResponse,
)
from blog_rag.cache import generation_cache, query_expansion_cache, retrieval_cache, stable_hash
from blog_rag.config import (
    BODY_COLLECTION_NAME,
    BODY_RETRIEVAL_K,
    MAX_BATCH_URLS,
    RERANK_TOP_K,
    RETRIEVAL_EXPANSION_QUERIES,
    TITLE_COLLECTION_NAME,
    TITLE_RETRIEVAL_K,
)
from blog_rag.document_processing import ExtractedContent, build_documents, extract_from_markdown, extract_from_url
from blog_rag.retrieval_quality import assess_retrieval_quality
from blog_rag.runtime import RuntimeComponents


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _scope_filter(corpus_id: str, tenant_id: str, user_id: Optional[str] = None, permission_group: Optional[str] = None) -> Filter:
    conditions = [
        FieldCondition(key="metadata.corpus_id", match=MatchValue(value=corpus_id)),
        FieldCondition(key="metadata.tenant_id", match=MatchValue(value=tenant_id)),
    ]
    if user_id:
        conditions.append(FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)))
    if permission_group:
        conditions.append(FieldCondition(key="metadata.permission_group", match=MatchValue(value=permission_group)))
    return Filter(must=conditions)


def _source_filter(source: str, tenant_id: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="metadata.source", match=MatchValue(value=source)),
            FieldCondition(key="metadata.tenant_id", match=MatchValue(value=tenant_id)),
        ]
    )


def _delete_by_filter(components: RuntimeComponents, collection_name: str, scroll_filter: Filter):
    points, _ = components.client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        with_payload=False,
        with_vectors=False,
        limit=10_000,
    )
    point_ids = [point.id for point in points if point.id is not None]
    if point_ids:
        components.client.delete(collection_name=collection_name, points_selector=PointIdsList(points=point_ids))


def _add_security_metadata(docs: list[Document], tenant_id: str, user_id: Optional[str], permission_group: Optional[str]):
    for doc in docs:
        doc.metadata["tenant_id"] = tenant_id
        if user_id:
            doc.metadata["user_id"] = user_id
        if permission_group:
            doc.metadata["permission_group"] = permission_group


def ingest_corpus(request: IngestRequest, components: RuntimeComponents) -> IngestResponse:
    if not request.urls and not request.markdown_documents:
        raise ValueError("At least one URL or markdown document is required.")
    if len(request.urls) > MAX_BATCH_URLS:
        raise ValueError(f"Too many URLs. MAX_BATCH_URLS={MAX_BATCH_URLS}.")

    extracted_items: list[ExtractedContent] = []
    for url in request.urls:
        extracted_items.append(extract_from_url(str(url)))
    for document in request.markdown_documents:
        extracted_items.append(extract_from_markdown(document.filename, document.content))

    corpus_id = str(uuid4())
    imported_at = utc_now_iso()
    title_docs: list[Document] = []
    body_docs: list[Document] = []
    ingested_documents: list[IngestedDocument] = []

    for index, extracted in enumerate(extracted_items, start=1):
        title_doc, current_body_docs = build_documents(
            extracted,
            corpus_id,
            imported_at,
            document_id=str(uuid4()),
            document_index=index,
        )
        if not title_doc or not current_body_docs:
            continue

        scoped_docs = [title_doc, *current_body_docs]
        _add_security_metadata(scoped_docs, request.tenant_id, request.user_id, request.permission_group)

        if request.replace_existing_source:
            source_scope = _source_filter(extracted.source, request.tenant_id)
            _delete_by_filter(components, TITLE_COLLECTION_NAME, source_scope)
            _delete_by_filter(components, BODY_COLLECTION_NAME, source_scope)

        title_docs.append(title_doc)
        body_docs.extend(current_body_docs)
        ingested_documents.append(
            IngestedDocument(
                title=extracted.title,
                source=extracted.source,
                content_kind=extracted.content_kind,
                image_count=len(extracted.image_notes),
                body_chunks=len(current_body_docs),
            )
        )

    if not title_docs or not body_docs:
        raise ValueError("No indexable content was extracted.")

    components.title_db.add_documents(title_docs, ids=[str(uuid4()) for _ in title_docs])
    components.body_db.add_documents(body_docs, ids=[str(uuid4()) for _ in body_docs])

    return IngestResponse(
        corpus_id=corpus_id,
        imported_at=imported_at,
        document_count=len(ingested_documents),
        title_points=len(title_docs),
        body_points=len(body_docs),
        documents=ingested_documents,
    )


def _cosine_similarity(vec1, vec2) -> float:
    numerator = sum(x * y for x, y in zip(vec1, vec2))
    denominator = (sum(x * x for x in vec1) ** 0.5) * (sum(y * y for y in vec2) ** 0.5)
    return numerator / denominator if denominator else 0.0


def _rerank(query: str, components: RuntimeComponents, title_docs: list[Document], body_docs: list[Document]):
    query_vector = components.embedding_model.embed_query(query)
    scored_docs = []
    for doc in title_docs:
        score = _cosine_similarity(query_vector, components.embedding_model.embed_query(doc.page_content)) * 1.2
        scored_docs.append((score, "title", doc))
    for doc in body_docs:
        score = _cosine_similarity(query_vector, components.embedding_model.embed_query(doc.page_content))
        scored_docs.append((score, "body", doc))
    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return scored_docs[:RERANK_TOP_K], assess_retrieval_quality(query, scored_docs[:RERANK_TOP_K])


def _dedupe_documents(docs: list[Document]) -> list[Document]:
    seen = set()
    unique_docs = []
    for doc in docs:
        key = (doc.metadata.get("document_id"), doc.metadata.get("content_type"), doc.page_content[:180])
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)
    return unique_docs


def _build_query_expansions(question: str, components: RuntimeComponents) -> list[str]:
    cache_key = stable_hash({"kind": "query_expansion", "provider": components.settings.model_provider, "question": question})
    cached = query_expansion_cache.get(cache_key)
    if cached is not None:
        return cached

    prompt = f"""
Generate {RETRIEVAL_EXPANSION_QUERIES} alternative retrieval queries for the question below.
Use concise Chinese or mixed technical terms when helpful.
Return one query per line, no numbering.

Question:
{question}
"""
    response = components.chat_model.invoke(prompt).content
    variants = [line.strip(" -\t") for line in response.splitlines() if line.strip()]
    expansions = [variant for variant in variants if variant.lower() != question.lower()][:RETRIEVAL_EXPANSION_QUERIES]
    query_expansion_cache.set(cache_key, expansions)
    return expansions


def _request_cache_payload(request: ChatRequest) -> dict:
    return {
        "corpus_id": request.corpus_id,
        "tenant_id": request.tenant_id,
        "user_id": request.user_id,
        "permission_group": request.permission_group,
        "question": request.question,
        "history": [message.model_dump() for message in request.chat_history[-6:]],
        "memory": request.conversation_memory,
    }


def retrieve_context(request: ChatRequest, components: RuntimeComponents):
    cache_key = stable_hash({"kind": "retrieval", **_request_cache_payload(request)})
    cached = retrieval_cache.get(cache_key)
    if cached is not None:
        scored_docs, quality, chunks, events = cached
        events = [*events, {"event": "retrieval_cache_hit"}]
        return scored_docs, quality, chunks, events

    events = [{"event": "retrieve_start", "query": request.question, "corpus_id": request.corpus_id}]
    active_filter = _scope_filter(request.corpus_id, request.tenant_id, request.user_id, request.permission_group)
    title_docs = components.title_db.similarity_search(request.question, k=TITLE_RETRIEVAL_K, filter=active_filter)
    body_docs = components.body_db.similarity_search(request.question, k=BODY_RETRIEVAL_K, filter=active_filter)
    scored_docs, quality = _rerank(request.question, components, title_docs, body_docs)
    events.append({"event": "quality_check", **asdict(quality)})

    if quality.should_expand:
        expansions = _build_query_expansions(request.question, components)
        events.append({"event": "recall_expansion_start", "expansions": expansions, "reason": quality.reason})
        for expanded_query in expansions:
            title_docs.extend(components.title_db.similarity_search(expanded_query, k=TITLE_RETRIEVAL_K, filter=active_filter))
            body_docs.extend(components.body_db.similarity_search(expanded_query, k=BODY_RETRIEVAL_K, filter=active_filter))
        title_docs = _dedupe_documents(title_docs)
        body_docs = _dedupe_documents(body_docs)
        scored_docs, quality = _rerank(request.question, components, title_docs, body_docs)
        events.append({"event": "recall_expansion_done", "title_candidates": len(title_docs), "body_candidates": len(body_docs)})

    chunks = [
        RetrievedChunk(
            rank=index,
            score=round(score, 4),
            type=doc_type,
            title=doc.metadata.get("page_title") or doc.metadata.get("title") or "Untitled",
            source=doc.metadata.get("source", ""),
            document_id=doc.metadata.get("document_id", ""),
            preview=doc.page_content[:360],
        )
        for index, (score, doc_type, doc) in enumerate(scored_docs, start=1)
    ]
    events.append({"event": "retrieve_done", "returned": len(chunks)})
    retrieval_cache.set(cache_key, (scored_docs, quality, chunks, events))
    return scored_docs, quality, chunks, events


def _format_context(scored_docs: list[tuple[float, str, Document]]) -> str:
    sections = []
    for index, (score, doc_type, doc) in enumerate(scored_docs, start=1):
        title = doc.metadata.get("page_title") or doc.metadata.get("title") or "Untitled"
        sections.append(
            f"[{index}] score={score:.4f} type={doc_type}\n"
            f"title: {title}\n"
            f"source: {doc.metadata.get('source', '')}\n"
            f"document_id: {doc.metadata.get('document_id', '')}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(sections)


def _format_history(messages: list[ChatMessage], limit: int = 6) -> str:
    lines = []
    for message in messages[-limit:]:
        role = "User" if message.role == "user" else "Assistant"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)


def generate_answer(request: ChatRequest, components: RuntimeComponents) -> ChatResponse:
    cache_key = stable_hash({"kind": "generation", "provider": components.settings.model_provider, **_request_cache_payload(request)})
    cached = generation_cache.get(cache_key)
    if cached is not None:
        cached.cache = {"hit": True, "layer": "generation", "key": cache_key[:12]}
        cached.events = [*cached.events, {"event": "generation_cache_hit"}]
        return cached

    scored_docs, quality, chunks, events = retrieve_context(request, components)
    context = _format_context(scored_docs)
    history = _format_history(request.chat_history)

    prompt = PromptTemplate(
        template="""
你是企业知识库 RAG 问答服务。请严格基于检索上下文回答，不能编造上下文之外的事实。
如果上下文不足，请明确说明“根据当前检索内容，信息有限”。
回答需要自然、准确、可执行；涉及关键事实时在句末标注来源编号，例如 [1]。

对话记忆:
{memory}

最近对话:
{history}

用户问题:
{question}

检索上下文:
{context}

请直接输出答案:
""",
        input_variables=["memory", "history", "question", "context"],
    )
    answer = (prompt | components.chat_model | StrOutputParser()).invoke(
        {
            "memory": request.conversation_memory or "None",
            "history": history or "None",
            "question": request.question,
            "context": context or "None",
        }
    )

    response = ChatResponse(
        answer=answer,
        corpus_id=request.corpus_id,
        quality=RetrievalQualityResponse(**asdict(quality)),
        retrieved_chunks=chunks,
        events=events,
        cache={"hit": False, "layer": "generation", "key": cache_key[:12]},
    )
    generation_cache.set(cache_key, response)
    return response
