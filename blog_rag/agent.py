from functools import partial
from typing import Annotated, Optional, Sequence

import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from qdrant_client.models import FieldCondition, Filter, MatchValue
from typing_extensions import TypedDict

from .config import BODY_RETRIEVAL_K, MAX_REWRITE_ATTEMPTS, RERANK_TOP_K, RETRIEVAL_EXPANSION_QUERIES, TITLE_RETRIEVAL_K
from .models import create_chat_model
from .retrieval_quality import assess_retrieval_quality
from .session_state import record_retrieval_event


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    rewrite_count: int


def get_recent_chat_history_text(limit: int = 6) -> str:
    history = st.session_state.get("chat_history", [])
    lines = []
    for item in history[-limit:]:
        role = "User" if item["role"] == "user" else "Assistant"
        lines.append(f"{role}: {item['content']}")
    return "\n".join(lines)


def update_conversation_memory(user_query: str, assistant_response: str):
    previous_memory = st.session_state.get("conversation_memory", "")
    prompt = f"""
You are maintaining compressed conversational memory for a blog QA assistant.

Existing memory:
{previous_memory or "None"}

Latest user question:
{user_query}

Latest assistant answer:
{assistant_response}

Return a concise Chinese memory summary. Keep only durable facts, references and user intent.
"""
    try:
        response = create_chat_model().invoke(prompt)
        st.session_state.conversation_memory = response.content.strip()
    except Exception:
        st.session_state.conversation_memory = "\n".join([previous_memory, f"最近问题: {user_query}", f"最近回答: {assistant_response[:300]}"]).strip()


def build_current_blog_filter() -> Optional[Filter]:
    blog_id = st.session_state.get("current_blog_id", "")
    if blog_id:
        return Filter(must=[FieldCondition(key="metadata.blog_id", match=MatchValue(value=blog_id))])
    source = st.session_state.get("last_loaded_url", "")
    if source:
        return Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))])
    return None


def cosine_similarity(vec1, vec2) -> float:
    numerator = sum(x * y for x, y in zip(vec1, vec2))
    denominator = (sum(x * x for x in vec1) ** 0.5) * (sum(y * y for y in vec2) ** 0.5)
    return numerator / denominator if denominator else 0.0


def rerank_documents(query: str, embedding_model, title_docs: list[Document], body_docs: list[Document]):
    query_vector = embedding_model.embed_query(query)
    scored_docs = []
    for doc in title_docs:
        score = cosine_similarity(query_vector, embedding_model.embed_query(doc.page_content)) * 1.2
        scored_docs.append((score, "title", doc))
    for doc in body_docs:
        score = cosine_similarity(query_vector, embedding_model.embed_query(doc.page_content))
        scored_docs.append((score, "body", doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    quality = assess_retrieval_quality(query, scored_docs[:RERANK_TOP_K])
    st.session_state.retrieval_quality = quality.__dict__
    record_retrieval_event(
        "quality_check",
        doc_count=quality.doc_count,
        max_score=round(quality.max_score, 4),
        query_coverage=round(quality.query_coverage, 4),
        source_count=quality.source_count,
        should_expand=quality.should_expand,
        reason=quality.reason,
    )
    top_items = scored_docs[:RERANK_TOP_K]
    st.session_state.retrieval_debug = [
        {
            "rank": index,
            "score": round(score, 4),
            "type": doc_type,
            "title": doc.metadata.get("page_title") or doc.metadata.get("title") or "Untitled",
            "source": doc.metadata.get("source", ""),
            "blog_id": doc.metadata.get("blog_id", ""),
            "imported_at": doc.metadata.get("imported_at", ""),
            "preview": doc.page_content[:260],
        }
        for index, (score, doc_type, doc) in enumerate(top_items, start=1)
    ]
    record_retrieval_event("rerank", query=query, title_candidates=len(title_docs), body_candidates=len(body_docs), top_k=len(top_items))
    return [doc for _, _, doc in top_items]


def dedupe_documents(docs: list[Document]) -> list[Document]:
    seen = set()
    unique_docs = []
    for doc in docs:
        key = (
            doc.metadata.get("document_id"),
            doc.metadata.get("content_type"),
            doc.page_content[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)
    return unique_docs


def build_query_expansions(query: str) -> list[str]:
    prompt = f"""
Generate {RETRIEVAL_EXPANSION_QUERIES} alternative retrieval queries for the question below.
Use a mix of Chinese and key technical terms if useful. Keep each query short.
Return one query per line, no numbering.

Question:
{query}
"""
    try:
        response = create_chat_model().invoke(prompt).content
        variants = [line.strip(" -\t") for line in response.splitlines() if line.strip()]
    except Exception:
        variants = []
    variants = [variant for variant in variants if variant and variant.lower() != query.lower()]
    return variants[:RETRIEVAL_EXPANSION_QUERIES]


def format_docs_for_prompt(docs: list[Document]) -> str:
    sections = []
    for index, doc in enumerate(docs, start=1):
        title = doc.metadata.get("page_title") or doc.metadata.get("title") or "Untitled"
        content_type = doc.metadata.get("content_type", "body")
        sections.append(
            f"[{index}] ({content_type}) {title}\n"
            f"source: {doc.metadata.get('source', '')}\n"
            f"imported_at: {doc.metadata.get('imported_at', '')}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(sections)


def create_blog_retrieval_tool(title_db, body_db, embedding_model):
    @tool("retrieve_blog_posts")
    def retrieve_blog_posts(query: str) -> str:
        """Search and return information about the document attached to the current chat session."""
        active_filter = build_current_blog_filter()
        record_retrieval_event("retrieve_start", query=query, active_blog_id=st.session_state.get("current_blog_id", ""))
        title_docs = title_db.similarity_search(query, k=TITLE_RETRIEVAL_K, filter=active_filter)
        body_docs = body_db.similarity_search(query, k=BODY_RETRIEVAL_K, filter=active_filter)
        docs = rerank_documents(query, embedding_model, title_docs, body_docs)
        quality = st.session_state.get("retrieval_quality", {})

        if quality.get("should_expand"):
            expansions = build_query_expansions(query)
            record_retrieval_event("recall_expansion_start", reason=quality.get("reason"), expansions=expansions)
            for expanded_query in expansions:
                title_docs.extend(title_db.similarity_search(expanded_query, k=TITLE_RETRIEVAL_K, filter=active_filter))
                body_docs.extend(body_db.similarity_search(expanded_query, k=BODY_RETRIEVAL_K, filter=active_filter))
            title_docs = dedupe_documents(title_docs)
            body_docs = dedupe_documents(body_docs)
            docs = rerank_documents(query, embedding_model, title_docs, body_docs)
            record_retrieval_event(
                "recall_expansion_done",
                expanded_queries=len(expansions),
                title_candidates=len(title_docs),
                body_candidates=len(body_docs),
            )

        record_retrieval_event("retrieve_done", returned=len(docs), titles=len(title_docs), bodies=len(body_docs))
        return format_docs_for_prompt(docs)

    return retrieve_blog_posts


def grade_documents(state):
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    rewrite_count = state.get("rewrite_count", 0)
    prompt = f"""
You are a grader assessing relevance of retrieved context to a user question.

Context:
{docs}

Question:
{question}

Answer ONLY with "yes" or "no".
"""
    score = create_chat_model().invoke(prompt).content.lower().strip()
    if "yes" in score or rewrite_count >= MAX_REWRITE_ATTEMPTS:
        return "generate"
    return "rewrite"


def agent_node(state, tools):
    messages = state["messages"]
    memory_summary = st.session_state.get("conversation_memory", "")
    recent_history = get_recent_chat_history_text()
    system_prompt = SystemMessage(
        content=f"""
You are an AI assistant that MUST use the retrieval tool before answering.
Base final answers ONLY on retrieved content. The indexed source may include web text, Markdown text, and image descriptions extracted from the page or Markdown.

Conversation memory:
{memory_summary or "None"}

Recent chat history:
{recent_history or "None"}
"""
    )
    response = create_chat_model().bind_tools(tools).invoke([system_prompt] + list(messages))
    return {"messages": [response]}


def rewrite(state):
    question = state["messages"][0].content
    response = create_chat_model().invoke(
        [
            HumanMessage(
                content=f"""
Rewrite the question for semantic retrieval. Preserve the user's intent and resolve references from chat memory if possible.

Question:
{question}
"""
            )
        ]
    )
    return {"messages": [response], "rewrite_count": state.get("rewrite_count", 0) + 1}


def generate(state):
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    memory_summary = st.session_state.get("conversation_memory", "")
    recent_history = get_recent_chat_history_text()
    prompt_template = PromptTemplate(
        template="""
你是一名高质量的文档问答助手。请严格基于检索上下文，用中文给出清晰、充分、可信的回答。

要求：
1. 优先依据 context，不要脱离上下文编造事实。
2. 如果 context 信息不足，明确说明“根据当前检索内容，信息有限”。
3. 如问题涉及图片或界面视觉信息，请结合 context 中的“图片/多模态线索”回答。
4. 结构自然即可，避免空泛套话。

问题与对话线索：
{question}

检索上下文：
{context}

请直接输出答案：
""",
        input_variables=["context", "question"],
    )
    augmented_question = question
    if memory_summary:
        augmented_question += f"\n\n对话记忆:\n{memory_summary}"
    if recent_history:
        augmented_question += f"\n\n最近对话:\n{recent_history}"
    return {"messages": [(prompt_template | create_chat_model() | StrOutputParser()).invoke({"context": docs, "question": augmented_question})]}


def get_graph(retriever_tool):
    tools = [retriever_tool]
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", partial(agent_node, tools=tools))
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")
    return workflow.compile()


def run_graph_with_progress(graph, inputs, status=None) -> str:
    generated_message = ""
    labels = {
        "agent": "正在分析问题并选择检索方式...",
        "retrieve": "正在检索标题、正文和图片线索...",
        "rewrite": "检索结果不够理想，正在改写查询...",
        "generate": "已经拿到依据，正在组织最终回答...",
    }
    for output in graph.stream(inputs):
        for key, value in output.items():
            if status and key in labels:
                status.update(label=labels[key], state="running")
            if key == "generate" and isinstance(value, dict):
                generated_message = value.get("messages", [""])[0]
    if status:
        status.update(label="回答已生成", state="complete")
    return generated_message
