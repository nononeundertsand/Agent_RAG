import json
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import streamlit as st

from .config import (
    CHAT_SESSIONS_FILE,
    DEFAULT_MODEL_MAX_TOKENS,
    DEFAULT_MODEL_STREAMING,
    DEFAULT_MODEL_TEMPERATURE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_session_state():
    defaults = {
        "qdrant_host": "",
        "qdrant_api_key": "",
        "gemini_api_key": "",
        "deepseek_api_key": "",
        "retrieval_debug": [],
        "retrieval_events": [],
        "retrieval_quality": {},
        "last_loaded_url": "",
        "current_blog_id": "",
        "current_blog_title": "",
        "current_blog_imported_at": "",
        "current_content_kind": "",
        "current_image_count": 0,
        "current_document_count": 0,
        "current_sources": [],
        "chat_sessions": {},
        "active_session_id": "",
        "chat_history": [],
        "conversation_memory": "",
        "model_temperature": DEFAULT_MODEL_TEMPERATURE,
        "model_streaming": DEFAULT_MODEL_STREAMING,
        "model_max_tokens": DEFAULT_MODEL_MAX_TOKENS,
        "answer_pending": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    load_chat_sessions()


def load_chat_sessions():
    if st.session_state.get("chat_sessions") or not CHAT_SESSIONS_FILE.exists():
        return
    try:
        data = json.loads(CHAT_SESSIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        st.session_state.chat_sessions = {}
        return

    st.session_state.chat_sessions = data.get("sessions", {})
    active_session_id = data.get("active_session_id", "")
    if active_session_id in st.session_state.chat_sessions:
        switch_chat_session(active_session_id, persist=False)


def save_chat_sessions():
    data = {
        "active_session_id": st.session_state.get("active_session_id", ""),
        "sessions": st.session_state.get("chat_sessions", {}),
    }
    CHAT_SESSIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_active_session() -> Optional[dict]:
    session_id = st.session_state.get("active_session_id", "")
    if not session_id:
        return None
    return st.session_state.chat_sessions.get(session_id)


def sync_active_session():
    session = get_active_session()
    if not session:
        return
    session.update(
        {
            "url": st.session_state.get("last_loaded_url", ""),
            "blog_id": st.session_state.get("current_blog_id", ""),
            "blog_title": st.session_state.get("current_blog_title", ""),
            "imported_at": st.session_state.get("current_blog_imported_at", ""),
            "content_kind": st.session_state.get("current_content_kind", ""),
            "image_count": st.session_state.get("current_image_count", 0),
            "document_count": st.session_state.get("current_document_count", 0),
            "sources": st.session_state.get("current_sources", []),
            "chat_history": st.session_state.get("chat_history", []),
            "conversation_memory": st.session_state.get("conversation_memory", ""),
            "updated_at": utc_now_iso(),
        }
    )
    save_chat_sessions()


def switch_chat_session(session_id: str, persist: bool = True):
    session = st.session_state.chat_sessions.get(session_id)
    if not session:
        return
    st.session_state.active_session_id = session_id
    st.session_state.last_loaded_url = session.get("url", "")
    st.session_state.current_blog_id = session.get("blog_id", "")
    st.session_state.current_blog_title = session.get("blog_title", "")
    st.session_state.current_blog_imported_at = session.get("imported_at", "")
    st.session_state.current_content_kind = session.get("content_kind", "")
    st.session_state.current_image_count = session.get("image_count", 0)
    st.session_state.current_document_count = session.get("document_count", 1 if session.get("url") else 0)
    st.session_state.current_sources = session.get("sources", [session.get("url", "")] if session.get("url") else [])
    st.session_state.chat_history = session.get("chat_history", [])
    st.session_state.conversation_memory = session.get("conversation_memory", "")
    st.session_state.retrieval_debug = []
    st.session_state.retrieval_events = []
    if persist:
        save_chat_sessions()


def create_chat_session(
    source: str,
    blog_id: str,
    title: str,
    imported_at: str,
    content_kind: str,
    image_count: int,
    sources: Optional[list[str]] = None,
    document_count: int = 1,
):
    session_id = str(uuid4())
    display_title = title or source
    st.session_state.chat_sessions[session_id] = {
        "id": session_id,
        "title": display_title[:80],
        "url": source,
        "blog_id": blog_id,
        "blog_title": title,
        "imported_at": imported_at,
        "content_kind": content_kind,
        "image_count": image_count,
        "document_count": document_count,
        "sources": sources or ([source] if source else []),
        "created_at": imported_at,
        "updated_at": imported_at,
        "chat_history": [],
        "conversation_memory": "",
    }
    switch_chat_session(session_id)


def record_retrieval_event(event: str, **payload):
    events = st.session_state.get("retrieval_events", [])
    events.append({"time": utc_now_iso(), "event": event, **payload})
    st.session_state.retrieval_events = events[-40:]
