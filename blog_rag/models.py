import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .config import (
    BODY_COLLECTION_NAME,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    FASTEMBED_MODEL,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    MODEL_PROVIDER,
    TITLE_COLLECTION_NAME,
)


def create_embedding_model():
    if MODEL_PROVIDER == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=st.session_state.gemini_api_key,
        )
    if MODEL_PROVIDER == "deepseek":
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=FASTEMBED_MODEL, cache_dir="./model_cache")
    raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")


def create_chat_model():
    temperature = float(st.session_state.model_temperature)
    streaming = bool(st.session_state.model_streaming)
    max_tokens = int(st.session_state.model_max_tokens)

    if MODEL_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(
            api_key=st.session_state.gemini_api_key,
            temperature=temperature,
            streaming=streaming,
            max_output_tokens=max_tokens,
            model=GEMINI_CHAT_MODEL,
        )
    if MODEL_PROVIDER == "deepseek":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=st.session_state.deepseek_api_key,
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_CHAT_MODEL,
            temperature=temperature,
            streaming=streaming,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")


def required_config_ready() -> bool:
    common_ready = bool(st.session_state.qdrant_host and st.session_state.qdrant_api_key)
    if not common_ready:
        return False
    if MODEL_PROVIDER == "gemini":
        return bool(st.session_state.gemini_api_key)
    if MODEL_PROVIDER == "deepseek":
        return bool(st.session_state.deepseek_api_key)
    return False


def initialize_components():
    if not required_config_ready():
        return None, None, None, None

    current_embedding_model = create_embedding_model()
    host = st.session_state.qdrant_host
    if not host.startswith("http"):
        host = "https://" + host

    client = QdrantClient(url=host, api_key=st.session_state.qdrant_api_key, check_compatibility=False)
    dim = 1536 if MODEL_PROVIDER == "gemini" else 384
    existing_collections = {collection.name for collection in client.get_collections().collections}

    for collection_name in [TITLE_COLLECTION_NAME, BODY_COLLECTION_NAME]:
        if collection_name not in existing_collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        for field_name in [
            "metadata.source",
            "metadata.blog_id",
            "metadata.corpus_id",
            "metadata.document_id",
            "metadata.imported_at",
            "metadata.content_kind",
        ]:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema="keyword",
                )
            except Exception:
                pass

    title_db = QdrantVectorStore(client=client, collection_name=TITLE_COLLECTION_NAME, embedding=current_embedding_model)
    body_db = QdrantVectorStore(client=client, collection_name=BODY_COLLECTION_NAME, embedding=current_embedding_model)
    return current_embedding_model, client, title_db, body_db
