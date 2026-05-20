import os
from dataclasses import dataclass
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .config import (
    BODY_COLLECTION_NAME,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    DEFAULT_MODEL_MAX_TOKENS,
    DEFAULT_MODEL_STREAMING,
    DEFAULT_MODEL_TEMPERATURE,
    FASTEMBED_MODEL,
    GEMINI_CHAT_MODEL,
    GEMINI_EMBEDDING_MODEL,
    MODEL_PROVIDER,
    TITLE_COLLECTION_NAME,
)


@dataclass(frozen=True)
class RuntimeSettings:
    model_provider: str
    qdrant_url: str
    qdrant_api_key: str
    gemini_api_key: str
    deepseek_api_key: str
    model_temperature: float
    model_streaming: bool
    model_max_tokens: int


@dataclass
class RuntimeComponents:
    settings: RuntimeSettings
    embedding_model: object
    chat_model: object
    client: QdrantClient
    title_db: QdrantVectorStore
    body_db: QdrantVectorStore


def get_runtime_settings() -> RuntimeSettings:
    qdrant_url = os.getenv("QDRANT_URL") or os.getenv("QDRANT_HOST") or ""
    if qdrant_url and not qdrant_url.startswith("http"):
        qdrant_url = "https://" + qdrant_url

    return RuntimeSettings(
        model_provider=os.getenv("MODEL_PROVIDER", MODEL_PROVIDER).lower(),
        qdrant_url=qdrant_url,
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        model_temperature=float(os.getenv("MODEL_TEMPERATURE", DEFAULT_MODEL_TEMPERATURE)),
        model_streaming=os.getenv("MODEL_STREAMING", str(DEFAULT_MODEL_STREAMING)).lower() in {"1", "true", "yes"},
        model_max_tokens=int(os.getenv("MODEL_MAX_TOKENS", DEFAULT_MODEL_MAX_TOKENS)),
    )


def validate_settings(settings: RuntimeSettings):
    if not settings.qdrant_url:
        raise ValueError("Missing QDRANT_URL or QDRANT_HOST.")
    if not settings.qdrant_api_key:
        raise ValueError("Missing QDRANT_API_KEY.")
    if settings.model_provider == "gemini" and not settings.gemini_api_key:
        raise ValueError("Missing GEMINI_API_KEY.")
    if settings.model_provider == "deepseek" and not settings.deepseek_api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY.")


def create_runtime_embedding_model(settings: RuntimeSettings):
    if settings.model_provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=settings.gemini_api_key,
        )
    if settings.model_provider == "deepseek":
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=FASTEMBED_MODEL, cache_dir="./model_cache")
    raise ValueError(f"Unsupported MODEL_PROVIDER: {settings.model_provider}")


def create_runtime_chat_model(settings: RuntimeSettings):
    if settings.model_provider == "gemini":
        return ChatGoogleGenerativeAI(
            api_key=settings.gemini_api_key,
            temperature=settings.model_temperature,
            streaming=settings.model_streaming,
            max_output_tokens=settings.model_max_tokens,
            model=GEMINI_CHAT_MODEL,
        )
    if settings.model_provider == "deepseek":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=DEEPSEEK_BASE_URL,
            model=DEEPSEEK_CHAT_MODEL,
            temperature=settings.model_temperature,
            streaming=settings.model_streaming,
            max_tokens=settings.model_max_tokens,
        )
    raise ValueError(f"Unsupported MODEL_PROVIDER: {settings.model_provider}")


def ensure_collections(client: QdrantClient, vector_size: int):
    existing_collections = {collection.name for collection in client.get_collections().collections}
    for collection_name in [TITLE_COLLECTION_NAME, BODY_COLLECTION_NAME]:
        if collection_name not in existing_collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        for field_name in [
            "metadata.tenant_id",
            "metadata.user_id",
            "metadata.permission_group",
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


@lru_cache(maxsize=1)
def get_runtime_components() -> RuntimeComponents:
    settings = get_runtime_settings()
    validate_settings(settings)

    embedding_model = create_runtime_embedding_model(settings)
    chat_model = create_runtime_chat_model(settings)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, check_compatibility=False)
    vector_size = 1536 if settings.model_provider == "gemini" else 384
    ensure_collections(client, vector_size)

    title_db = QdrantVectorStore(client=client, collection_name=TITLE_COLLECTION_NAME, embedding=embedding_model)
    body_db = QdrantVectorStore(client=client, collection_name=BODY_COLLECTION_NAME, embedding=embedding_model)
    return RuntimeComponents(
        settings=settings,
        embedding_model=embedding_model,
        chat_model=chat_model,
        client=client,
        title_db=title_db,
        body_db=body_db,
    )
