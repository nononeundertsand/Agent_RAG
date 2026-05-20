from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl


class MarkdownDocument(BaseModel):
    filename: str = Field(..., min_length=1, max_length=240)
    content: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    urls: list[HttpUrl] = Field(default_factory=list)
    markdown_documents: list[MarkdownDocument] = Field(default_factory=list)
    tenant_id: str = Field(default="default", min_length=1, max_length=120)
    user_id: Optional[str] = Field(default=None, max_length=120)
    permission_group: Optional[str] = Field(default=None, max_length=120)
    replace_existing_source: bool = True


class IngestedDocument(BaseModel):
    title: str
    source: str
    content_kind: str
    image_count: int
    body_chunks: int


class IngestResponse(BaseModel):
    corpus_id: str
    imported_at: str
    document_count: int
    title_points: int
    body_points: int
    documents: list[IngestedDocument]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    corpus_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    tenant_id: str = Field(default="default", min_length=1, max_length=120)
    user_id: Optional[str] = Field(default=None, max_length=120)
    permission_group: Optional[str] = Field(default=None, max_length=120)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    conversation_memory: str = ""


class RetrievedChunk(BaseModel):
    rank: int
    score: float
    type: str
    title: str
    source: str
    document_id: str
    preview: str


class RetrievalQualityResponse(BaseModel):
    doc_count: int
    max_score: float
    query_coverage: float
    source_count: int
    should_expand: bool
    reason: str


class ChatResponse(BaseModel):
    answer: str
    corpus_id: str
    quality: RetrievalQualityResponse
    retrieved_chunks: list[RetrievedChunk]
    events: list[dict]
    cache: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_provider: str
    qdrant_configured: bool
    cache: dict = Field(default_factory=dict)
