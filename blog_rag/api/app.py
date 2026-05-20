from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from blog_rag.api.schemas import ChatRequest, ChatResponse, HealthResponse, IngestRequest, IngestResponse
from blog_rag.cache import generation_cache, query_expansion_cache, retrieval_cache
from blog_rag.runtime import RuntimeComponents, get_runtime_components, get_runtime_settings
from blog_rag.services.rag_service import generate_answer, ingest_corpus


app = FastAPI(
    title="Agent RAG API",
    version="0.2.0",
    description="Production-oriented RAG API with corpus ingestion, scoped retrieval and grounded answer generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    response = await call_next(request)
    request_id = request.headers.get("x-request-id")
    if request_id:
        response.headers["x-request-id"] = request_id
    return response


def components_dependency() -> RuntimeComponents:
    try:
        return get_runtime_components()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
def health():
    settings = get_runtime_settings()
    return HealthResponse(
        status="ok" if settings.qdrant_url and settings.qdrant_api_key else "degraded",
        model_provider=settings.model_provider,
        qdrant_configured=bool(settings.qdrant_url and settings.qdrant_api_key),
        cache={
            "retrieval": retrieval_cache.snapshot(),
            "generation": generation_cache.snapshot(),
            "query_expansion": query_expansion_cache.snapshot(),
        },
    )


@app.post("/v1/corpora/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, components: RuntimeComponents = Depends(components_dependency)):
    try:
        return ingest_corpus(request, components)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest, components: RuntimeComponents = Depends(components_dependency)):
    try:
        return generate_answer(request, components)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
