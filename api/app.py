from fastapi import APIRouter, FastAPI, Request  # pyright: ignore[reportMissingImports]
from .models import RAGRequest, RAGResponse, RAGUsedContext
from .middleware import RequestIDMiddleware
from .Agent.graph import rag_agent_wrapper as run_agent_wrapper
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()

@rag_router.post("")
@rag_router.post("/")
def chat(request: Request, payload: RAGRequest) -> RAGResponse:
    result = run_agent_wrapper(payload.query, payload.thread_id)
    return RAGResponse(
        request_id=getattr(request.state, "request_id", ""),
        answer=result['answer'],
        used_context=[RAGUsedContext(**item) for item in result['used_context']]
    )

api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])

app = FastAPI(title="Amazon Ecommerce RAG API")
app.add_middleware(RequestIDMiddleware)
app.include_router(api_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}