import logging
from fastapi import APIRouter, FastAPI, Request  # pyright: ignore[reportMissingImports]
from .models import RAGRequest,RAGResponse,RAGUsedContext,FeedbackRequest,FeedbackResponse
from .middleware import RequestIDMiddleware
from .Agent.graph import rag_agent_wrapper
from .processors import submit_feedback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()
feedback_router = APIRouter()


@rag_router.post("")
@rag_router.post("/")
def chat(request: Request, payload: RAGRequest) -> RAGResponse:
    result = rag_agent_wrapper(payload.query, payload.thread_id)
    return RAGResponse(
        request_id=getattr(request.state, "request_id", ""),
        answer=result['answer'],
        used_context=[RAGUsedContext(**item) for item in result['used_context']],
        trace_id=result.get('trace_id'),
    )


@feedback_router.post("")
@feedback_router.post("/")
def send_feedback(request: Request, payload: FeedbackRequest) -> FeedbackResponse:
    submit_feedback(
        run_id=payload.run_id,
        feedback_score=payload.feedback_score,
        feedback_text=payload.feedback_text,
        feedback_source_type=payload.feedback_source_type,
    )
    return FeedbackResponse(
        request_id=getattr(request.state, "request_id", ""),
        status="success",
    )


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
api_router.include_router(feedback_router, prefix="/submit_feedback", tags=["feedback"])

app = FastAPI(title="Amazon Ecommerce RAG API")
app.add_middleware(RequestIDMiddleware)
app.include_router(api_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}