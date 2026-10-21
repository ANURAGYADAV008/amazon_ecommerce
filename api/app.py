import logging
import os
from fastapi import APIRouter, FastAPI, Request, HTTPException  # pyright: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware  # pyright: ignore[reportMissingImports]
from .models import RAGRequest,RAGResponse,RAGUsedContext,FeedbackRequest,FeedbackResponse
from .middleware import RequestIDMiddleware
from .Agent.graph import rag_agent_wrapper
from .processors import submit_feedback
from slowapi.util import get_remote_address
from slowapi import Limiter
from slowapi.middleware import SlowAPIASGIMiddleware
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

rag_router = APIRouter()
feedback_router = APIRouter()

limiter=Limiter(key_func=get_remote_address)




@rag_router.post("/")
@limiter.limit("2/minute")
def chat(request: Request, payload: RAGRequest) -> RAGResponse:
    try:
        result = rag_agent_wrapper(payload.query, payload.thread_id)
        return RAGResponse(
            request_id=getattr(request.state, "request_id", ""),
            answer=result['answer'],
            used_context=[RAGUsedContext(**item) for item in result['used_context']],
            trace_id=result.get('trace_id'),
            shopping_cart=result.get('shopping_cart', []),
        )
    except Exception as e:
        logger.exception("Failed to process RAG chat request: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error processing request: {str(e)}"
        )


@feedback_router.post("/")
def send_feedback(request: Request, payload: FeedbackRequest) -> FeedbackResponse:
    try:
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
    except Exception as e:
        logger.exception("Failed to submit feedback: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error submitting feedback: {str(e)}"
        )


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
api_router.include_router(feedback_router, prefix="/submit_feedback", tags=["feedback"])


def _cors_origins() -> list[str]:
    """Origins allowed to call the API from a browser.

    Set CORS_ORIGINS to a comma-separated list to override the defaults.
    Vercel deployments (production and preview) are matched by regex below.
    """
    configured = os.getenv("CORS_ORIGINS", "")
    origins = [
        origin.strip().rstrip("/")
        for origin in configured.split(",")
        if origin.strip()
    ]
    if origins:
        return origins

    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]


app = FastAPI(title="Amazon Ecommerce RAG API")
app.state.limiter=limiter

app.add_exception_handler(
    RateLimitExceeded,
     # pyrefly: ignore[bad-argument-count]
    lambda request,exc:JSONResponse(
        status_code=429,
        content={"details":"Too many request",},

    )
    
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"https://([a-z0-9-]+\.)*vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.include_router(api_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}