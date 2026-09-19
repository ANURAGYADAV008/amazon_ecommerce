from pydantic import BaseModel, Field
from typing import Optional


class RAGRequest(BaseModel):
    query: str = Field(
        ...,
        description="The query to be used in the RAG pipeline"
    )
    thread_id: str = Field(
        default="default-thread",
        description="The Thread_id"
    )


class RAGUsedContext(BaseModel):
    image_url: str = Field(..., description="The URL of the image of the item")

    price: Optional[float] = Field(
        None,
        description="The price of the item"
    )

    description: str = Field(
        ...,
        description="The description of the item"
    )


class RAGResponse(BaseModel):
    request_id: str = Field(
        ...,
        description="The request ID"
    )

    answer: str = Field(
        ...,
        description="The answer to the query"
    )

    used_context: list[RAGUsedContext] = Field(
        ...,
        description="Information about the items used to answer the query"
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="The Trace ID"
    )


class FeedbackRequest(BaseModel):
    run_id: str = Field(
        ...,
        description="The run ID or trace ID for LangSmith feedback"
    )
    feedback_score: Optional[int] = Field(
        None,
        description="Feedback score (e.g. 1 for thumbs up, 0 for thumbs down)"
    )
    feedback_text: Optional[str] = Field(
        "",
        description="Feedback comment or text"
    )
    feedback_source_type: Optional[str] = Field(
        "api",
        description="Source of feedback"
    )


class FeedbackResponse(BaseModel):
    request_id: str = Field(
        ...,
        description="The request ID"
    )
    status: str = Field(
        default="success",
        description="Status of the feedback submission"
    )