import logging
from typing import Optional
from langsmith import Client

logger = logging.getLogger(__name__)

try:
    client = Client()
except Exception as e:
    logger.warning("Could not initialize LangSmith Client: %s", e)
    client = None


def submit_feedback(
    run_id: str,
    feedback_score: Optional[int] = None,
    feedback_text: Optional[str] = "",
    feedback_source_type: Optional[str] = "api",
) -> None:
    if not client:
        logger.warning("LangSmith client is not initialized; skipping feedback submission.")
        return

    source: str = feedback_source_type if feedback_source_type is not None else "api"

    try:
        if feedback_score is not None:
            client.create_feedback(
                run_id=run_id,
                key="thumbs",
                score=feedback_score,
                feedback_source_type=source,
            )

        if feedback_text:
            client.create_feedback(
                run_id=run_id,
                key="comment",
                comment=feedback_text,
                feedback_source_type=source,
            )
    except Exception as exc:
        logger.error("Failed to submit feedback to LangSmith for run_id %s: %s", run_id, exc)