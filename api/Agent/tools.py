import os
from dotenv import load_dotenv  # type: ignore
from openai import OpenAI  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import Prefetch, Document, models  # type: ignore
from langsmith import traceable, get_current_run_tree  # type: ignore
import instructor
import openai
import cohere  # type: ignore

load_dotenv()
client = OpenAI()
client2 = instructor.from_openai(openai.OpenAI())
CO_API_KEY = os.getenv("CO_API_KEY")


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_modelname": "text-embedding-3-small"}
)
def get_embedding(text, model="text-embedding-3-small"):
    """Creates embedding for text using OpenAI's text-embedding-3-small model."""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_token": response.usage.prompt_tokens,
            "output_token": response.usage.total_tokens,
            "total_token": response.usage.total_tokens
        }
    return response.data[0].embedding


@traceable(
    name="reteriver_data",
    run_type="retriever"
)
def retrieve_data(query, qdrant_client, collection_name=None, k=5):
    if collection_name is None:
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "Amazon-items-collection-02-hybrid-serach")
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-model-3-small",
                limit=20
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25",
                ),
                using="bm25",
                limit=20
            )
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[3, 1])),
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_rating = []
    retrieved_image_urls = []
    retrieved_prices = []

    for point in results.points:
        payload = point.payload or {}
        retrieved_context_ids.append(payload.get("parent_asin"))
        retrieved_context.append(payload.get("description", ""))
        retrieved_context_rating.append(payload.get("average_rating"))
        similarity_scores.append(point.score)
        retrieved_image_urls.append(payload.get("image", payload.get("image_url", "")))
        retrieved_prices.append(payload.get("price"))

    return (
        retrieved_context,
        retrieved_context_ids,
        retrieved_context_rating,
        similarity_scores,
        retrieved_image_urls,
        retrieved_prices
    )


@traceable(
    name="process_context",
    run_type="tool"
)
def process_context(context, ids, ratings, scores):
    formatted_context = ""

    for id, chunk, rating, score in zip(ids, context, ratings, scores):
        formatted_context += (
            f"- ID: {id}\n"
            f"  Description: {chunk}\n"
            f"  Rating: {rating}\n"
            f"  Similarity Score: {score:.4f}\n\n"
        )

    return formatted_context


@traceable(
    name="retriever_rerank",
    run_type="retriever"
)
def retrivererank(query, context, top_n=None):
    """Reranks retrieved items using Cohere Rerank."""
    retrieved_context = context[0]
    retrieved_context_ids = context[1]
    retrieved_context_rating = context[2]
    similarity_scores = context[3]
    retrieved_image_urls = context[4] if len(context) > 4 else []
    retrieved_prices = context[5] if len(context) > 5 else []

    if not retrieved_context or not CO_API_KEY:
        return context

    cohere_client = cohere.ClientV2(CO_API_KEY)
    response_rerank = cohere_client.rerank(
        model="rerank-v4.0-fast",
        query=query,
        documents=retrieved_context,
        top_n=top_n or len(retrieved_context)
    )

    reranked_context = []
    reranked_context_ids = []
    reranked_context_rating = []
    reranked_scores = []
    reranked_image_urls = []
    reranked_prices = []

    for item in response_rerank.results:
        idx = item.index
        reranked_context.append(retrieved_context[idx])
        reranked_context_ids.append(retrieved_context_ids[idx])
        reranked_context_rating.append(retrieved_context_rating[idx])
        reranked_scores.append(item.relevance_score)
        if retrieved_image_urls:
            reranked_image_urls.append(retrieved_image_urls[idx])
        if retrieved_prices:
            reranked_prices.append(retrieved_prices[idx])

    return (
        reranked_context,
        reranked_context_ids,
        reranked_context_rating,
        reranked_scores,
        reranked_image_urls,
        reranked_prices
    )


def get_formatted_context(query: str, top_k: int = 6) -> str:
    """ Get the top k context each representing an inventory 
      item for a given query

    Args:
       query: The Query to get The top k context for
       top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
       A string of the top k context chunks with IDs and the average ratings prepending 
       each chunk, each representing an inventory item
    """
    q_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6335"))
    context = retrieve_data(query=query, qdrant_client=q_client, k=6)
    ##context = retrivererank(query=query, context=context, top_n=7)
    formatted_context = process_context(
        context=context[0],
        ids=context[1],
        ratings=context[2],
        scores=context[3]
    )
    return formatted_context