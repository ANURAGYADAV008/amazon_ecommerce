import os
from openai import OpenAI  # type: ignore
from dotenv import load_dotenv  # type: ignore
from qdrant_client import QdrantClient , # type: ignore
from qdrant_client.models import Distance,VectorParams,PointStruct,Filter,FieldCondition,MatchValue,Prefetch,Document,models  # type: ignore
load_dotenv()
client=OpenAI()
from langsmith import traceable,get_current_run_tree  # type: ignore
import instructor
import openai
client2=instructor.from_openai(openai.OpenAI())
from pydantic import BaseModel,Field

@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider":"openai","ls_modelname":"text-embedding-3-small"}
)
def get_embedding(text,model="text-embedding-3-small"):
    """Creates embedding for text using OpenAI's text-embedding-3-small model.
    """
    response=client.embeddings.create(
        model=model,
        input=text
    )
    current_run=get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"]={
            "input_token":response.usage.prompt_tokens,
            "output_token":response.usage.total_tokens

        }
    return response.data[0].embedding

@traceable(
    name="reteriver_data",
    run_type="retriever"
)
def retrieve_data(query, qdrant_client, collection_name='amazon-items-collection-01-hybrid-search', k=5):
    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
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
        query=models.RrfQuery(rrf=models.Rrf(weights=[3,1])),
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
        retrieved_image_urls.append(payload.get("image_url", ""))
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


def get_formatted_context(query: str, top_k: int = 5) -> str:
    """ Get the top k context each representing an inventory 
      item for a given query

    Args:
       query: The Query to get The top k context for
       top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
       A string of the top k context chunks with IDs and the average ratings prepending 
       each chunk, each representing an inventory item
    """
    q_client = QdrantClient(url="http://localhost:6333/")
    context = retrieve_data(query=query, qdrant_client=q_client, k=top_k)
    formatted_context = process_context(
        context=context[0],
        ids=context[1],
        ratings=context[2],
        scores=context[3]
    )
    return formatted_context