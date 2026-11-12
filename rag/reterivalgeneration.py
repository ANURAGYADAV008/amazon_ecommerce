import os
from openai import OpenAI  # type: ignore
from dotenv import load_dotenv  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import Distance,VectorParams,PointStruct,Filter,FieldCondition,MatchValue,Prefetch,Document,models  # type: ignore
import cohere  # type: ignore
load_dotenv()
client=OpenAI()
COHERE_API_KEY = os.getenv("CO_API_KEY") or os.getenv("COHERE_API_KEY")
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY) if COHERE_API_KEY else None
from langsmith import traceable,get_current_run_tree  # type: ignore
import instructor
import openai
client2=instructor.from_openai(openai.OpenAI())
from pydantic import BaseModel,Field

class RAGUsedContext(BaseModel):
    id:str=Field(description="The ID Of the item used answer the questions")
    description:str=Field(description="Short description of the item used to answer the Question")

class RAGGenerationResponse(BaseModel):
    answer:str=Field(description="The Answer of the Question")
    references:list[RAGUsedContext]=Field(description="List of item used to answer the Question")



QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6335")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

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
def retrieve_data(query, qdrant_client, collection_name=None, k=5, candidate_limit=20):
    if collection_name is None:
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "Amazon-items-collection-02-hybrid-serach")
    query_embedding = get_embedding(query)

    # 1. Stage 1: Broad hybrid retrieval from Qdrant (high recall)
    results = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-model-3-small",
                limit=candidate_limit
            ),
            Prefetch(
                query=Document(
                    text=query,
                    model="qdrant/bm25",
                ),
                using="bm25",
                limit=candidate_limit
            )
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[3, 1])),
        limit=candidate_limit
    )

    candidates = []
    for point in results.points:
        payload = point.payload or {}
        candidates.append({
            "id": payload.get("parent_asin"),
            "description": payload.get("description", ""),
            "rating": payload.get("average_rating"),
            "score": point.score,
            "image_url": payload.get("image_url", "") or payload.get("image", ""),
            "price": payload.get("price"),
        })

    # 2. Stage 2: Cohere Reranking (high precision)
    if cohere_client and candidates:
        docs = [c["description"] for c in candidates]
        try:
            rerank_resp = cohere_client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=docs,
                top_n=min(k, len(candidates))
            )
            top_candidates = [
                {**candidates[item.index], "score": float(item.relevance_score)}
                for item in rerank_resp.results
            ]
        except Exception as e:
            # Graceful fallback if Cohere call fails
            top_candidates = candidates[:k]
    else:
        top_candidates = candidates[:k]

    retrieved_context_ids = [c["id"] for c in top_candidates]
    retrieved_context = [c["description"] for c in top_candidates]
    retrieved_context_rating = [c["rating"] for c in top_candidates]
    similarity_scores = [c["score"] for c in top_candidates]
    retrieved_image_urls = [c["image_url"] for c in top_candidates]
    retrieved_prices = [c["price"] for c in top_candidates]

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
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(processed_context, question):
    prompt = f"""
You are a helpful AI shopping assistant.

Use ONLY the information provided in the retrieved product context below to answer the user's question.

Rules:
- Do not make up information.
- If the answer is not present in the context, reply:
  "I couldn't find that information in the retrieved products."
- Keep your answer concise and helpful.
- Mention product IDs when relevant.

======================
Retrieved Context:
{processed_context}
======================

User Question:
{question}

Answer:
"""

    return prompt

@traceable(
    name="generate_chat",
    run_type="prompt",
    metadata={"ls_provider":"openai","ls_modelname":"gpt-4.1-nano"}
)
def generate_chat(prompt):
    response,raw_response = client2.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            
        ],
        temperature=0.2,
        response_model=RAGGenerationResponse
       
    )
    current_run=get_current_run_tree()
    if current_run and raw_response.usage:
        current_run.metadata["usage_metadata"]={
            "input_token":raw_response.usage.prompt_tokens,
            "output_token":raw_response.usage.completion_tokens,
            "total_token":raw_response.usage.total_tokens

        }

    return response

@traceable(
    name="rag_pipeline"
)
def rag_pipeline(question, qdrant_client=None, top_k=5):
    if qdrant_client is None:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    retrieved_context = retrieve_data(
        question,
        qdrant_client,
        k=top_k
    )

    processed_context = process_context(
        retrieved_context[0],
        retrieved_context[1],
        retrieved_context[2],
        retrieved_context[3]
    )

    prompt = build_prompt(
        processed_context,
        question
    )

    answer = generate_chat(prompt)
    final_result={
        "original_output":answer,
        "answer":answer.answer,
        'references':answer.references,
        "Question":question,
        "reterived_context_ids":retrieved_context[1],
        "reterived_context":retrieved_context[0],
        "similarity_score":retrieved_context[3]
    }

    return final_result

@traceable(
    name="rag_pipeline_wrapper"
)
def rag_pipeline_wrapper(question, topk=5):
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    result = rag_pipeline(question, qdrant_client, topk)

    used_context = []

    for reference in result.get('references', []):
        try:
            scroll_results, _ = qdrant_client.scroll(
                collection_name=os.getenv("QDRANT_COLLECTION_NAME", "Amazon-items-collection-02-hybrid-serach"),
                with_payload=True,
                with_vectors=False,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key='parent_asin', match=MatchValue(value=reference.id))
                    ]
                ),
            )
            if not scroll_results:
                continue
            payload = scroll_results[0].payload or {}
        except Exception:
            continue

        image_url = payload.get('image', payload.get('image_url', ''))
        price = payload.get('price', None)
        
        if image_url:
            used_context.append({
                'image_url': image_url,
                'price': price,
                'description': reference.description,
            })
        
    return {
        'answer': result['answer'],
        'used_context': used_context,
    }