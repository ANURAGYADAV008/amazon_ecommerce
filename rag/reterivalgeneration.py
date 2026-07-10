import os
from openai import OpenAI  # type: ignore
from dotenv import load_dotenv  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import Distance,VectorParams,PointStruct  # type: ignore
load_dotenv()
client=OpenAI()
from langsmith import traceable,get_current_run_tree  # type: ignore




QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider":"openai","ls_modelname":"text-embedding-3-small"}
)
def get_embedding(text,model="text-embedding-3-small"):
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
def reteriver_data(query, qdrant_client, k):
    query_embedding = get_embedding(query)

    result = qdrant_client.query_points(
        collection_name="Amazon_items_collection-00",
        query=query_embedding,
        limit=k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_rating = []
    retrieved_image_urls = []
    retrieved_prices = []

    for point in result.points:
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
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            
        ],
        temperature=0.2,
        max_tokens=300
    )
    current_run=get_current_run_tree()
    if current_run and response.usage:
        current_run.metadata["usage_metadata"]={
            "input_token":response.usage.prompt_tokens,
            "output_token":response.usage.completion_tokens,
            "total_token":response.usage.total_tokens

        }

    return response.choices[0].message.content

@traceable(
    name="rag_pipeline"
)
def rag_pipeline(question, top_k=5):
    qdrant_client = QdrantClient(QDRANT_URL)

    retrieved_context = reteriver_data(
        question,
        qdrant_client,
        top_k
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

    return answer

@traceable(
    name="rag_pipeline_wrapper"
)
def rag_pipeline_wrapper(question, top_k=5):
    """Run the RAG pipeline and return a dict shaped for the API response:
    {"answer": str, "used_context": [{"image_url", "price", "description"}, ...]}.
    """
    qdrant_client = QdrantClient(QDRANT_URL)

    (
        context,
        ids,
        ratings,
        scores,
        image_urls,
        prices,
    ) = reteriver_data(question, qdrant_client, top_k)

    processed_context = process_context(context, ids, ratings, scores)
    prompt = build_prompt(processed_context, question)
    answer = generate_chat(prompt)

    used_context = [
        {
            "image_url": image_url or "",
            "price": price,
            "description": description or "",
        }
        for image_url, price, description in zip(image_urls, prices, context)
    ]

    return {"answer": answer, "used_context": used_context}