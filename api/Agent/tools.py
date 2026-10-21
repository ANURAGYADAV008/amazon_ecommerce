from anyio import run
from distro import name
import os
from dotenv import load_dotenv  # type: ignore
from openai import OpenAI  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import Prefetch, Document, models ,Filter,MatchValue,FieldCondition,FusionQuery # type: ignore
from langsmith import traceable, get_current_run_tree  # type: ignore
import instructor
import openai
import cohere  # type: ignore
import numpy as np
from psycopg2.extras import RealDictCursor
import psycopg2

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
        collection_name = os.getenv("QDRANT_COLLECTION_NAME")
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


def get_formatted_context_with_metadata(query: str, top_k: int = 6) -> tuple[str, dict]:
    """Get the top k context each representing an inventory item for a given query,
    along with a dictionary mapping product IDs to their full metadata.
    """
    q_client = QdrantClient(url=os.getenv("QDRANT_URL"),api_key=os.getenv("QDRANT_API_KEY"))
    context = retrieve_data(query=query, qdrant_client=q_client, k=top_k)
    ##context = retrivererank(query=query, context=context, top_n=7)
    formatted_context = process_context(
        context=context[0],
        ids=context[1],
        ratings=context[2],
        scores=context[3]
    )

    metadata = {}
    descriptions = context[0] if len(context) > 0 else []
    ids = context[1] if len(context) > 1 else []
    ratings = context[2] if len(context) > 2 else []
    image_urls = context[4] if len(context) > 4 else []
    prices = context[5] if len(context) > 5 else []

    for i, asin in enumerate(ids):
        if asin:
            metadata[asin] = {
                "id": asin,
                "description": descriptions[i] if i < len(descriptions) else "",
                "rating": float(ratings[i]) if i < len(ratings) and ratings[i] is not None else None,
                "image_url": image_urls[i] if i < len(image_urls) else "",
                "price": float(prices[i]) if i < len(prices) and prices[i] is not None else None,
            }

    return formatted_context, metadata


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
    formatted_context, _ = get_formatted_context_with_metadata(query=query, top_k=top_k)
    return formatted_context

_shopping_cart_table_initialized = False

def get_db_connection():
    db_uri = os.getenv("DATABASE_URL") or os.getenv("DB_URI")
    if not db_uri:
        raise ValueError("DATABASE_URL or DB_URI environment variable is not configured.")
    conn = psycopg2.connect(db_uri)
    conn.autocommit = True

    global _shopping_cart_table_initialized
    if not _shopping_cart_table_initialized:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE SCHEMA IF NOT EXISTS shopping_carts;
                CREATE TABLE IF NOT EXISTS shopping_carts.shopping_cart_items (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    shopping_cart_id VARCHAR(255) NOT NULL,
                    product_id VARCHAR(255) NOT NULL,
                    price NUMERIC(10, 2),
                    quantity INTEGER NOT NULL DEFAULT 1,
                    currency VARCHAR(10) DEFAULT 'USD',
                    product_image_url TEXT,
                    added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
        _shopping_cart_table_initialized = True
    return conn

## add To shopping Cart Tool

@traceable(
    name="add_to_shopping_cart",
    run_type="tool"
)
def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> str:

    """Add a list of provided items to the shopping cart.
    
    Args:
        items: A list of items to add to the shopping cart. Each item is a dictionary with the following keys: product_id, quantity.
        user_id: The id of the user to add the items to the shopping cart.
        cart_id: The id of the shopping cart to add the items to.
        
    Returns:
        A list of the items added to the shopping cart.
    """

    conn = get_db_connection()

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        
        for item in items:
            product_id = item['product_id']
            quantity = item['quantity']

            qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"),api_key=os.getenv("QDRANT_API_KEY"))

            dummy_vector = np.zeros(1536).tolist()
            payload = qdrant_client.query_points(
                collection_name="Amazon-items-collection-02-hybrid-serach",
                prefetch=[
                    Prefetch(
                        query=dummy_vector,
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="parent_asin",
                                    match=MatchValue(value=product_id)
                                )
                            ]
                    ),
                        using="text-embedding-model-3-small",
                        limit=20
                    )
                ],
                query=FusionQuery(fusion="rrf"),
                limit=1,
            )
            if not payload.points:
                print(f"Product {product_id} not found in Qdrant; skipping.")
                continue
            payload = payload.points[0].payload or {}

            product_image_url = payload.get("image")
            price = payload.get("price")
            currency = 'USD'
        
            # Check if item already exists
            check_query = """
                SELECT id, quantity, price 
                FROM shopping_carts.shopping_cart_items 
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
            cursor.execute(check_query, (user_id, cart_id, product_id))
            existing_item = cursor.fetchone()
            
            if existing_item:
                # Update existing item
                new_quantity = existing_item['quantity'] + quantity
                
                update_query = """
                    UPDATE shopping_carts.shopping_cart_items 
                    SET 
                        quantity = %s,
                        price = %s,
                        currency = %s,
                        product_image_url = COALESCE(%s, product_image_url)
                    WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                    RETURNING id, quantity, price
                """
                
                cursor.execute(update_query, (new_quantity, price, currency, product_image_url, user_id, cart_id, product_id))
            
            else:
                # Insert new item
                insert_query = """
                    INSERT INTO shopping_carts.shopping_cart_items (
                        user_id, shopping_cart_id, product_id,
                        price, quantity, currency, product_image_url
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, quantity, price
                """
                
                cursor.execute(insert_query, (user_id, cart_id, product_id, price, quantity, currency, product_image_url))
            
    return f"Added {items} to the shopping cart."


## Get Shopping Cart Item
@traceable(
    name="Get Shopping Cart Item",
    run_type='tool'
    
)
def get_shopping_cart(user_id:str,cart_id:str)->list[dict]:
    """
    Reterive all items in a user's shopping cart
     
    Args:
      user_id:User ID
      cart_id:Cart identifier

    Return:
      List of dictionaries conatining cart items
    """
    conn = get_db_connection()

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
      query="""
        SELECT 
           product_id,price,quantity,
           currency,product_image_url,
           (price * quantity) as total_price
          FROM shopping_carts.shopping_cart_items
          WHERE user_id=%s AND shopping_cart_id=%s
          ORDER BY added_at DESC
      """
      cursor.execute(query,(user_id,cart_id))
      return [dict(row) for row in cursor.fetchall()]


## Remove The Item From cart
@traceable(
    name="Remove from Cart",
    run_type="tool"
)
def remove_from_cart(product_id:str,user_id:str,cart_id:str)->str:
    """Remove an item Completely from the Shopping Cart,
    Args:
      user_id:user ID
      product_id:Product Id to remove
      cart_id: Cart Identifier

    Return:
      True if item was removed ,False if item wasn't found

    """
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
      query="""
        DELETE FROM shopping_carts.shopping_cart_items
        WHERE user_id =%s AND shopping_cart_id=%s AND product_id=%s
      """
      cursor.execute(query,(user_id,cart_id,product_id))

    # pyrefly: ignore[bad-return]
    return cursor.rowcount>0
