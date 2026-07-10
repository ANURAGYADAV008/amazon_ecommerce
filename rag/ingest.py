"""Ingest the Amazon Electronics sample into Qdrant.

Creates the `Amazon_items_collection-00` collection and upserts one point per
product, with the payload fields the retriever reads in reterivalgeneration.py:
parent_asin, description, average_rating, image_url, price.

Run inside the api container:
    docker compose exec api uv run python -m rag.ingest
"""
import json
import os
import uuid

from openai import OpenAI # type: ignore
from dotenv import load_dotenv  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import Distance, VectorParams, PointStruct  # type: ignore

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "Amazon_items_collection-00"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
DATA_PATH = os.getenv(
    "INGEST_DATA_PATH",
    "data/meta_Electronics_2022_23_with_categeory_rating_100_sample_2000.jsonl",
)
BATCH = 100

client = OpenAI()


def build_text(item: dict) -> str:
    """Text used for the embedding: title + features + description."""
    parts = [item.get("title") or ""]
    parts.extend(item.get("features") or [])
    parts.extend(item.get("description") or [])
    return "\n".join(p for p in parts if p).strip()


def build_description(item: dict) -> str:
    """Human-facing description shown in the UI (falls back to title)."""
    desc = " ".join(item.get("description") or []).strip()
    return desc or (item.get("title") or "")


def first_image(item: dict) -> str:
    for img in item.get("images") or []:
        url = img.get("large") or img.get("hi_res") or img.get("thumb")
        if url:
            return url
    return ""


def load_items(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = build_text(item)
            if not text:
                continue  # nothing to embed
            items.append(item)
    return items


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main() -> None:
    qdrant = QdrantClient(QDRANT_URL)

    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    items = load_items(DATA_PATH)
    print(f"Loaded {len(items)} items with text from {DATA_PATH}")

    total = 0
    for start in range(0, len(items), BATCH):
        chunk = items[start:start + BATCH]
        vectors = embed_batch([build_text(it) for it in chunk])

        points = []
        for item, vector in zip(chunk, vectors):
            asin = item.get("parent_asin") or str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, asin)),
                    vector=vector,
                    payload={
                        "parent_asin": item.get("parent_asin"),
                        "description": build_description(item),
                        "average_rating": item.get("average_rating"),
                        "image_url": first_image(item),
                        "price": item.get("price"),
                    },
                )
            )

        qdrant.upsert(collection_name=COLLECTION, points=points)
        total += len(points)
        print(f"Upserted {total}/{len(items)}")

    count = qdrant.count(collection_name=COLLECTION).count
    print(f"Done. Collection '{COLLECTION}' now has {count} points.")


if __name__ == "__main__":
    main()
