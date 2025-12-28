import os
import time
import requests
import xml.etree.ElementTree as ET
import trafilatura
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import cohere
from cohere.errors import GatewayTimeoutError
from dotenv import load_dotenv

load_dotenv()

SITEMAP_URL = "https://areeba-binteali.github.io/Physical-AI-Textbook/sitemap.xml"
COLLECTION_NAME = "physical_ai_textbook"

cohere_client = cohere.Client(os.getenv("COHERE_API"))
EMBED_MODEL = "embed-english-v3.0"

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API")
)


def get_all_urls(sitemap_url):
    xml = requests.get(sitemap_url, timeout=15).text
    root = ET.fromstring(xml)

    urls = []
    for child in root:
        loc = child.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
        if loc is not None:
            urls.append(loc.text)

    print(f"\nFOUND {len(urls)} URLS")
    return urls


def extract_text_from_url(url):
    html = requests.get(url, timeout=15).text
    text = trafilatura.extract(html)

    if not text:
        print("⚠️ No text extracted:", url)

    return text


def chunk_text(text, max_chars=700):
    chunks = []
    text = text.strip()[:200_000]

    while len(text) > max_chars:
        split = text.rfind(". ", 0, max_chars)
        if split < 200:
            split = max_chars
        chunks.append(text[:split].strip())
        text = text[split:].strip()

    if text:
        chunks.append(text)

    return chunks


def embed(text, retries=3):
    for i in range(retries):
        try:
            res = cohere_client.embed(
                model=EMBED_MODEL,
                input_type="search_query",
                texts=[text],
            )
            return res.embeddings[0]

        except GatewayTimeoutError:
            print(f"⏳ Cohere timeout → retry {i + 1}")
            time.sleep(5)

    raise RuntimeError("❌ Embedding failed after retries")


def create_collection():
    print("\nCreating Qdrant collection...")
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )


def save_chunk(chunk, chunk_id, url):
    vector = embed(chunk)

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "url": url,
                    "text": chunk,
                    "chunk_id": chunk_id
                }
            )
        ]
    )


def ingest_book():
    urls = get_all_urls(SITEMAP_URL)
    create_collection()

    chunk_id = 1

    for url in urls:
        print("\nProcessing:", url)
        text = extract_text_from_url(url)
        if not text:
            continue

        for chunk in chunk_text(text):
            save_chunk(chunk, chunk_id, url)
            print(f"✅ Saved chunk {chunk_id}")
            chunk_id += 1
            time.sleep(2.8)   # 🔥 throttle API

    print("\n✔️ DONE. Total chunks:", chunk_id - 1)


if __name__ == "__main__":
    ingest_book()
