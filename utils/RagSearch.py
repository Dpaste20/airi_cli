import hashlib
import json
import os
from typing import Dict, Optional

from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.tools import tool
from agno.vectordb.qdrant import Qdrant

_knowledge_base: Optional[Knowledge] = None
STATE_FILE = "tmp/rag_state.json"


def get_knowledge_base() -> Optional[Knowledge]:
    return _knowledge_base


def get_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except FileNotFoundError:
        return ""


def load_ingestion_state() -> Dict[str, str]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load RAG state: {e}")
    return {}


def save_ingestion_state(state: Dict[str, str]):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save RAG state: {e}")


async def initialize_rag() -> Knowledge:
    global _knowledge_base

    print("Initializing RAG System...")

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = "airi_knowledge"

    vector_db = Qdrant(
        collection=collection_name,
        url=qdrant_url,
        embedder=OllamaEmbedder(id="embeddinggemma:latest", dimensions=768),
    )

    _knowledge_base = Knowledge(vector_db=vector_db)

    try:
        if vector_db.client.collection_exists(collection_name):
            info = vector_db.client.get_collection(collection_name)
            print(f"RAG Status: Connected. DB contains {info.points_count} vectors.")
    except Exception as e:
        print(f"Qdrant connection check skipped: {e}")

    documents = [
        {"path": "tmp/test_sample.pdf", "metadata": {"subject": "PS", "batch": 2026}},
        {"path": "tmp/test_sample2.pdf", "metadata": {"subject": "DP", "batch": 2026}},
    ]

    ingestion_state = load_ingestion_state()
    state_changed = False

    for doc in documents:
        path = doc["path"]
        meta = doc["metadata"]

        if os.path.exists(path):
            current_hash = get_file_hash(path)

            if path in ingestion_state and ingestion_state[path] == current_hash:
                print(f"Skipping {path} (Already up to date)")
                continue

            print(f"Ingesting {path}...")
            await _knowledge_base.add_content_async(path=path, metadata=meta)

            ingestion_state[path] = current_hash
            state_changed = True
        else:
            print(f"RAG Warning: File not found at {path}")

    if state_changed:
        save_ingestion_state(ingestion_state)
        print("RAG Ingestion complete & state updated.")
    else:
        print("RAG System ready (No new files to ingest).")

    return _knowledge_base


@tool
def rag_search_tool(query: str) -> str:
    kb = get_knowledge_base()
    if not kb:
        return "Error: Knowledge base not initialized."

    try:
        results = kb.vector_db.search(query=query, limit=3)
        return str(results)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"
