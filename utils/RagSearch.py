import os
from typing import Optional

from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.tools import tool
from agno.vectordb.qdrant import Qdrant

_knowledge_base: Optional[Knowledge] = None


def get_knowledge_base() -> Optional[Knowledge]:
    return _knowledge_base


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
            print(f"RAG Status: {info.points_count} existing vectors.")
    except Exception as e:
        print(f"Qdrant connection check skipped: {e}")

    documents = [
        {"path": "tmp/test_sample.pdf", "metadata": {"subject": "PS", "batch": 2026}},
        {"path": "tmp/test_sample2.pdf", "metadata": {"subject": "DP", "batch": 2026}},
    ]

    for doc in documents:
        path = doc["path"]
        meta = doc["metadata"]

        if os.path.exists(path):
            print(f"Ingesting {path}...")
            await _knowledge_base.add_content_async(path=path, metadata=meta)
        else:
            print(f"RAG Warning: File not found at {path}")

    return _knowledge_base


@tool
def rag_search_tool(query: str) -> str:
    kb = get_knowledge_base()
    if not kb:
        return "Error: Knowledge base not initialized."

    try:
        results = kb.vector_db.search(query=query, limit=5)
        return str(results)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"
