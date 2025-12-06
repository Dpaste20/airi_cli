import os

from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.tools import tool
from agno.vectordb.qdrant import Qdrant

vector_db = None


def get_vector_db():
    """Lazy load the vector db connection."""
    global vector_db
    if vector_db is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        vector_db = Qdrant(
            collection="airi_knowledge",
            url=qdrant_url,
            embedder=OllamaEmbedder(id="embeddinggemma:latest", dimensions=768),
        )
    return vector_db


@tool
def rag_search(query: str) -> str:
    """Search the knowledge base for relevant information."""
    try:
        db = get_vector_db()
        results = db.search(query=query, limit=10)
        return str(results)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"
