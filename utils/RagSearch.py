import os

from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.tools import tool
from agno.vectordb.qdrant import Qdrant

qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
vector_db = Qdrant(
    collection="airi_knowledge",
    url=qdrant_url,
    embedder=OllamaEmbedder(id="nomic-embed-text:v1.5", dimensions=768),
)


@tool
def rag_search(query: str) -> str:
    """Search the knowledge base for relevant information."""
    results = vector_db.search(query=query, limit=10)
    return str(results)
