import threading
import uuid
from typing import Any, Dict, List, Optional

from ollama import Client as OllamaClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .config import MemoryConfig

POINT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def stable_point_id(text_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, text_id))

VECTOR_FIELDS = {
    "content": "content",
    "type": "type",
    "priority": "priority",
    "scene_name": "scene_name",
    "session_id": "session_id",
}

_LOCK = threading.Lock()


class MemoryVectors:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._embed_lock = threading.Lock()
        self._ollama = OllamaClient(host="http://localhost:11434")
        self._qdrant_l0 = self._make_client(config.l0_collection)
        self._qdrant_l1 = self._make_client(config.l1_collection)

    def _make_client(self, collection: str) -> QdrantClient:
        client = QdrantClient(url=self.config.qdrant_url, timeout=10)
        try:
            if not client.collection_exists(collection):
                client.create_collection(
                    collection_name=collection,
                    vectors_config=qmodels.VectorParams(
                        size=self.config.embedding.dimensions,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
        except Exception as e:
            print(f"Warning: Qdrant collection '{collection}' unavailable: {e}")
        return client

    def _delete_collection(self, client: QdrantClient, collection: str) -> None:
        try:
            if client.collection_exists(collection):
                client.delete_collection(collection)
            client.create_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(
                    size=self.config.embedding.dimensions,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        except Exception as e:
            print(f"Warning: could not wipe Qdrant collection '{collection}': {e}")

    def embed(self, text: str) -> List[float]:
        with self._embed_lock:
            try:
                response = self._ollama.embed(
                    model=self.config.embedding.id,
                    input=text,
                    dimensions=self.config.embedding.dimensions,
                )
                embeddings = getattr(response, "embeddings", None)
                if embeddings:
                    return embeddings[0]
            except Exception as e:
                print(f"Warning: embedding failed: {e}")
        return []

    def get_embedding_or_none(self, text: str) -> Optional[List[float]]:
        vec = self.embed(text)
        return vec if len(vec) == self.config.embedding.dimensions else None

    # ── L0 ──

    def upsert_l0(self, point_id: str, text: str, payload: Dict[str, Any]) -> bool:
        vec = self.get_embedding_or_none(text)
        if not vec:
            return False
        try:
            self._qdrant_l0.upsert(
                collection_name=self.config.l0_collection,
                points=[
                    qmodels.PointStruct(
                        id=stable_point_id(point_id), vector=vec, payload=payload
                    )
                ],
            )
            return True
        except Exception as e:
            print(f"Warning: L0 upsert failed: {e}")
            return False

    def search_l0(
        self, query: str, limit: int, score_threshold: float
    ) -> List[Dict[str, Any]]:
        vec = self.get_embedding_or_none(query)
        if not vec:
            return []
        try:
            results = self._qdrant_l0.query_points(
                collection_name=self.config.l0_collection,
                query=vec,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            ).points
            return [
                {"id": str(p.id), "score": p.score, "payload": p.payload or {}}
                for p in results
            ]
        except Exception as e:
            print(f"Warning: L0 search failed: {e}")
            return []

    # ── L1 ──

    def upsert_l1(self, point_id: str, text: str, payload: Dict[str, Any]) -> bool:
        vec = self.get_embedding_or_none(text)
        if not vec:
            return False
        try:
            self._qdrant_l1.upsert(
                collection_name=self.config.l1_collection,
                points=[
                    qmodels.PointStruct(
                        id=stable_point_id(point_id), vector=vec, payload=payload
                    )
                ],
            )
            return True
        except Exception as e:
            print(f"Warning: L1 upsert failed: {e}")
            return False

    def search_l1(
        self, query: str, limit: int, score_threshold: float
    ) -> List[Dict[str, Any]]:
        vec = self.get_embedding_or_none(query)
        if not vec:
            return []
        return self.search_l1_with_vector(vec, limit, score_threshold)

    def delete_l1(self, record_id: str) -> bool:
        try:
            self._qdrant_l1.delete(
                collection_name=self.config.l1_collection,
                points_selector=[stable_point_id(record_id)],
            )
            return True
        except Exception as e:
            print(f"Warning: L1 delete failed: {e}")
            return False

    def wipe_l0(self) -> None:
        self._delete_collection(self._qdrant_l0, self.config.l0_collection)

    def wipe_l1(self) -> None:
        self._delete_collection(self._qdrant_l1, self.config.l1_collection)

    def search_l1_with_vector(
        self, vec: List[float], limit: int, score_threshold: float
    ) -> List[Dict[str, Any]]:
        try:
            results = self._qdrant_l1.query_points(
                collection_name=self.config.l1_collection,
                query=vec,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            ).points
            return [
                {"id": str(p.id), "score": p.score, "payload": p.payload or {}}
                for p in results
            ]
        except Exception as e:
            print(f"Warning: L1 search failed: {e}")
            return []