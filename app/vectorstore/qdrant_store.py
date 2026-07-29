"""A thin wrapper over qdrant-client.

Enough to store chunks and search them. The interesting improvements live in the
TODO comments — this is a plain single-vector cosine index and nothing more.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient, models

from ..config import get_settings


class VectorStore:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection

    # --- inspection ---------------------------------------------------------
    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.collection).count

    # --- write --------------------------------------------------------------
    def ensure_collection(self, dim: int, reset: bool = False) -> None:
        """Create the collection sized to the embedding dimension.

        The vector size is fixed at creation, so if you change embedding models you
        must re-ingest (or ingest into a differently named collection).
        """
        if reset and self.exists():
            self.client.delete_collection(self.collection)
        if not self.exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            # TODO(level-3): a payload index on e.g. `page` lets you filter searches
            # (search only the references section, only tables, ...). See
            # client.create_payload_index(...).

    def upsert(self, points: list[models.PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=points)

    # --- read ---------------------------------------------------------------
    def search(self, vector: list[float], top_k: int) -> list[models.ScoredPoint]:
        return self.client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )

    def scroll_all(self) -> list[models.Record]:
        """Fetch every point in the collection (no vector search)."""
        records: list[models.Record] = []
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            records.extend(batch)
            if next_offset is None:
                break
            offset = next_offset
        return records


@lru_cache
def get_store() -> VectorStore:
    s = get_settings()
    return VectorStore(s.qdrant_url, s.qdrant_collection)
