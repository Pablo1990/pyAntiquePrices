"""Vector store abstraction for local comparable search."""

from __future__ import annotations

from typing import Protocol


class VectorStore(Protocol):
    def add(self, item_id: int, embedding: list[float], metadata: dict | None = None) -> None:
        ...

    def search(self, embedding: list[float], top_k: int = 50) -> list[dict]:
        ...

    def delete(self, item_id: int) -> None:
        ...


class InMemoryVectorStore:
    """Simple in-memory vector store for local MVP/testing."""

    def __init__(self) -> None:
        self._items: dict[int, dict] = {}

    def add(self, item_id: int, embedding: list[float], metadata: dict | None = None) -> None:
        self._items[item_id] = {
            "id": item_id,
            "embedding": embedding,
            "metadata": metadata or {},
        }

    def search(self, embedding: list[float], top_k: int = 50) -> list[dict]:
        scored = []
        for item in self._items.values():
            score = cosine_similarity(embedding, item["embedding"])
            scored.append(
                {
                    "id": item["id"],
                    "score": score,
                    "metadata": item["metadata"],
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def delete(self, item_id: int) -> None:
        self._items.pop(item_id, None)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b))
    left = sum(x * x for x in a) ** 0.5
    right = sum(y * y for y in b) ** 0.5
    if left == 0 or right == 0:
        return 0.0
    return numerator / (left * right)
