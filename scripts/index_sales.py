#!/usr/bin/env python3
"""Index historical sales with text embeddings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyantique_prices.config import settings
from pyantique_prices.data.database import create_tables, get_engine, get_session_factory
from pyantique_prices.data.models import HistoricalSale
from pyantique_prices.embeddings.ollama_embeddings import OllamaTextEmbedder
from pyantique_prices.retrieval.vector_store import InMemoryVectorStore


def build_text_document(sale: HistoricalSale) -> str:
    parts = filter(
        None,
        [
            sale.object_type,
            sale.manufacturer,
            sale.artist,
            sale.period,
            sale.material,
            sale.technique,
            sale.country,
            sale.description,
            sale.condition,
        ],
    )
    return " ".join(parts)


def main():
    engine = get_engine(settings.database_url)
    create_tables(engine)
    session_factory = get_session_factory(engine)
    embedder = OllamaTextEmbedder(
        host=settings.ollama_host,
        model=settings.ollama_embed_model,
    )
    vector_store = InMemoryVectorStore()

    with session_factory() as session:
        sales = (
            session.query(HistoricalSale)
            .filter(HistoricalSale.text_embedding.is_(None))
            .all()
        )
        print(f"Indexing {len(sales)} sales...")
        indexed = 0
        for sale in sales:
            document = build_text_document(sale)
            if not document.strip():
                continue
            try:
                embedding = embedder.embed(document)
                sale.text_embedding = embedding
                vector_store.add(
                    item_id=sale.id,
                    embedding=embedding,
                    metadata={"title": sale.title},
                )
                indexed += 1
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to embed sale {sale.id}: {exc}")
        session.commit()
    print(f"Stored vectors for {indexed} sales.")
    print("Indexing complete.")


if __name__ == "__main__":
    main()
