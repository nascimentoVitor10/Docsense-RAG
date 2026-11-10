import pytest
from qdrant_client import QdrantClient

from docsense_rag.ingestion.chunk import Chunk
from docsense_rag.storage.qdrant_store import QdrantVectorStore


@pytest.fixture
def store():
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_chunks",
        vector_size=3,
    )

    yield vector_store

    client.close()


def test_replace_document_removes_obsolete_chunks(
    store: QdrantVectorStore,
) -> None:
    original_chunks = [
        Chunk("edital-001", 0, "Trecho um.", 1, 1, 2),
        Chunk("edital-001", 1, "Trecho dois.", 2, 2, 2),
        Chunk("edital-001", 2, "Trecho três.", 3, 3, 2),
    ]

    store.upsert_chunks(original_chunks, [[1.0, 0.0, 0.0]] * 3)

    replacement = [
        Chunk("edital-001", 0, "Trecho atualizado.", 1, 1, 2),
    ]

    store.replace_document(
        "edital-001",
        replacement,
        [[1.0, 0.0, 0.0]],
    )

    results = store.search([1.0, 0.0, 0.0], limit=10)

    assert len(results) == 1
    assert results[0].payload is not None
    assert results[0].payload["text"] == "Trecho atualizado."


def test_replace_document_with_empty_chunks_removes_document(
    store: QdrantVectorStore,
) -> None:
    chunk = Chunk("edital-001", 0, "Trecho original.", 1, 1, 2)

    store.upsert_chunks([chunk], [[1.0, 0.0, 0.0]])
    store.replace_document("edital-001", [], [])

    results = store.search([1.0, 0.0, 0.0], limit=10)

    assert results == []


def test_replace_document_rejects_chunks_from_another_document(
    store: QdrantVectorStore,
) -> None:
    chunk = Chunk("edital-002", 0, "Trecho.", 1, 1, 1)

    with pytest.raises(ValueError, match="pertencer ao documento"):
        store.replace_document(
            "edital-001",
            [chunk],
            [[1.0, 0.0, 0.0]],
        )
