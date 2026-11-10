from unittest.mock import Mock

import pytest

from docsense_rag.ingestion.chunk import Chunk
from docsense_rag.ingestion.segment import TextSegment
from docsense_rag.ingestion.service import IngestionService


@pytest.fixture
def segments() -> list[TextSegment]:
    return [
        TextSegment(
            text="O prazo de entrega é de 30 dias.",
            page_number=1,
        ),
        TextSegment(
            text="O pagamento ocorrerá após a entrega.",
            page_number=2,
        ),
    ]


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(
            document_id="edital-001",
            chunk_index=0,
            text="O prazo de entrega é de 30 dias.",
            page_start=1,
            page_end=1,
            token_count=8,
        ),
        Chunk(
            document_id="edital-001",
            chunk_index=1,
            text="O pagamento ocorrerá após a entrega.",
            page_start=2,
            page_end=2,
            token_count=7,
        ),
    ]


def test_ingest_coordinates_chunking_embedding_and_storage(
    segments: list[TextSegment],
    chunks: list[Chunk],
) -> None:
    chunker = Mock()
    chunker.chunk.return_value = chunks

    embedder = Mock()
    embedder.embed_documents.return_value = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    vector_store = Mock()

    service = IngestionService(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
    )

    result = service.ingest("edital-001", segments)

    assert result == chunks

    chunker.chunk.assert_called_once_with("edital-001", segments)
    embedder.embed_documents.assert_called_once_with([chunk.text for chunk in chunks])
    vector_store.replace_document.assert_called_once_with(
        document_id="edital-001",
        chunks=chunks,
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )


def test_ingest_cancels_when_no_chunks_are_generated(
    segments: list[TextSegment],
) -> None:
    chunker = Mock()
    chunker.chunk.return_value = []

    embedder = Mock()
    vector_store = Mock()

    service = IngestionService(chunker, embedder, vector_store)

    with pytest.raises(ValueError, match="não gerou chunks"):
        service.ingest("edital-001", segments)

    embedder.embed_documents.assert_not_called()
    vector_store.replace_document.assert_not_called()


def test_ingest_rejects_inconsistent_embedding_count(
    segments: list[TextSegment],
    chunks: list[Chunk],
) -> None:
    chunker = Mock()
    chunker.chunk.return_value = chunks

    embedder = Mock()
    embedder.embed_documents.return_value = [[1.0, 0.0, 0.0]]

    vector_store = Mock()

    service = IngestionService(chunker, embedder, vector_store)

    with pytest.raises(ValueError, match="quantidade incorreta"):
        service.ingest("edital-001", segments)

    vector_store.replace_document.assert_not_called()
