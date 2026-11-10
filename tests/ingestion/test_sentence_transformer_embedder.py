from unittest.mock import Mock

import numpy as np
import pytest

from docsense_rag.ingestion.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)


@pytest.fixture
def fake_model() -> Mock:
    model = Mock()
    model.encode.side_effect = lambda texts, **kwargs: np.array(
        [[1.0, 0.0] for _ in texts]
    )
    return model


def test_embed_documents_adds_passage_prefix(fake_model: Mock) -> None:
    embedder = SentenceTransformerEmbedder(model=fake_model)

    vectors = embedder.embed_documents(["edital de contratação"])

    assert vectors == [[1.0, 0.0]]
    fake_model.encode.assert_called_once_with(
        ["passage: edital de contratação"],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def test_embed_query_adds_query_prefix(fake_model: Mock) -> None:
    embedder = SentenceTransformerEmbedder(model=fake_model)

    vector = embedder.embed_query("Qual é o prazo de entrega?")

    assert vector == [1.0, 0.0]
    fake_model.encode.assert_called_once_with(
        ["query: Qual é o prazo de entrega?"],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def test_embed_documents_returns_empty_list_without_encoding(
    fake_model: Mock,
) -> None:
    embedder = SentenceTransformerEmbedder(model=fake_model)

    assert embedder.embed_documents([]) == []
    fake_model.encode.assert_not_called()


def test_embed_query_rejects_blank_text(fake_model: Mock) -> None:
    embedder = SentenceTransformerEmbedder(model=fake_model)

    with pytest.raises(ValueError, match="consulta não pode ser vazia"):
        embedder.embed_query("   ")


def test_embed_documents_rejects_blank_text(fake_model: Mock) -> None:
    embedder = SentenceTransformerEmbedder(model=fake_model)

    with pytest.raises(ValueError, match="textos dos documentos não podem ser vazios"):
        embedder.embed_documents(["edital", "  "])


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerEmbedder(batch_size=0)


def test_batch_size_rejects_boolean() -> None:
    with pytest.raises(TypeError, match="batch_size"):
        SentenceTransformerEmbedder(batch_size=True)
