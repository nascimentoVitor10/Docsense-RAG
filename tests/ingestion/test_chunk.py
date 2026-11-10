from dataclasses import FrozenInstanceError

import pytest

from docsense_rag.ingestion.chunk import Chunk


def test_chunk_accepts_valid_data() -> None:
    chunk = Chunk(
        document_id="edital-001",
        chunk_index=0,
        text="A contratada deverá manter equipe técnica qualificada.",
        page_start=12,
        page_end=13,
        token_count=18,
    )

    assert chunk.document_id == "edital-001"
    assert chunk.chunk_index == 0
    assert chunk.page_start == 12
    assert chunk.page_end == 13
    assert chunk.token_count == 18


@pytest.mark.parametrize("document_id", ["", "   "])
def test_chunk_rejects_empty_document_id(document_id: str) -> None:
    with pytest.raises(ValueError, match="document_id"):
        Chunk(
            document_id=document_id,
            chunk_index=0,
            text="Texto válido",
            page_start=1,
            page_end=1,
            token_count=3,
        )


@pytest.mark.parametrize("chunk_index", [-1, True, 1.5])
def test_chunk_rejects_invalid_index(chunk_index: int) -> None:
    with pytest.raises(ValueError, match="chunk_index"):
        Chunk(
            document_id="edital-001",
            chunk_index=chunk_index,
            text="Texto válido",
            page_start=1,
            page_end=1,
            token_count=3,
        )


@pytest.mark.parametrize("text", ["", "   "])
def test_chunk_rejects_empty_text(text: str) -> None:
    with pytest.raises(ValueError, match="text"):
        Chunk(
            document_id="edital-001",
            chunk_index=0,
            text=text,
            page_start=1,
            page_end=1,
            token_count=3,
        )


@pytest.mark.parametrize(
    ("page_start", "page_end"),
    [
        (0, 1),
        (-1, 1),
        (2, 1),
        (True, 1),
        (1, False),
    ],
)
def test_chunk_rejects_invalid_page_range(
    page_start: int,
    page_end: int,
) -> None:
    with pytest.raises(ValueError):
        Chunk(
            document_id="edital-001",
            chunk_index=0,
            text="Texto válido",
            page_start=page_start,
            page_end=page_end,
            token_count=3,
        )


@pytest.mark.parametrize("token_count", [0, -1, True, 1.5])
def test_chunk_rejects_invalid_token_count(token_count: int) -> None:
    with pytest.raises(ValueError, match="token_count"):
        Chunk(
            document_id="edital-001",
            chunk_index=0,
            text="Texto válido",
            page_start=1,
            page_end=1,
            token_count=token_count,
        )


def test_chunk_is_immutable() -> None:
    chunk = Chunk(
        document_id="edital-001",
        chunk_index=0,
        text="Texto válido",
        page_start=1,
        page_end=1,
        token_count=3,
    )

    with pytest.raises(FrozenInstanceError):
        chunk.text = "Texto alterado"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("document_id", ""),
        ("document_id", "   "),
        ("text", ""),
        ("text", "   "),
    ],
)
def test_rejects_empty_string_fields(
    field: str,
    value: str,
) -> None:
    values = {
        "document_id": "edital-001",
        "chunk_index": 0,
        "text": "Conteúdo do edital",
        "page_start": 1,
        "page_end": 1,
        "token_count": 3,
    }
    values[field] = value

    with pytest.raises(ValueError):
        Chunk(**values)
