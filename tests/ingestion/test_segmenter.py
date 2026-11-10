from pathlib import Path

import pytest

from docsense_rag.ingestion.parser import (
    ParsedDocument,
    ParsedPage,
)
from docsense_rag.ingestion.segmenter import segment_document


def make_document(*page_texts: str) -> ParsedDocument:
    """Cria um documento de teste com páginas numeradas sequencialmente."""
    pages = tuple(
        ParsedPage(page_number=index, text=text)
        for index, text in enumerate(page_texts, start=1)
    )

    return ParsedDocument(
        document_id="documento-teste",
        source_path=Path("documento-teste.pdf"),
        pages=pages,
    )


def test_splits_paragraphs() -> None:
    document = make_document("Objeto da contratação.\n\nRequisitos técnicos.")

    segments = segment_document(document)

    assert [segment.text for segment in segments] == [
        "Objeto da contratação.",
        "Requisitos técnicos.",
    ]


def test_normalizes_internal_whitespace() -> None:
    document = make_document(
        "A contratada deverá\n"
        "disponibilizar suporte técnico.\n\n"
        "O contrato terá vigência de 12 meses."
    )

    segments = segment_document(document)

    assert [segment.text for segment in segments] == [
        "A contratada deverá disponibilizar suporte técnico.",
        "O contrato terá vigência de 12 meses.",
    ]


def test_preserves_source_page_numbers() -> None:
    document = make_document(
        "Objeto da contratação.",
        "",
        "Requisitos técnicos.",
    )

    segments = segment_document(document)

    assert [segment.page_number for segment in segments] == [1, 3]


def test_preserves_segment_order_across_pages() -> None:
    document = make_document(
        "Primeiro parágrafo.\n\nSegundo parágrafo.",
        "Terceiro parágrafo.",
    )

    segments = segment_document(document)

    assert [segment.text for segment in segments] == [
        "Primeiro parágrafo.",
        "Segundo parágrafo.",
        "Terceiro parágrafo.",
    ]


@pytest.mark.parametrize("page_text", ["", " ", "\n\n", " \t\n  "])
def test_ignores_pages_without_useful_text(page_text: str) -> None:
    document = make_document(page_text)

    assert segment_document(document) == []


def test_ignores_empty_paragraphs() -> None:
    document = make_document("\n\nPrimeiro parágrafo.\n\n\nSegundo parágrafo.\n\n")

    segments = segment_document(document)

    assert [segment.text for segment in segments] == [
        "Primeiro parágrafo.",
        "Segundo parágrafo.",
    ]
