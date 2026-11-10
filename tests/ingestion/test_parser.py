from pathlib import Path

import pymupdf
import pytest

from docsense_rag.ingestion.parser import (
    PDFExtractionError,
    parse_pdf,
)


def create_pdf(path: Path, texts: list[str]) -> Path:
    """Cria um PDF temporário com uma página por texto."""
    with pymupdf.open() as pdf:
        for text in texts:
            page = pdf.new_page()
            page.insert_text((72, 72), text)

        pdf.save(path)

    return path


def test_parse_pdf_preserves_pages_and_text(tmp_path: Path) -> None:
    pdf_path = create_pdf(
        tmp_path / "sample.pdf",
        ["Texto da primeira página", "Texto da segunda página"],
    )

    result = parse_pdf(pdf_path, "documento-teste")

    assert result.document_id == "documento-teste"
    assert result.total_pages == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "Texto da primeira página" in result.pages[0].text
    assert "Texto da segunda página" in result.pages[1].text
    assert result.pages_without_text == ()


def test_parse_pdf_rejects_missing_file(tmp_path: Path) -> None:
    pdf_path = tmp_path / "inexistente.pdf"

    with pytest.raises(PDFExtractionError, match="não encontrado"):
        parse_pdf(pdf_path, "documento-teste")


def test_parse_pdf_rejects_invalid_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "invalido.pdf"
    pdf_path.write_text("Este arquivo não é um PDF válido.", encoding="utf-8")

    with pytest.raises(PDFExtractionError):
        parse_pdf(pdf_path, "documento-teste")


def test_parse_pdf_rejects_empty_document_id(tmp_path: Path) -> None:
    pdf_path = create_pdf(
        tmp_path / "sample.pdf",
        ["Conteúdo de teste"],
    )

    with pytest.raises(ValueError, match="document_id"):
        parse_pdf(pdf_path, "   ")
