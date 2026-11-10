from dataclasses import dataclass
from pathlib import Path

import pymupdf


class PDFExtractionError(ValueError):
    """Erro ao abrir ou extrair texto de um PDF"""


@dataclass(frozen=True)
class ParsedPage:
    """Texto extraído de uma página, com sua referência de origem"""

    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    """Documento identificado e extraído"""

    document_id: str
    source_path: Path
    pages: tuple[ParsedPage, ...]

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def pages_without_text(self) -> tuple[int, ...]:
        return tuple(page.page_number for page in self.pages if not page.text.strip())


def parse_pdf(
    pdf_path: Path,
    document_id: str,
) -> ParsedDocument:
    """Extrai o texto de um PDF, preservando o número de páginas"""

    if not pdf_path.is_file():
        raise PDFExtractionError(f"Arquivo PDF não encontrado: {pdf_path}")

    if not document_id.strip():
        raise ValueError("document_id não pode ser vazio.")

    try:
        with pymupdf.open(pdf_path) as pdf:
            if pdf.needs_pass:
                raise PDFExtractionError(
                    f"O PDF {document_id} está protegido por senha. "
                )

            if pdf.page_count == 0:
                raise PDFExtractionError(f"O PDF {document_id} não possui páginas.")

            pages = tuple(
                ParsedPage(page_number=page_number, text=page.get_text("text").strip())
                for page_number, page in enumerate(pdf, start=1)
            )

    except PDFExtractionError:
        raise
    except (pymupdf.FileDataError, OSError, RuntimeError) as exc:
        raise PDFExtractionError(
            f"Não foi possível extrair o PDF {document_id}."
        ) from exc

    return ParsedDocument(document_id=document_id, source_path=pdf_path, pages=pages)
