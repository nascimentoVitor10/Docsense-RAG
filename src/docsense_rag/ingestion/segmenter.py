import re

from docsense_rag.ingestion.parser import ParsedDocument
from docsense_rag.ingestion.segment import TextSegment


def segment_document(document: ParsedDocument) -> list[TextSegment]:
    """Divide o texto de um documento em segmentos, preservando a página."""
    segments: list[TextSegment] = []

    for page in document.pages:
        page_text = page.text.strip()

        if not page_text:
            continue

        paragraphs = re.split(r"\n[ \t]*\n+", page_text)

        for paragraph in paragraphs:
            normalized_text = re.sub(r"\s+", " ", paragraph).strip()

            if not normalized_text:
                continue

            segments.append(
                TextSegment(
                    text=normalized_text,
                    page_number=page.page_number,
                )
            )

    return segments
