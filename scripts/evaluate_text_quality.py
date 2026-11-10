
from pathlib import Path

import pymupdf

from docsense_rag.ingestion.text_quality import assess_text_quality


PDF_PATH = Path("data/raw/edital-pe-58-2026.pdf")


def main() -> None:
    with pymupdf.open(PDF_PATH) as document:
        pages_flagged = 0

        for page in document:
            text = page.get_text("text").strip()
            report = assess_text_quality(text)

            if report.needs_ocr:
                pages_flagged += 1

            status = "OCR sugerido" if report.needs_ocr else "Sem alerta"

            print(
                f"Página {page.number + 1:02}: "
                f"{status} | "
                f"palavras={report.total_words} | "
                f"suspeitas={len(report.suspicious_words)} | "
                f"proporção={report.suspicious_ratio:.2%}"
            )

            if report.suspicious_words:
                print(
                    "  Exemplos: "
                    + ", ".join(report.suspicious_words[:10])
                )

        print("\n=== Resumo ===")
        print(f"Total de páginas: {len(document)}")
        print(f"Páginas sinalizadas: {pages_flagged}")


if __name__ == "__main__":
    main()
