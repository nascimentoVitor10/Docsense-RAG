
from pathlib import Path

import easyocr
import pymupdf

from docsense_rag.ingestion.text_quality import assess_text_quality


PDF_PATH = Path("data/raw/edital-pe-58-2026.pdf")
OUTPUT_DIR = Path("data/processed/ocr_diagnostics")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reader = easyocr.Reader(["pt", "en"], gpu=False)

    with pymupdf.open(PDF_PATH) as document:
        combined_text: list[str] = []
        pages_processed = 0

        for page in document:
            native_text = page.get_text("text").strip()
            report = assess_text_quality(native_text)

            # OCR somente nas páginas sinalizadas.
            if not report.needs_ocr:
                selected_text = native_text
                extraction_method = "PyMuPDF"
            else:
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(3, 3),
                    alpha=False,
                )

                image_path = OUTPUT_DIR / f"page_{page.number + 1:02}.png"
                pixmap.save(image_path)

                ocr_results = reader.readtext(
                    str(image_path),
                    detail=0,
                )

                selected_text = "\n".join(ocr_results)
                extraction_method = "EasyOCR"
                pages_processed += 1

                ocr_path = (
                    OUTPUT_DIR / f"page_{page.number + 1:02}.txt"
                )
                ocr_path.write_text(
                    selected_text,
                    encoding="utf-8",
                )

            combined_text.append(
                f"\n{'=' * 30}\n"
                f"PÁGINA {page.number + 1}\n"
                f"MÉTODO: {extraction_method}\n"
                f"{'=' * 30}\n"
                f"{selected_text}"
            )

            print(
                f"Página {page.number + 1:02}: "
                f"{extraction_method}"
            )

        output_path = OUTPUT_DIR / "document.txt"
        output_path.write_text(
            "\n".join(combined_text),
            encoding="utf-8",
        )

    print("\n=== Resumo ===")
    print(f"Páginas processadas com OCR: {pages_processed}")
    print(f"Texto consolidado: {output_path}")


if __name__ == "__main__":
    main()
