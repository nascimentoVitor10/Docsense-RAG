
from __future__ import annotations

import csv
from pathlib import Path

import easyocr
import fitz


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = PROJECT_ROOT / "data" / "raw" / "edital-pe-58-2026.pdf"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ocr_validation"

# Páginas em numeração humana: 2, 7, 11 e 14.
PAGES_TO_VALIDATE = [2, 7, 11, 14]

OCR_LANGUAGES = ["pt", "en"]
RENDER_SCALE = 3


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)

    with fitz.open(PDF_PATH) as document:
        for page_number in PAGES_TO_VALIDATE:
            page = document[page_number - 1]

            page_dir = OUTPUT_DIR / f"page_{page_number:02d}"
            page_dir.mkdir(parents=True, exist_ok=True)

            # 1. Salva a imagem original renderizada do PDF.
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE),
                alpha=False,
            )
            image_path = page_dir / "original.png"
            pixmap.save(image_path)

            # 2. Extrai o texto nativo.
            native_text = page.get_text("text", sort=True)
            (page_dir / "native.txt").write_text(
                native_text,
                encoding="utf-8",
            )

            # 3. Executa o OCR.
            results = reader.readtext(
                str(image_path),
                detail=1,
                paragraph=False,
            )

            ocr_lines: list[str] = []

            with (page_dir / "ocr_details.csv").open(
                "w",
                encoding="utf-8-sig",
                newline="",
            ) as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["text", "confidence", "bounding_box"])

                for bounding_box, text, confidence in results:
                    ocr_lines.append(text)
                    writer.writerow(
                        [
                            text,
                            round(float(confidence), 4),
                            bounding_box,
                        ]
                    )

            (page_dir / "ocr.txt").write_text(
                "\n".join(ocr_lines),
                encoding="utf-8",
            )

            print(f"Página {page_number} validada.")
            print(f"Arquivos gerados em: {page_dir}")

    print("\nValidação concluída.")


if __name__ == "__main__":
    main()
