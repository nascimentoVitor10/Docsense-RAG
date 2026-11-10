from pathlib import Path

import pymupdf

pdf_path = Path("data/raw/edital-pe-58-2026.pdf")
ocr_path = Path("data/raw/diagnostic-page-2-ocr.txt")

expected_terms = [
    "software",
    "Justiça",
    "participar",
    "estiverem",
    "atividades",
    "exatidão",
    "praticados",
]

with pymupdf.open(pdf_path) as document:
    native_text = document[1].get_text("text")

ocr_text = ocr_path.read_text(encoding="utf-8")

for term in expected_terms:
    native_found = term.casefold() in native_text.casefold()
    ocr_found = term.casefold() in ocr_text.casefold()

    print(
        f"{term:15} "
        f"PyMuPDF={'OK' if native_found else 'ERRO':5} "
        f"OCR={'OK' if ocr_found else 'ERRO':5}"
    )
