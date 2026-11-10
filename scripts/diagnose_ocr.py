from pathlib import Path

import easyocr
import pymupdf

pdf_path = Path("data/raw/edital-pe-58-2026.pdf")
image_path = Path("data/raw/diagnostic-page-2.png")

# Renderiza a página 2 em alta resolução.
with pymupdf.open(pdf_path) as document:
    page = document[1]
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(3, 3),
        alpha=False,
    )
    pixmap.save(image_path)

# Executa OCR em português e inglês, em CPU.
reader = easyocr.Reader(["pt", "en"], gpu=False)
results = reader.readtext(str(image_path), detail=0)

text = "\n".join(results)
print(text)

output_path = Path("data/raw/diagnostic-page-2-ocr.txt")
output_path.write_text(text, encoding="utf-8")
print(f"\nResultado salvo em: {output_path}")
