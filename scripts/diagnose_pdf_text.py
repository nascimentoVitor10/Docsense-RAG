from pathlib import Path

import pymupdf

pdf_path = Path("data/raw/edital-pe-58-2026.pdf")
terms = ("soJware", "JusLça", "parLcipar")

with pymupdf.open(pdf_path) as pdf:
    for page_number in (2, 3):
        page = pdf[page_number - 1]
        data = page.get_text("rawdict")

        for block in data["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    chars = span.get("chars", [])
                    text = "".join(char["c"] for char in chars)

                    if any(term in text for term in terms):
                        print(f"\nPágina: {page_number}")
                        print(f"Fonte: {span.get('font', 'desconhecida')}")
                        print(f"Texto: {text!r}")
                        print("Caracteres:")

                        for char in chars:
                            print(f"  caractere={char['c']!r}, origem={char['origin']}")
