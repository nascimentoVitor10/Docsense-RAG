from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    """Trecho textual extraído de uma página de um documento"""

    text: str
    page_number: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text não pode ser vazio.")

        if (
            not isinstance(self.page_number, int)
            or isinstance(self.page_number, bool)
            or self.page_number < 1
        ):
            raise ValueError("page_number deve ser um inteiro maior ou igual a 1.")
