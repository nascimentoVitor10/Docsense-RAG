from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """Unidade textual rastreável derivada de um documento."""

    document_id: str
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    token_count: int

    def __post_init__(self) -> None:
        """Valida as invariantes do chunk."""

        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id não pode ser vazio.")

        if (
            not isinstance(self.chunk_index, int)
            or isinstance(self.chunk_index, bool)
            or self.chunk_index < 0
        ):
            raise ValueError("chunk_index deve ser um inteiro maior ou igual a zero.")

        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text não pode ser vazio.")

        if (
            not isinstance(self.page_start, int)
            or isinstance(self.page_start, bool)
            or self.page_start < 1
        ):
            raise ValueError("page_start deve ser um inteiro maior ou igual a 1.")

        if (
            not isinstance(self.page_end, int)
            or isinstance(self.page_end, bool)
            or self.page_end < self.page_start
        ):
            raise ValueError(
                "page_end deve ser um inteiro maior ou igual a page_start."
            )

        if (
            not isinstance(self.token_count, int)
            or isinstance(self.token_count, bool)
            or self.token_count <= 0
        ):
            raise ValueError("token_count deve ser um inteiro positivo.")
