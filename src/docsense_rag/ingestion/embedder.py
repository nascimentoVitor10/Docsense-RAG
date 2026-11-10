from typing import Protocol


class Embedder(Protocol):
    """Contrato para geração de embeddings de documentos e consultas."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings para vários trechos de documentos."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Gera um embedding para uma consulta de busca."""
        ...
