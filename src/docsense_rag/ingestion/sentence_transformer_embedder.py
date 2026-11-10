from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedder:
    """Gera embeddings usando um modelo Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str | None = None,
        batch_size: int = 32,
        model: SentenceTransformer | None = None,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size deve ser um inteiro.")

        if batch_size <= 0:
            raise ValueError("batch_size deve ser maior que zero.")

        self.batch_size = batch_size

        if model is not None:
            self._model = model
        elif device is None:
            self._model = SentenceTransformer(model_name)
        else:
            self._model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings para trechos de documentos."""

        if not texts:
            return []

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Os textos dos documentos não podem ser vazios.")

        prefixed_texts = [f"passage: {text}" for text in texts]
        return self._encode(prefixed_texts)

    def embed_query(self, text: str) -> list[float]:
        """Gera um embedding para uma consulta."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("A consulta não pode ser vazia.")

        return self._encode([f"query: {text}"])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Codifica textos e converte os vetores para listas Python."""

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return [[float(value) for value in vector] for vector in embeddings.tolist()]
