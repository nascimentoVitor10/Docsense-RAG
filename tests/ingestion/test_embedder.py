from docsense_rag.ingestion.embedder import Embedder


class FakeEmbedder:
    """Implementação determinística para testes."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def test_fake_embedder_satisfies_contract() -> None:
    embedder: Embedder = FakeEmbedder()

    vector = embedder.embed_query("edital de contratação")

    assert vector == [float(len("edital de contratação")), 1.0]


def test_fake_embedder_processes_multiple_documents() -> None:
    embedder: Embedder = FakeEmbedder()

    vectors = embedder.embed_documents(["edital", "contrato"])

    assert vectors == [[6.0, 1.0], [8.0, 1.0]]


def test_embed_documents_returns_empty_list_for_empty_input() -> None:
    embedder: Embedder = FakeEmbedder()

    assert embedder.embed_documents([]) == []
