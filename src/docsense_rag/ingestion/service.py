from docsense_rag.ingestion.chunk import Chunk
from docsense_rag.ingestion.chunker import Chunker
from docsense_rag.ingestion.embedder import Embedder
from docsense_rag.ingestion.segment import TextSegment
from docsense_rag.storage.qdrant_store import QdrantVectorStore


class IngestionService:
    """Coordena a geração e a persistência de chunks de um documento."""

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(
        self,
        document_id: str,
        segments: list[TextSegment],
    ) -> list[Chunk]:
        """Gera chunks e embeddings e substitui a versão indexada."""

        chunks = self.chunker.chunk(document_id, segments)

        if not chunks:
            raise ValueError("O documento não gerou chunks; a indexação foi cancelada.")

        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])

        if len(embeddings) != len(chunks):
            raise ValueError(
                "O embedder retornou uma quantidade incorreta de embeddings."
            )

        self.vector_store.replace_document(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
        )

        return chunks
