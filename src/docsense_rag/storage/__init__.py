from math import isfinite
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from docsense_rag.ingestion.chunk import Chunk


class QdrantVectorStore:
    """Armazena e recupera chunks por similaridade vetorial."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "docsense_chunks",
        vector_size: int = 384,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("collection_name não pode ser vazio.")

        if (
            not isinstance(vector_size, int)
            or isinstance(vector_size, bool)
            or vector_size <= 0
        ):
            raise ValueError("vector_size deve ser um inteiro positivo.")

        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        """Cria a coleção se ela ainda não existir."""

        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Persiste chunks e seus vetores correspondentes."""

        if len(chunks) != len(embeddings):
            raise ValueError("A quantidade de chunks deve ser igual à de embeddings.")

        if not chunks:
            return

        self._validate_embeddings(embeddings)
        self.ensure_collection()

        points = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            point_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"docsense-rag:{chunk.document_id}:{chunk.chunk_index}",
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "token_count": chunk.token_count,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[models.ScoredPoint]:
        """Busca os chunks mais semelhantes à consulta."""

        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit deve ser um inteiro positivo.")

        self._validate_embeddings([query_vector])
        self.ensure_collection()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )

        return response.points

    def _validate_embeddings(
        self,
        embeddings: list[list[float]],
    ) -> None:
        """Valida dimensão e valores dos vetores."""

        for embedding in embeddings:
            if len(embedding) != self.vector_size:
                raise ValueError(f"Embedding deve ter {self.vector_size} dimensões.")

            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isfinite(value)
                for value in embedding
            ):
                raise ValueError("Embeddings devem conter apenas números finitos.")
