import argparse
from pathlib import Path
from time import perf_counter

from qdrant_client import QdrantClient

from docsense_rag.ingestion.chunker import Chunker
from docsense_rag.ingestion.huggingface_tokenizer import HuggingFaceTokenizer
from docsense_rag.ingestion.parser import parse_pdf
from docsense_rag.ingestion.segmenter import segment_document
from docsense_rag.ingestion.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from docsense_rag.ingestion.service import IngestionService
from docsense_rag.storage.qdrant_store import QdrantVectorStore

MODEL_NAME = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384


class TimedEmbedder:
    """Adiciona métricas de tempo a um embedder existente."""

    def __init__(self, embedder: SentenceTransformerEmbedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        start = perf_counter()
        embeddings = self._embedder.embed_documents(texts)
        elapsed = perf_counter() - start

        print(
            f"Tempo de embeddings dos documentos: {elapsed:.2f}s ({len(texts)} textos)"
        )
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        start = perf_counter()
        embedding = self._embedder.embed_query(text)
        elapsed = perf_counter() - start

        print(f"Tempo de embedding da consulta: {elapsed:.2f}s")
        return embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingere um PDF no Qdrant e testa a busca semântica."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Caminho do PDF a ser ingerido.",
    )
    parser.add_argument(
        "--document-id",
        required=True,
        help="Identificador estável do documento.",
    )
    parser.add_argument(
        "--query",
        default="Qual é o objeto da contratação e quais são os prazos previstos?",
        help="Pergunta usada para testar a busca semântica.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Quantidade máxima de resultados da busca.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf.resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    if args.limit <= 0:
        raise ValueError("--limit deve ser maior que zero.")

    pipeline_start = perf_counter()

    # 1. Extração do PDF.
    document = parse_pdf(
        pdf_path=pdf_path,
        document_id=args.document_id,
    )

    print(f"\nDocumento: {document.document_id}")
    print(f"Arquivo: {document.source_path}")
    print(f"Páginas: {document.total_pages}")
    print(f"Páginas sem texto: {document.pages_without_text}")

    # 2. Segmentação com preservação das páginas de origem.
    segments = segment_document(document)

    print(f"Segmentos extraídos: {len(segments)}")

    if not segments:
        raise ValueError(
            "O PDF não produziu segmentos de texto. "
            "Verifique se é um PDF digital ou se precisa de OCR."
        )

    # 3. Tokenizador, chunker e modelo de embeddings.
    tokenizer = HuggingFaceTokenizer(MODEL_NAME)

    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=450,
        overlap_tokens=75,
    )

    embedder = TimedEmbedder(
        SentenceTransformerEmbedder(
            model_name=MODEL_NAME,
            device="cpu",
        )
    )

    # 4. Cliente Qdrant em memória.
    storage_path = Path("data/qdrant")
    storage_path.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(storage_path))

    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="docsense_chunks",
            vector_size=VECTOR_SIZE,
        )

        vector_store.ensure_collection()

        service = IngestionService(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )

        # 5. Ingestão: chunks -> embeddings -> persistência.
        chunks = service.ingest(
            document_id=document.document_id,
            segments=segments,
        )

        print(f"Chunks indexados: {len(chunks)}")
        print(
            "Tokens por chunk: "
            f"mínimo={min(chunk.token_count for chunk in chunks)}, "
            f"máximo={max(chunk.token_count for chunk in chunks)}"
        )

        # 6. Busca semântica com o mesmo modelo.
        query_vector = embedder.embed_query(args.query)
        results = vector_store.search(
            query_vector=query_vector,
            limit=args.limit,
        )

        print(f"\nConsulta: {args.query}")
        print(f"Resultados encontrados: {len(results)}")

        for rank, result in enumerate(results, start=1):
            payload = result.payload or {}

            print(f"\n--- Resultado {rank} ---")
            print(f"Similaridade: {result.score:.4f}")
            print(f"Documento: {payload.get('document_id')}")
            print(f"Páginas: {payload.get('page_start')}–{payload.get('page_end')}")
            print(f"Chunk: {payload.get('chunk_index')}")
            print(f"Texto: {payload.get('text')}")

        print(f"\nTempo total: {perf_counter() - pipeline_start:.2f}s")

    finally:
        client.close()


if __name__ == "__main__":
    main()
