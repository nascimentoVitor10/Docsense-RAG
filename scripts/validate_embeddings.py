import math

from docsense_rag.ingestion.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)


def main() -> None:
    embedder = SentenceTransformerEmbedder(device="cpu")

    documents = [
        (
            "O prazo para entrega dos produtos é de 30 dias "
            "após a emissão da ordem de fornecimento."
        ),
        "A contratação tem como objetivo a prestação de serviços de limpeza predial.",
        (
            "O pagamento será efetuado em até 10 dias úteis "
            "após o recebimento da nota fiscal."
        ),
    ]

    query = "Qual é o prazo para entregar os produtos?"

    document_vectors = embedder.embed_documents(documents)
    query_vector = embedder.embed_query(query)

    if not document_vectors:
        raise RuntimeError("Nenhum embedding de documento foi gerado.")

    dimension = len(query_vector)

    if dimension != 384:
        raise RuntimeError(f"Dimensão inesperada: {dimension}; esperado: 384.")

    for index, vector in enumerate([query_vector, *document_vectors]):
        if len(vector) != dimension:
            raise RuntimeError(f"Dimensão inconsistente no vetor {index}.")

        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError(f"Valores inválidos no vetor {index}.")

        norm = math.sqrt(sum(value * value for value in vector))

        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise RuntimeError(f"Vetor {index} não está normalizado: {norm}.")

    # Como os vetores são normalizados, o produto escalar
    # equivale à similaridade por cosseno.
    results = [
        (
            sum(
                query_value * document_value
                for query_value, document_value in zip(
                    query_vector, document_vector, strict=True
                )
            ),
            document,
        )
        for document, document_vector in zip(documents, document_vectors, strict=True)
    ]

    results.sort(key=lambda item: item[0], reverse=True)

    print(f"Dimensão dos embeddings: {dimension}")
    print(f"Quantidade de documentos: {len(document_vectors)}")
    print("\nResultados para a consulta:")

    for score, document in results:
        print(f"\nSimilaridade: {score:.4f}")
        print(f"Texto: {document}")


if __name__ == "__main__":
    main()
