from docsense_rag.ingestion.chunk import Chunk
from docsense_rag.ingestion.segment import TextSegment
from docsense_rag.ingestion.tokenizer import Tokenizer


class Chunker:
    def __init__(
        self,
        tokenizer: Tokenizer,
        max_tokens: int = 500,
        overlap_tokens: int = 75,
    ) -> None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise TypeError("O valor de max_tokens é de um tipo inválido.")

        if not isinstance(overlap_tokens, int) or isinstance(overlap_tokens, bool):
            raise TypeError("O valor de overlap_tokens é de um tipo inválido.")

        if max_tokens <= 0:
            raise ValueError("max_tokens deve ser maior do que zero.")

        if overlap_tokens < 0:
            raise ValueError("overlap_tokens deve ser um inteiro positivo.")

        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens deve ser menor do que max_tokens.")

        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(
        self,
        document_id: str,
        segments: list[TextSegment],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        pending_segments: list[TextSegment] = []

        def flush_pending() -> None:
            if not pending_segments:
                return

            text = "\n".join(segment.text for segment in pending_segments)
            token_count = len(self.tokenizer.encode(text))

            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_index=len(chunks),
                    text=text,
                    page_start=pending_segments[0].page_number,
                    page_end=pending_segments[-1].page_number,
                    token_count=token_count,
                )
            )
            pending_segments.clear()

        def split_long_segment(segment: TextSegment) -> None:
            token_ids = self.tokenizer.encode(segment.text)

            if not token_ids:
                raise ValueError(
                    "O tokenizador não retornou tokens para um segmento não vazio."
                )

            step = self.max_tokens - self.overlap_tokens

            for start in range(0, len(token_ids), step):
                window = token_ids[start : start + self.max_tokens]
                text = self.tokenizer.decode(window)

                # Conta os tokens do texto que será efetivamente armazenado.
                token_count = len(self.tokenizer.encode(text))

                if token_count > self.max_tokens:
                    raise ValueError(
                        "O texto decodificado excedeu max_tokens após a "
                        "retokenização. Revise o comportamento do tokenizador."
                    )

                if token_count == 0:
                    raise ValueError("O texto decodificado não produziu tokens.")

                chunks.append(
                    Chunk(
                        document_id=document_id,
                        chunk_index=len(chunks),
                        text=text,
                        page_start=segment.page_number,
                        page_end=segment.page_number,
                        token_count=token_count,
                    )
                )

                if start + self.max_tokens >= len(token_ids):
                    break

        for segment in segments:
            segment_token_count = len(self.tokenizer.encode(segment.text))

            if segment_token_count == 0:
                raise ValueError(
                    "O tokenizador não retornou tokens para um segmento não vazio."
                )

            if segment_token_count > self.max_tokens:
                flush_pending()
                split_long_segment(segment)
                continue

            candidate_segments = [*pending_segments, segment]
            candidate_text = "\n".join(item.text for item in candidate_segments)
            candidate_token_count = len(self.tokenizer.encode(candidate_text))

            if candidate_token_count <= self.max_tokens:
                pending_segments.append(segment)
            else:
                flush_pending()
                pending_segments.append(segment)

        flush_pending()

        return chunks
