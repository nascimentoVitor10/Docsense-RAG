import re

import pytest

from docsense_rag.ingestion.chunker import Chunker
from docsense_rag.ingestion.segment import TextSegment


class FakeTokenizer:
    """Tokenizador determinístico para testes, sem dependências externas."""

    _TOKEN_PATTERN = re.compile(r"\S+|\s+", re.UNICODE)

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []

        for token in self._TOKEN_PATTERN.findall(text):
            if token.isspace():
                continue

            if token not in self._token_to_id:
                token_id = len(self._token_to_id)
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token

            token_ids.append(self._token_to_id[token])

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        try:
            return " ".join(self._id_to_token[token_id] for token_id in token_ids)
        except KeyError as exc:
            raise ValueError(f"ID de token desconhecido: {exc.args[0]}") from exc


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


def test_creates_chunker_with_valid_configuration(
    tokenizer: FakeTokenizer,
) -> None:
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=500,
        overlap_tokens=75,
    )

    assert chunker.tokenizer is tokenizer
    assert chunker.max_tokens == 500
    assert chunker.overlap_tokens == 75


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_rejects_non_positive_max_tokens(
    tokenizer: FakeTokenizer,
    max_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        Chunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=0,
        )


@pytest.mark.parametrize("overlap_tokens", [-1])
def test_rejects_negative_overlap(
    tokenizer: FakeTokenizer,
    overlap_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        Chunker(
            tokenizer=tokenizer,
            max_tokens=10,
            overlap_tokens=overlap_tokens,
        )


def test_rejects_overlap_equal_to_max_tokens(
    tokenizer: FakeTokenizer,
) -> None:
    with pytest.raises(ValueError):
        Chunker(
            tokenizer=tokenizer,
            max_tokens=10,
            overlap_tokens=10,
        )


@pytest.mark.parametrize("max_tokens", [True, False, 1.5, "500", None])
def test_rejects_invalid_max_tokens_type(
    tokenizer: FakeTokenizer,
    max_tokens: object,
) -> None:
    with pytest.raises(TypeError):
        Chunker(
            tokenizer=tokenizer,
            max_tokens=max_tokens,  # type: ignore[arg-type]
            overlap_tokens=0,
        )


@pytest.mark.parametrize("overlap_tokens", [True, False, 1.5, "75", None])
def test_rejects_invalid_overlap_tokens_type(
    tokenizer: FakeTokenizer,
    overlap_tokens: object,
) -> None:
    with pytest.raises(TypeError):
        Chunker(
            tokenizer=tokenizer,
            max_tokens=10,
            overlap_tokens=overlap_tokens,  # type: ignore[arg-type]
        )


def test_creates_chunk_from_single_segment(tokenizer: FakeTokenizer) -> None:
    segment = TextSegment(
        text="Objeto da contratação.",
        page_number=3,
    )
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=10,
        overlap_tokens=2,
    )

    chunks = chunker.chunk("edital-001", [segment])

    assert len(chunks) == 1
    assert chunks[0].document_id == "edital-001"
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "Objeto da contratação."
    assert chunks[0].page_start == 3
    assert chunks[0].page_end == 3
    assert chunks[0].token_count == len(tokenizer.encode(chunks[0].text))


def test_returns_empty_list_when_no_segments(
    tokenizer: FakeTokenizer,
) -> None:
    chunker = Chunker(tokenizer=tokenizer)

    assert chunker.chunk("edital-001", []) == []


def test_fake_tokenizer_round_trip(tokenizer: FakeTokenizer) -> None:
    text = "O órgão contratante, porém, exige 3 documentos."

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_fake_tokenizer_preserves_repeated_tokens(
    tokenizer: FakeTokenizer,
) -> None:
    text = "edital edital termo"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_fake_tokenizer_handles_empty_text(tokenizer: FakeTokenizer) -> None:
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_splits_long_segment_with_overlap(tokenizer: FakeTokenizer) -> None:
    text = "A B C D E F G H I J K L"
    segment = TextSegment(text=text, page_number=4)
    chunker = Chunker(tokenizer=tokenizer, max_tokens=5, overlap_tokens=2)

    chunks = chunker.chunk("edital-001", [segment])

    assert len(chunks) == 4
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.token_count for chunk in chunks] == [5, 5, 5, 3]
    assert all(chunk.page_start == 4 for chunk in chunks)
    assert all(chunk.page_end == 4 for chunk in chunks)
    assert all(chunk.document_id == "edital-001" for chunk in chunks)

    token_ids = [tokenizer.encode(chunk.text) for chunk in chunks]
    assert token_ids[0][-2:] == token_ids[1][:2]
    assert token_ids[1][-2:] == token_ids[2][:2]


def test_does_not_split_segment_at_exact_limit(
    tokenizer: FakeTokenizer,
) -> None:
    segment = TextSegment(text="A B C D E", page_number=2)
    chunker = Chunker(tokenizer=tokenizer, max_tokens=5, overlap_tokens=2)

    chunks = chunker.chunk("edital-001", [segment])

    assert len(chunks) == 1
    assert chunks[0].token_count == 5


def test_splits_segment_one_token_over_limit(
    tokenizer: FakeTokenizer,
) -> None:
    segment = TextSegment(text="A B C D E F", page_number=2)
    chunker = Chunker(tokenizer=tokenizer, max_tokens=5, overlap_tokens=2)

    chunks = chunker.chunk("edital-001", [segment])

    assert len(chunks) == 2
    assert all(chunk.token_count <= 5 for chunk in chunks)


def test_splits_long_segment_without_overlap(
    tokenizer: FakeTokenizer,
) -> None:
    segment = TextSegment(
        text="A B C D E F G",
        page_number=2,
    )
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=3,
        overlap_tokens=0,
    )

    chunks = chunker.chunk("edital-001", [segment])

    assert len(chunks) == 3
    assert [chunk.token_count for chunk in chunks] == [3, 3, 1]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]

    encoded_chunks = [tokenizer.encode(chunk.text) for chunk in chunks]

    # Sem overlap, o fim de um chunk não deve se repetir no seguinte.
    assert set(encoded_chunks[0]).isdisjoint(set(encoded_chunks[1]))
    assert set(encoded_chunks[1]).isdisjoint(set(encoded_chunks[2]))


def test_preserves_token_sequence_when_splitting(
    tokenizer: FakeTokenizer,
) -> None:
    original_text = "A B C D E F G H"
    segment = TextSegment(text=original_text, page_number=5)
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=4,
        overlap_tokens=1,
    )

    chunks = chunker.chunk("edital-001", [segment])

    original_tokens = tokenizer.encode(original_text)
    reconstructed_tokens: list[int] = []

    for index, chunk in enumerate(chunks):
        chunk_tokens = tokenizer.encode(chunk.text)

        if index > 0:
            # Remove o token repetido pelo overlap.
            chunk_tokens = chunk_tokens[1:]

        reconstructed_tokens.extend(chunk_tokens)

    assert reconstructed_tokens == original_tokens
    assert all(chunk.token_count <= 4 for chunk in chunks)


def test_preserves_indices_and_pages_across_segments(
    tokenizer: FakeTokenizer,
) -> None:
    segments = [
        TextSegment(text="Primeiro parágrafo.", page_number=1),
        TextSegment(text="Segundo parágrafo.", page_number=2),
        TextSegment(text="Terceiro parágrafo.", page_number=3),
    ]
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=3,
        overlap_tokens=0,
    )

    chunks = chunker.chunk("edital-001", segments)

    assert len(chunks) == 3
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert [chunk.page_start for chunk in chunks] == [1, 2, 3]
    assert [chunk.page_end for chunk in chunks] == [1, 2, 3]


def test_groups_segments_when_they_fit_within_limit(
    tokenizer: FakeTokenizer,
) -> None:
    segments = [
        TextSegment(text="Primeiro parágrafo.", page_number=1),
        TextSegment(text="Segundo parágrafo.", page_number=1),
        TextSegment(text="Terceiro parágrafo.", page_number=2),
    ]
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=30,
        overlap_tokens=0,
    )

    chunks = chunker.chunk("edital-001", segments)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert chunks[0].text == (
        "Primeiro parágrafo.\nSegundo parágrafo.\nTerceiro parágrafo."
    )
    assert chunks[0].token_count == len(tokenizer.encode(chunks[0].text))


def test_starts_new_chunk_when_next_segment_exceeds_limit(
    tokenizer: FakeTokenizer,
) -> None:
    segments = [
        TextSegment(text="A B C", page_number=1),
        TextSegment(text="D E F", page_number=2),
    ]
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=5,
        overlap_tokens=0,
    )

    chunks = chunker.chunk("edital-001", segments)

    assert len(chunks) == 2
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.page_start for chunk in chunks] == [1, 2]
    assert [chunk.page_end for chunk in chunks] == [1, 2]
    assert all(chunk.token_count <= 5 for chunk in chunks)


def test_preserves_page_range_when_grouping_segments(
    tokenizer: FakeTokenizer,
) -> None:
    segments = [
        TextSegment(text="Objeto da contratação.", page_number=4),
        TextSegment(text="Requisitos técnicos.", page_number=5),
    ]
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=30,
        overlap_tokens=0,
    )

    chunks = chunker.chunk("edital-001", segments)

    assert len(chunks) == 1
    assert chunks[0].page_start == 4
    assert chunks[0].page_end == 5
