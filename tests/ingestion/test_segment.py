from dataclasses import FrozenInstanceError

import pytest

from docsense_rag.ingestion.segment import TextSegment


def test_accepts_valid_segment() -> None:
    segment = TextSegment(
        text="A contratada deverá fornecer suporte técnico.",
        page_number=3,
    )

    assert segment.text == "A contratada deverá fornecer suporte técnico."
    assert segment.page_number == 3


def test_preserves_original_text() -> None:
    text = "  Texto com espaços nas extremidades.  "

    segment = TextSegment(text=text, page_number=1)

    assert segment.text == text


@pytest.mark.parametrize("text", ["", " ", "\n\t", "   \n  "])
def test_rejects_empty_or_whitespace_text(text: str) -> None:
    with pytest.raises(ValueError, match="text não pode ser vazio"):
        TextSegment(text=text, page_number=1)


@pytest.mark.parametrize(
    "page_number",
    [0, -1, True, False, 1.5, "1", None],
)
def test_rejects_invalid_page_number(page_number: object) -> None:
    with pytest.raises(ValueError, match="page_number"):
        TextSegment(text="Texto válido.", page_number=page_number)  # type: ignore[arg-type]


def test_segment_is_immutable() -> None:
    segment = TextSegment(text="Texto válido.", page_number=1)

    with pytest.raises(FrozenInstanceError):
        segment.page_number = 2  # type: ignore[misc]
