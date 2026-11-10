import re
from dataclasses import dataclass

_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TextQualityReport:
    """Relatório de indicadores de qualidade do texto extraído."""

    total_words: int
    suspicious_words: tuple[str, ...]
    suspicious_ratio: float
    needs_ocr: bool


def assess_text_quality(
    text: str,
    *,
    min_suspicious_words: int = 2,
    min_suspicious_ratio: float = 0.01,
) -> TextQualityReport:
    """Identifica palavras com maiúsculas inesperadas no interior."""

    if not isinstance(text, str):
        raise TypeError("O texto deve ser uma string.")

    if (
        isinstance(min_suspicious_words, bool)
        or not isinstance(min_suspicious_words, int)
        or min_suspicious_words < 1
    ):
        raise ValueError("min_suspicious_words deve ser um inteiro positivo.")

    if (
        isinstance(min_suspicious_ratio, bool)
        or not isinstance(min_suspicious_ratio, (int, float))
        or not 0 <= min_suspicious_ratio <= 1
    ):
        raise ValueError("min_suspicious_ratio deve estar entre 0 e 1.")

    words = _WORD_PATTERN.findall(text)

    suspicious_words = tuple(
        word
        for word in words
        if any(char.isupper() for char in word[1:]) and not word.isupper()
    )

    total_words = len(words)
    suspicious_count = len(suspicious_words)

    suspicious_ratio = suspicious_count / total_words if total_words else 0.0

    needs_ocr = suspicious_count >= min_suspicious_words or (
        suspicious_count > 0 and suspicious_ratio >= min_suspicious_ratio
    )

    return TextQualityReport(
        total_words=total_words,
        suspicious_words=suspicious_words,
        suspicious_ratio=suspicious_ratio,
        needs_ocr=needs_ocr,
    )
