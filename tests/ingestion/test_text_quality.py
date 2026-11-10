import pytest

from docsense_rag.ingestion.text_quality import assess_text_quality


def test_detects_corrupted_words() -> None:
    text = (
        "Contratação de soJware para o Ministério da JusLça. "
        "Poderão parLcipar os interessados."
    )

    report = assess_text_quality(text)

    assert report.needs_ocr is True
    assert report.total_words > 0
    assert "soJware" in report.suspicious_words
    assert "JusLça" in report.suspicious_words
    assert "parLcipar" in report.suspicious_words


def test_accepts_normal_portuguese_text() -> None:
    text = (
        "O objeto é a contratação de serviços de software "
        "para o Ministério da Justiça e Segurança Pública."
    )

    report = assess_text_quality(text)

    assert report.needs_ocr is False
    assert report.suspicious_words == ()
    assert report.suspicious_ratio == 0.0


def test_ignores_acronyms_written_in_uppercase() -> None:
    report = assess_text_quality("O edital utiliza os sistemas MJSP, SICAF e UASG.")

    assert report.suspicious_words == ()
    assert report.needs_ocr is False


def test_empty_text_does_not_require_ocr() -> None:
    report = assess_text_quality("   \n\t")

    assert report.total_words == 0
    assert report.suspicious_words == ()
    assert report.suspicious_ratio == 0.0
    assert report.needs_ocr is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_suspicious_words": 0}, "min_suspicious_words"),
        ({"min_suspicious_words": True}, "min_suspicious_words"),
        ({"min_suspicious_ratio": -0.1}, "min_suspicious_ratio"),
        ({"min_suspicious_ratio": 1.1}, "min_suspicious_ratio"),
        ({"min_suspicious_ratio": True}, "min_suspicious_ratio"),
    ],
)
def test_rejects_invalid_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        assess_text_quality("Texto de exemplo.", **kwargs)
