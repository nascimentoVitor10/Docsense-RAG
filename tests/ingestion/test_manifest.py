import json
from pathlib import Path

import pytest

from docsense_rag.ingestion.reader import ManifestError, load_manifest


def write_manifest(path: Path, *records: str) -> None:
    """Cria um manifesto de teste com as linhas fornecidas."""
    path.write_text("\n".join(records), encoding="utf-8")


def valid_record(document_id: str = "edital-001") -> str:
    return json.dumps(
        {
            "document_id": document_id,
            "title": "Edital de exemplo para testes",
            "source_page_url": "https://www.gov.br/",
            "download_url": "https://www.gov.br/edital.pdf",
            "document_type": "edital",
            "organization": "Órgão de exemplo",
        }
    )


def test_loads_valid_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, valid_record(), valid_record("edital-002"))

    entries = load_manifest(path)

    assert len(entries) == 2
    assert entries[0].document_id == "edital-001"
    assert entries[1].document_id == "edital-002"


def test_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, valid_record(), '{"document_id":')

    with pytest.raises(ManifestError, match="Linha 2: JSON inválido"):
        load_manifest(path)


def test_rejects_invalid_record(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path,
        valid_record(),
        json.dumps(
            {
                "document_id": "edital-002",
                "title": "Edital de exemplo",
                "source_page_url": "https://www.gov.br/",
                "document_type": "contrato",
                "organization": "Órgão de exemplo",
            }
        ),
    )

    with pytest.raises(ManifestError, match="Linha 2: registro inválido"):
        load_manifest(path)


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    record = valid_record()
    write_manifest(path, record, record)

    with pytest.raises(ManifestError, match="document_id duplicado"):
        load_manifest(path)


def test_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, "", valid_record(), "   ")

    entries = load_manifest(path)

    assert len(entries) == 1
    assert entries[0].document_id == "edital-001"
