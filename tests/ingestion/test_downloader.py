from pathlib import Path

import httpx
import pytest

from docsense_rag.ingestion.downloader import (
    DownloadError,
    download_document,
)
from docsense_rag.ingestion.manifest import DocumentManifestEntry

PDF_CONTENT = b"%PDF-1.7\nconteudo de teste"


def make_entry() -> DocumentManifestEntry:
    return DocumentManifestEntry(
        document_id="edital-001",
        title="Edital de exemplo para testes",
        source_page_url="https://www.gov.br/",
        download_url="https://www.gov.br/edital.pdf",
        document_type="edital",
        organization="Órgão de exemplo",
    )


def make_client(
    handler,
) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    )


def test_downloads_valid_pdf(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF_CONTENT)

    with make_client(handler) as client:
        result = download_document(make_entry(), tmp_path, client)

    assert result.path == tmp_path / "edital-001.pdf"
    assert result.path.read_bytes() == PDF_CONTENT
    assert result.size_bytes == len(PDF_CONTENT)
    assert len(result.sha256) == 64


def test_rejects_non_pdf_response(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"<html>pagina de erro</html>",
        )

    with (
        make_client(handler) as client,
        pytest.raises(DownloadError, match="assinatura PDF"),
    ):
        download_document(make_entry(), tmp_path, client)

    assert not (tmp_path / "edital-001.pdf").exists()
    assert list(tmp_path.iterdir()) == []


def test_rejects_file_above_size_limit(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"%PDF-" + b"A" * 100,
        )

    with (
        make_client(handler) as client,
        pytest.raises(DownloadError, match="limite de tamanho"),
    ):
        download_document(
            make_entry(),
            tmp_path,
            client,
            max_bytes=50,
        )

    assert not (tmp_path / "edital-001.pdf").exists()
    assert list(tmp_path.iterdir()) == []


def test_rejects_missing_download_url(tmp_path: Path) -> None:
    entry = make_entry().model_copy(update={"download_url": None})

    with (
        make_client(lambda request: httpx.Response(200, content=PDF_CONTENT)) as client,
        pytest.raises(DownloadError, match="não possui download_url"),
    ):
        download_document(entry, tmp_path, client)

    assert list(tmp_path.iterdir()) == []


def test_rejects_http_url(tmp_path: Path) -> None:
    entry = make_entry().model_copy(
        update={"download_url": "http://www.gov.br/edital.pdf"}
    )

    with (
        make_client(lambda request: httpx.Response(200, content=PDF_CONTENT)) as client,
        pytest.raises(DownloadError, match="HTTPS"),
    ):
        download_document(entry, tmp_path, client)

    assert list(tmp_path.iterdir()) == []


def test_rejects_http_error_status(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with (
        make_client(handler) as client,
        pytest.raises(DownloadError, match="Falha ao baixar"),
    ):
        download_document(make_entry(), tmp_path, client)

    assert list(tmp_path.iterdir()) == []
