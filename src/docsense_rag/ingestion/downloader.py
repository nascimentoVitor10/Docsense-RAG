import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from docsense_rag.ingestion.manifest import DocumentManifestEntry


class DownloadError(ValueError):
    """Erro durante o download ou a validação de um documento."""


@dataclass(frozen=True)
class DownloadResult:
    """Metadados do documento baixado e validado."""

    document_id: str
    path: Path
    size_bytes: int
    sha256: str


def download_document(
    entry: DocumentManifestEntry,
    output_dir: Path,
    client: httpx.Client,
    max_bytes: int = 50_000_000,
) -> DownloadResult:
    """Baixa e valida um PDF antes de salvá-lo definitivamente."""

    if entry.download_url is None:
        raise DownloadError(f"Documento '{entry.document_id}' não possui download_url.")

    download_url = httpx.URL(str(entry.download_url))

    if download_url.scheme != "https":
        raise DownloadError("O download exige uma URL HTTPS.")

    if max_bytes <= 0:
        raise ValueError("max_bytes deve ser maior que zero.")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"{entry.document_id}.pdf"
    temp_path: Path | None = None

    try:
        digest = hashlib.sha256()
        header = bytearray()
        size_bytes = 0

        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{entry.document_id}.",
            suffix=".part",
            dir=output_dir,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

            with client.stream("GET", str(download_url)) as response:
                response.raise_for_status()

                content_length = response.headers.get("content-length")

                if content_length is not None:
                    try:
                        declared_size = int(content_length)
                    except ValueError:
                        declared_size = None

                    if declared_size is not None and declared_size > max_bytes:
                        raise DownloadError(
                            "Documento excede o limite de tamanho permitido."
                        )

                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    if not chunk:
                        continue

                    size_bytes += len(chunk)

                    if size_bytes > max_bytes:
                        raise DownloadError(
                            "Documento excede o limite de tamanho permitido."
                        )

                    if len(header) < 5:
                        header.extend(chunk[: 5 - len(header)])

                    digest.update(chunk)
                    temp_file.write(chunk)

        # Neste ponto, o arquivo temporário já está fechado.
        if bytes(header) != b"%PDF-":
            raise DownloadError(
                f"Documento '{entry.document_id}' não possui assinatura PDF válida."
            )

        temp_path.replace(target_path)

        return DownloadResult(
            document_id=entry.document_id,
            path=target_path,
            size_bytes=size_bytes,
            sha256=digest.hexdigest(),
        )

    except DownloadError:
        raise
    except (httpx.HTTPError, OSError) as exc:
        raise DownloadError(
            f"Falha ao baixar ou armazenar '{entry.document_id}'."
        ) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
