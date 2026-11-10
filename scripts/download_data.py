import argparse
import logging
from pathlib import Path

import httpx

from docsense_rag.ingestion.downloader import (
    DownloadError,
    download_document,
)
from docsense_rag.ingestion.reader import ManifestError, load_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa e valida os documentos PDF do manifesto."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifest.jsonl",
        help="Caminho do manifesto JSONL.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Diretório de destino dos PDFs.",
    )

    parser.add_argument(
        "--max-bytes",
        type=int,
        default=50_000_000,
        help="Tamanho máximo permitido por documento, em bytes.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        entries = load_manifest(args.manifest)
    except ManifestError as exc:
        logger.error("Não foi possível carregar o manifesto: %s", exc)
        return 1

    if not entries:
        logger.error("O manifesto não contém documentos para processar.")
        return 1

    logger.info("Documentos encontrados: %d", len(entries))

    successful = 0
    failed = 0

    timeout = httpx.Timeout(30.0, connect=10.0)

    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "DocSenseRAG/0.1"},
    ) as client:
        for entry in entries:
            logger.info(
                "Processando documento: %s",
                entry.document_id,
            )

            try:
                result = download_document(
                    entry=entry,
                    output_dir=args.output,
                    client=client,
                    max_bytes=args.max_bytes,
                )
            except DownloadError as exc:
                failed += 1
                logger.error(
                    "Falha em %s: %s",
                    entry.document_id,
                    exc,
                )
                continue

            successful += 1

            logger.info(
                "Download concluído: %s | %d bytes | SHA-256: %s",
                result.path,
                result.size_bytes,
                result.sha256,
            )

    logger.info(
        "Resumo: %d sucesso(s), %d falha(s).",
        successful,
        failed,
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
