import json
from pathlib import Path

from pydantic import ValidationError

from docsense_rag.ingestion.manifest import DocumentManifestEntry


class ManifestError(ValueError):
    """Erro de leitura ou validação do manifesto"""


def load_manifest(path: Path) -> list[DocumentManifestEntry]:
    """Lê, valida e verifica IDs duplicados no manifesto jsonl"""

    entries: list[DocumentManifestEntry] = []
    seen_ids: set[str] = set()

    try:
        with path.open("r", encoding="utf-8") as manifest_file:
            for line_number, line in enumerate(manifest_file, start=1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ManifestError(f"Linha {line_number}: JSON inválido") from exc

                try:
                    entry = DocumentManifestEntry.model_validate(data)
                except ValidationError as exc:
                    raise ManifestError(
                        f"Linha {line_number}: registro inválido : {exc}"
                    ) from exc

                if entry.document_id in seen_ids:
                    raise ManifestError(
                        f"Linha {line_number}: document_id duplicado"
                        f"{entry.document_id}."
                    )

                seen_ids.add(entry.document_id)
                entries.append(entry)

    except ValueError as exc:
        raise ManifestError(
            f"Não foi possível ler o manifesto '{path}': {exc}"
        ) from exc

    return entries
