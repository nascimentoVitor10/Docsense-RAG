from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class DocumentManifestEntry(BaseModel):
    """metadados de entrada de um documento de um corpus"""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9_-]{2,99}$",
    )
    title: str = Field(min_length=3, max_length=300)
    source_page_url: HttpUrl
    download_url: HttpUrl | None = None
    document_type: Literal["edital", "termo_referencia"]
    organization: str = Field(min_length=2, max_length=200)
