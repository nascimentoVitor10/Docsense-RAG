from fastapi import APIRouter

from docsense_rag.core.config import settings

router = APIRouter()


@router.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Informa se a aplicação está respondendo."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
    }
