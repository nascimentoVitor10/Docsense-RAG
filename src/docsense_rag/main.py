from fastapi import FastAPI

from docsense_rag.api.routes import router
from docsense_rag.core.config import settings

app = FastAPI(title=settings.app_name, version="0.1.0")

app.include_router(router)
