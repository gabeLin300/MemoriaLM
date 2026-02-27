from fastapi import FastAPI

from backend.api.artifacts import router as artifacts_router
from backend.api.chat import router as chat_router
from backend.api.notebooks import router as notebooks_router
from backend.api.sources import router as sources_router

app = FastAPI(title="RAG NotebookLM Clone")

app.include_router(notebooks_router, prefix="/api/notebooks", tags=["notebooks"])
app.include_router(sources_router, prefix="/api/notebooks", tags=["sources"])
app.include_router(chat_router, prefix="/api/notebooks", tags=["chat"])
app.include_router(artifacts_router, prefix="/api/notebooks", tags=["artifacts"])


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
