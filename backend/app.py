from fastapi import FastAPI
from backend.api.notebooks import router as notebooks_router

app = FastAPI(title="RAG NotebookLM Clone")

app.include_router(notebooks_router, prefix="/api/notebooks", tags=["notebooks"])


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
