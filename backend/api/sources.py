from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from backend.models.schemas import SourceListOut, SourceOut, UrlIngestRequest
from backend.modules.ingestion import ingest_uploaded_bytes, ingest_url, list_ingested_sources
from backend.services.storage import NotebookStore

router = APIRouter()
store = NotebookStore(base_dir="data")


@router.get("/{notebook_id}/sources", response_model=SourceListOut)
def list_sources(notebook_id: str, user_id: str = Query(...)) -> SourceListOut:
    try:
        items = list_ingested_sources(store, user_id=user_id, notebook_id=notebook_id)
        return SourceListOut(sources=items)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{notebook_id}/sources/url", response_model=SourceOut)
def ingest_source_url(notebook_id: str, payload: UrlIngestRequest) -> SourceOut:
    try:
        return ingest_url(
            store,
            user_id=payload.user_id,
            notebook_id=notebook_id,
            url=str(payload.url),
            source_name=payload.source_name,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"URL ingestion failed: {exc}") from exc


@router.post("/{notebook_id}/sources/upload", response_model=SourceOut)
async def upload_source_file(
    notebook_id: str,
    user_id: str = Form(...),
    file: UploadFile = File(...),
) -> SourceOut:
    try:
        content = await file.read()
        return ingest_uploaded_bytes(
            store,
            user_id=user_id,
            notebook_id=notebook_id,
            filename=file.filename or "upload.txt",
            content=content,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {exc}") from exc
