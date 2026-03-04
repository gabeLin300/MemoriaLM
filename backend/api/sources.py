from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.models.schemas import SourceListOut, SourceOut, SourceToggleRequest, UrlIngestRequest
from backend.modules.ingestion import ingest_uploaded_bytes, ingest_url, list_ingested_sources, set_source_enabled
from backend.services.auth import User, enforce_user_match, get_current_user
from backend.services.storage import NotebookStore

router = APIRouter()
store = NotebookStore(base_dir="data")


@router.get("/{notebook_id}/sources", response_model=SourceListOut)
def list_sources(notebook_id: str, current_user: User = Depends(get_current_user)) -> SourceListOut:
    try:
        items = list_ingested_sources(store, user_id=current_user.user_id, notebook_id=notebook_id)
        return SourceListOut(sources=items)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{notebook_id}/sources/url", response_model=SourceOut)
def ingest_source_url(
    notebook_id: str,
    payload: UrlIngestRequest,
    current_user: User = Depends(get_current_user),
) -> SourceOut:
    try:
        enforce_user_match(current_user, payload.user_id)
        return ingest_url(
            store,
            user_id=current_user.user_id,
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
    user_id: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
) -> SourceOut:
    try:
        enforce_user_match(current_user, user_id)
        content = await file.read()
        return ingest_uploaded_bytes(
            store,
            user_id=current_user.user_id,
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


@router.patch("/{notebook_id}/sources/{source_id}", response_model=SourceOut)
def toggle_source(
    notebook_id: str,
    source_id: str,
    payload: SourceToggleRequest,
    current_user: User = Depends(get_current_user),
) -> SourceOut:
    try:
        enforce_user_match(current_user, payload.user_id)
        return set_source_enabled(
            store,
            user_id=current_user.user_id,
            notebook_id=notebook_id,
            source_id=source_id,
            enabled=payload.enabled,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Source not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
