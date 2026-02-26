from typing import List

from fastapi import APIRouter, HTTPException, Query

from backend.models.schemas import NotebookCreate, NotebookOut, NotebookRename
from backend.services.storage import NotebookStore

router = APIRouter()
store = NotebookStore(base_dir="data")


@router.get("/", response_model=List[NotebookOut])
def list_notebooks(user_id: str = Query(...)) -> List[NotebookOut]:
    try:
        return store.list(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/", response_model=NotebookOut)
def create_notebook(payload: NotebookCreate) -> NotebookOut:
    try:
        return store.create(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{notebook_id}", response_model=NotebookOut)
def get_notebook(notebook_id: str, user_id: str = Query(...)) -> NotebookOut:
    try:
        notebook = store.get(user_id, notebook_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return notebook


@router.patch("/{notebook_id}", response_model=NotebookOut)
def rename_notebook(notebook_id: str, payload: NotebookRename) -> NotebookOut:
    try:
        notebook = store.rename(payload.user_id, notebook_id, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return notebook


@router.delete("/{notebook_id}")
def delete_notebook(notebook_id: str, user_id: str = Query(...)) -> dict:
    try:
        deleted = store.delete(user_id, notebook_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return {"deleted": True}
