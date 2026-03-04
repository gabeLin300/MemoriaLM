from typing import List

from fastapi import APIRouter, Depends, HTTPException

from backend.models.schemas import NotebookCreate, NotebookOut, NotebookRename
from backend.services.auth import User, enforce_user_match, get_current_user
from backend.services.storage import NotebookStore

router = APIRouter()
store = NotebookStore(base_dir="data")


@router.get("/", response_model=List[NotebookOut])
def list_notebooks(current_user: User = Depends(get_current_user)) -> List[NotebookOut]:
    try:
        return store.list(current_user.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/", response_model=NotebookOut)
def create_notebook(
    payload: NotebookCreate,
    current_user: User = Depends(get_current_user),
) -> NotebookOut:
    try:
        enforce_user_match(current_user, payload.user_id)
        return store.create(NotebookCreate(user_id=current_user.user_id, name=payload.name))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{notebook_id}", response_model=NotebookOut)
def get_notebook(notebook_id: str, current_user: User = Depends(get_current_user)) -> NotebookOut:
    try:
        notebook = store.get(current_user.user_id, notebook_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return notebook


@router.patch("/{notebook_id}", response_model=NotebookOut)
def rename_notebook(
    notebook_id: str,
    payload: NotebookRename,
    current_user: User = Depends(get_current_user),
) -> NotebookOut:
    try:
        enforce_user_match(current_user, payload.user_id)
        notebook = store.rename(current_user.user_id, notebook_id, payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return notebook


@router.delete("/{notebook_id}")
def delete_notebook(notebook_id: str, current_user: User = Depends(get_current_user)) -> dict:
    try:
        deleted = store.delete(current_user.user_id, notebook_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return {"deleted": True}
