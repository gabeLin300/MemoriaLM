from fastapi import APIRouter, HTTPException
from backend.models.schemas import NotebookCreate, NotebookOut
from backend.services.storage import NotebookStore

router = APIRouter()
store = NotebookStore(base_dir="data")


@router.post("/", response_model=NotebookOut)
def create_notebook(payload: NotebookCreate) -> NotebookOut:
    return store.create(payload)


@router.get("/{notebook_id}", response_model=NotebookOut)
def get_notebook(notebook_id: str) -> NotebookOut:
    notebook = store.get(notebook_id)
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return notebook


@router.delete("/{notebook_id}")
def delete_notebook(notebook_id: str) -> dict:
    deleted = store.delete(notebook_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return {"deleted": True}
