import json
import uuid
from pathlib import Path
from typing import Optional

from backend.models.schemas import NotebookCreate, NotebookOut


class NotebookStore:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _notebook_path(self, notebook_id: str) -> Path:
        return self.base_dir / notebook_id

    def create(self, payload: NotebookCreate) -> NotebookOut:
        notebook_id = str(uuid.uuid4())
        notebook_dir = self._notebook_path(notebook_id)
        notebook_dir.mkdir(parents=True, exist_ok=False)
        data = {
            "notebook_id": notebook_id,
            "user_id": payload.user_id,
            "name": payload.name,
        }
        (notebook_dir / "meta.json").write_text(json.dumps(data, indent=2))
        return NotebookOut(**data)

    def get(self, notebook_id: str) -> Optional[NotebookOut]:
        notebook_dir = self._notebook_path(notebook_id)
        meta_path = notebook_dir / "meta.json"
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text())
        return NotebookOut(**data)

    def delete(self, notebook_id: str) -> bool:
        notebook_dir = self._notebook_path(notebook_id)
        if not notebook_dir.exists():
            return False
        for child in notebook_dir.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted(notebook_dir.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        notebook_dir.rmdir()
        return True
