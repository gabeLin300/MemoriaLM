import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from backend.models.schemas import NotebookCreate, NotebookOut


class NotebookStore:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _validate_user_id(self, user_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9._@-]+", user_id):
            raise ValueError("Invalid user_id")
        return user_id

    def _users_root(self) -> Path:
        return self.base_dir / "users"

    def _user_root(self, user_id: str) -> Path:
        return self._users_root() / self._validate_user_id(user_id)

    def _notebooks_root(self, user_id: str) -> Path:
        return self._user_root(user_id) / "notebooks"

    def _index_path(self, user_id: str) -> Path:
        return self._notebooks_root(user_id) / "index.json"

    def _notebook_path(self, user_id: str, notebook_id: str) -> Path:
        return self._notebooks_root(user_id) / notebook_id

    def _meta_path(self, user_id: str, notebook_id: str) -> Path:
        return self._notebook_path(user_id, notebook_id) / "meta.json"

    def _now(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _ensure_user_tree(self, user_id: str) -> None:
        self._notebooks_root(user_id).mkdir(parents=True, exist_ok=True)

    def _read_json(self, path: Path, default):
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, data: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_index(self, user_id: str) -> List[dict]:
        self._ensure_user_tree(user_id)
        data = self._read_json(self._index_path(user_id), default=[])
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def _save_index(self, user_id: str, items: List[dict]) -> None:
        self._write_json(self._index_path(user_id), items)

    def _initialize_notebook_dirs(self, notebook_dir: Path) -> None:
        for rel in [
            "files_raw",
            "files_extracted",
            "chroma",
            "chat",
            "artifacts/reports",
            "artifacts/quizzes",
            "artifacts/podcasts",
        ]:
            (notebook_dir / rel).mkdir(parents=True, exist_ok=True)
        (notebook_dir / "chat" / "messages.jsonl").touch(exist_ok=True)

    def create(self, payload: NotebookCreate) -> NotebookOut:
        user_id = self._validate_user_id(payload.user_id)
        self._ensure_user_tree(user_id)

        notebook_id = str(uuid.uuid4())
        notebook_dir = self._notebook_path(user_id, notebook_id)
        notebook_dir.mkdir(parents=True, exist_ok=False)
        self._initialize_notebook_dirs(notebook_dir)

        now = self._now()
        data = {
            "notebook_id": notebook_id,
            "user_id": user_id,
            "name": payload.name,
            "created_at": now,
            "updated_at": now,
        }
        self._write_json(notebook_dir / "meta.json", data)

        index = self._load_index(user_id)
        index.append(data)
        self._save_index(user_id, index)
        return NotebookOut(**data)

    def list(self, user_id: str) -> List[NotebookOut]:
        self._validate_user_id(user_id)
        return [NotebookOut(**item) for item in self._load_index(user_id)]

    def get(self, user_id: str, notebook_id: str) -> Optional[NotebookOut]:
        user_id = self._validate_user_id(user_id)
        meta_path = self._meta_path(user_id, notebook_id)
        if not meta_path.exists():
            return None
        data = self._read_json(meta_path, default=None)
        if not isinstance(data, dict) or data.get("user_id") != user_id:
            return None
        return NotebookOut(**data)

    def require_notebook_dir(self, user_id: str, notebook_id: str) -> Path:
        notebook = self.get(user_id, notebook_id)
        if notebook is None:
            raise FileNotFoundError("Notebook not found")
        return self._notebook_path(user_id, notebook_id)

    def files_raw_dir(self, user_id: str, notebook_id: str) -> Path:
        return self.require_notebook_dir(user_id, notebook_id) / "files_raw"

    def files_extracted_dir(self, user_id: str, notebook_id: str) -> Path:
        return self.require_notebook_dir(user_id, notebook_id) / "files_extracted"

    def chroma_dir(self, user_id: str, notebook_id: str) -> Path:
        return self.require_notebook_dir(user_id, notebook_id) / "chroma"

    def chat_messages_path(self, user_id: str, notebook_id: str) -> Path:
        return self.require_notebook_dir(user_id, notebook_id) / "chat" / "messages.jsonl"

    def append_chat_message(self, user_id: str, notebook_id: str, message: dict) -> None:
        path = self.chat_messages_path(user_id, notebook_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message) + "\n")

    def read_chat_messages(self, user_id: str, notebook_id: str) -> List[dict]:
        path = self.chat_messages_path(user_id, notebook_id)
        if not path.exists():
            return []
        messages: List[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                messages.append(payload)
        return messages

    def rename(self, user_id: str, notebook_id: str, name: str) -> Optional[NotebookOut]:
        notebook = self.get(user_id, notebook_id)
        if notebook is None:
            return None

        updated = notebook.model_dump()
        updated["name"] = name
        updated["updated_at"] = self._now()
        self._write_json(self._meta_path(user_id, notebook_id), updated)

        index = self._load_index(user_id)
        for item in index:
            if item.get("notebook_id") == notebook_id:
                item.update(updated)
                break
        self._save_index(user_id, index)
        return NotebookOut(**updated)

    def delete(self, user_id: str, notebook_id: str) -> bool:
        user_id = self._validate_user_id(user_id)
        notebook_dir = self._notebook_path(user_id, notebook_id)
        if not notebook_dir.exists() or self.get(user_id, notebook_id) is None:
            return False

        shutil.rmtree(notebook_dir)

        index = [
            item
            for item in self._load_index(user_id)
            if item.get("notebook_id") != notebook_id
        ]
        self._save_index(user_id, index)
        return True
