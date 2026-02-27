from fastapi.testclient import TestClient

from backend.api import chat as chat_api
from backend.api import notebooks as notebooks_api
from backend.app import app
from backend.models.schemas import NotebookCreate
from backend.services.storage import NotebookStore


client = TestClient(app)


def test_chat_endpoint_success_with_mock(monkeypatch, tmp_path):
    store = NotebookStore(base_dir=str(tmp_path))
    notebooks_api.store = store
    chat_api.store = store
    created = store.create(NotebookCreate(user_id="u1", name="N1"))

    monkeypatch.setattr(
        chat_api,
        "answer_notebook_question",
        lambda *args, **kwargs: {
            "answer": "Mock answer",
            "citations": [
                {
                    "source_name": "a.txt",
                    "source_type": "txt",
                    "location": "full text",
                    "chunk_id": "chunk-1",
                }
            ],
            "used_chunks": 1,
        },
    )

    resp = client.post(
        f"/api/notebooks/{created.notebook_id}/chat",
        json={"user_id": "u1", "message": "Hi", "top_k": 3},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Mock answer"
    assert data["used_chunks"] == 1
    assert data["citations"][0]["source_name"] == "a.txt"


def test_chat_history_endpoint_reads_jsonl(tmp_path):
    store = NotebookStore(base_dir=str(tmp_path))
    notebooks_api.store = store
    chat_api.store = store
    created = store.create(NotebookCreate(user_id="u1", name="N1"))
    store.append_chat_message("u1", created.notebook_id, {"role": "user", "content": "Q", "created_at": "2026-01-01T00:00:00+00:00"})
    store.append_chat_message(
        "u1",
        created.notebook_id,
        {
            "role": "assistant",
            "content": "A",
            "created_at": "2026-01-01T00:00:01+00:00",
            "citations": [],
        },
    )

    resp = client.get(f"/api/notebooks/{created.notebook_id}/chat", params={"user_id": "u1"})
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["messages"]) == 2
    assert payload["messages"][1]["role"] == "assistant"
