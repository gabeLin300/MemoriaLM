from fastapi.testclient import TestClient

from backend.api import notebooks as notebooks_api
from backend.api import sources as sources_api
from backend.app import app
from backend.models.schemas import NotebookCreate
from backend.modules.ingestion import ingest_uploaded_bytes
from backend.services.storage import NotebookStore


client = TestClient(app)
AUTH_U1 = {"X-User-Id": "u1"}


def test_toggle_source_enabled_endpoint(monkeypatch, tmp_path):
    store = NotebookStore(base_dir=str(tmp_path))
    notebooks_api.store = store
    sources_api.store = store
    monkeypatch.setattr("backend.modules.ingestion.embedding_service.embed_texts", lambda texts: [[0.0] * 8 for _ in texts])
    monkeypatch.setattr("backend.modules.ingestion.ChromaNotebookStore.upsert_chunks", lambda self, chunks, embeddings: None)

    nb = store.create(NotebookCreate(user_id="u1", name="N1"))
    ingested = ingest_uploaded_bytes(
        store,
        user_id="u1",
        notebook_id=nb.notebook_id,
        filename="a.txt",
        content=b"hello world",
    )

    resp = client.patch(
        f"/api/notebooks/{nb.notebook_id}/sources/{ingested.source_id}",
        json={"user_id": "u1", "enabled": False},
        headers=AUTH_U1,
    )
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False

    listed = client.get(
        f"/api/notebooks/{nb.notebook_id}/sources",
        headers=AUTH_U1,
    )
    assert listed.status_code == 200
    assert listed.json()["sources"][0]["enabled"] is False
