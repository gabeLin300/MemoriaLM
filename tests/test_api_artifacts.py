import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api import artifacts as artifacts_api
from backend.api import notebooks as notebooks_api
from backend.app import app
from backend.models.schemas import NotebookCreate
from backend.modules import artifacts as artifacts_module
from backend.services.storage import NotebookStore


client = TestClient(app)


def _seed_source(store: NotebookStore, user_id: str, notebook_id: str, source_id: str = "src_demo") -> None:
    extracted_dir = store.files_extracted_dir(user_id, notebook_id)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    (extracted_dir / f"{source_id}.txt").write_text(
        "Neural networks are trained by minimizing a loss function.",
        encoding="utf-8",
    )
    (extracted_dir / f"{source_id}.meta.json").write_text(
        json.dumps({"source_id": source_id, "source_name": "lesson.txt", "source_type": "txt"}),
        encoding="utf-8",
    )


def test_artifact_endpoints_generate_list_and_download(monkeypatch, tmp_path: Path):
    store = NotebookStore(base_dir=str(tmp_path))
    notebooks_api.store = store
    artifacts_api.store = store

    nb = store.create(NotebookCreate(user_id="u1", name="N1"))
    _seed_source(store, "u1", nb.notebook_id)

    monkeypatch.setattr(artifacts_module.llm_service, "generate", lambda prompt: "Artifact markdown output")
    monkeypatch.setattr(artifacts_module, "_synthesize_podcast_mp3", lambda text: b"ID3api-test")

    report = client.post(
        f"/api/notebooks/{nb.notebook_id}/artifacts/report",
        json={"user_id": "u1", "prompt": "Focus on definitions"},
    )
    assert report.status_code == 200

    podcast = client.post(
        f"/api/notebooks/{nb.notebook_id}/artifacts/podcast",
        json={"user_id": "u1"},
    )
    assert podcast.status_code == 200
    podcast_audio_name = Path(podcast.json()["audio_path"]).name

    listed = client.get(f"/api/notebooks/{nb.notebook_id}/artifacts", params={"user_id": "u1"})
    assert listed.status_code == 200
    payload = listed.json()
    assert len(payload["reports"]) == 1
    assert len(payload["podcasts"]) == 1

    dl = client.get(
        f"/api/notebooks/{nb.notebook_id}/artifacts/download",
        params={"user_id": "u1", "artifact_type": "podcast", "filename": podcast_audio_name},
    )
    assert dl.status_code == 200
    assert dl.content.startswith(b"ID3")

