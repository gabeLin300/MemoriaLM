import json
from pathlib import Path

from backend.models.schemas import NotebookCreate
from backend.modules import artifacts
from backend.services.storage import NotebookStore


def _seed_source(store: NotebookStore, user_id: str, notebook_id: str, source_id: str = "src_demo") -> None:
    extracted_dir = store.files_extracted_dir(user_id, notebook_id)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    (extracted_dir / f"{source_id}.txt").write_text(
        "Machine learning uses data-driven methods to make predictions.",
        encoding="utf-8",
    )
    (extracted_dir / f"{source_id}.meta.json").write_text(
        json.dumps(
            {
                "source_id": source_id,
                "source_name": "sample.txt",
                "source_type": "txt",
            }
        ),
        encoding="utf-8",
    )


def test_generate_report_quiz_and_list(monkeypatch, tmp_path: Path):
    store = NotebookStore(base_dir=str(tmp_path))
    nb = store.create(NotebookCreate(user_id="u1", name="N1"))
    _seed_source(store, "u1", nb.notebook_id)

    monkeypatch.setattr(artifacts.llm_service, "generate", lambda prompt: "Generated markdown")

    report = artifacts.generate_report(store, user_id="u1", notebook_id=nb.notebook_id, prompt="Focus")
    quiz = artifacts.generate_quiz(store, user_id="u1", notebook_id=nb.notebook_id, num_questions=5)

    assert report.artifact_type == "report"
    assert quiz.artifact_type == "quiz"
    assert Path(report.markdown_path).exists()
    assert Path(quiz.markdown_path).exists()

    listed = artifacts.list_artifacts(store, user_id="u1", notebook_id=nb.notebook_id)
    assert len(listed.reports) == 1
    assert len(listed.quizzes) == 1
    assert listed.podcasts == []


def test_generate_podcast_writes_md_and_mp3(monkeypatch, tmp_path: Path):
    store = NotebookStore(base_dir=str(tmp_path))
    nb = store.create(NotebookCreate(user_id="u1", name="N1"))
    _seed_source(store, "u1", nb.notebook_id)

    monkeypatch.setattr(artifacts.llm_service, "generate", lambda prompt: "**Host:** Hi\n**Co-Host:** Hello")
    monkeypatch.setattr(artifacts, "_synthesize_podcast_mp3", lambda text: b"ID3mock-mp3-bytes")

    podcast = artifacts.generate_podcast(store, user_id="u1", notebook_id=nb.notebook_id)
    transcript_path = Path(podcast.markdown_path)
    audio_path = Path(podcast.audio_path or "")

    assert podcast.artifact_type == "podcast"
    assert transcript_path.exists()
    assert audio_path.exists()
    assert audio_path.read_bytes().startswith(b"ID3")

    listed = artifacts.list_artifacts(store, user_id="u1", notebook_id=nb.notebook_id)
    assert len(listed.podcasts) == 1
    assert listed.podcasts[0].transcript is not None
    assert listed.podcasts[0].audio is not None

