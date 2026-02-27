from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.models.schemas import ArtifactGenerateOut, ArtifactGenerateRequest, ArtifactListOut
from backend.modules.artifacts import (
    generate_podcast,
    generate_quiz,
    generate_report,
    list_artifacts,
    resolve_artifact_path,
)
from backend.services.storage import NotebookStore

router = APIRouter()
store = NotebookStore(base_dir="data")


@router.get("/{notebook_id}/artifacts", response_model=ArtifactListOut)
def list_notebook_artifacts(notebook_id: str, user_id: str = Query(...)) -> ArtifactListOut:
    try:
        return list_artifacts(store, user_id=user_id, notebook_id=notebook_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{notebook_id}/artifacts/report", response_model=ArtifactGenerateOut)
def create_report_artifact(notebook_id: str, payload: ArtifactGenerateRequest) -> ArtifactGenerateOut:
    try:
        return generate_report(
            store,
            user_id=payload.user_id,
            notebook_id=notebook_id,
            prompt=payload.prompt,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc


@router.post("/{notebook_id}/artifacts/quiz", response_model=ArtifactGenerateOut)
def create_quiz_artifact(notebook_id: str, payload: ArtifactGenerateRequest) -> ArtifactGenerateOut:
    try:
        return generate_quiz(
            store,
            user_id=payload.user_id,
            notebook_id=notebook_id,
            prompt=payload.prompt,
            num_questions=payload.num_questions,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {exc}") from exc


@router.post("/{notebook_id}/artifacts/podcast", response_model=ArtifactGenerateOut)
def create_podcast_artifact(notebook_id: str, payload: ArtifactGenerateRequest) -> ArtifactGenerateOut:
    try:
        return generate_podcast(
            store,
            user_id=payload.user_id,
            notebook_id=notebook_id,
            prompt=payload.prompt,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Podcast generation failed: {exc}") from exc


@router.get("/{notebook_id}/artifacts/download")
def download_artifact(
    notebook_id: str,
    user_id: str = Query(...),
    artifact_type: str = Query(...),
    filename: str = Query(...),
):
    try:
        path = resolve_artifact_path(
            store,
            user_id=user_id,
            notebook_id=notebook_id,
            artifact_type=artifact_type,
            filename=filename,
        )
        media_type = "audio/mpeg" if path.suffix.lower() == ".mp3" else "text/markdown"
        return FileResponse(path=path, filename=path.name, media_type=media_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artifact not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

