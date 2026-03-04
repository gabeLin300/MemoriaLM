import json
import os
import sys
import atexit
import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import gradio as gr
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.env import load_local_env

load_local_env()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
_backend_process: subprocess.Popen | None = None


def _is_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _maybe_start_local_backend() -> None:
    global _backend_process

    if os.getenv("DISABLE_AUTO_BACKEND", "").strip().lower() in {"1", "true", "yes"}:
        return

    parsed = urlparse(BACKEND_URL)
    host = parsed.hostname
    port = parsed.port or 80
    if host not in {"127.0.0.1", "localhost"}:
        return
    if _is_port_open(host, port):
        return

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(REPO_ROOT)

    _backend_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(REPO_ROOT),
        env=env,
    )

    def _stop_backend() -> None:
        proc = _backend_process
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=6)
            except Exception:
                proc.kill()

    atexit.register(_stop_backend)

    for _ in range(25):
        if _is_port_open(host, port):
            return
        time.sleep(0.2)


def _api_request(
    method: str,
    path: str,
    *,
    params=None,
    json_body=None,
    files=None,
    data=None,
    headers=None,
    timeout: int = 60,
):
    url = f"{BACKEND_URL}{path}"
    try:
        resp = requests.request(
            method=method,
            url=url,
            params=params,
            json=json_body,
            files=files,
            data=data,
            headers=headers,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Backend unavailable at {BACKEND_URL}: {exc}") from exc

    if not resp.ok:
        detail = None
        try:
            payload = resp.json()
            detail = payload.get("detail") if isinstance(payload, dict) else None
        except Exception:
            detail = resp.text
        raise RuntimeError(f"{resp.status_code} {resp.reason}: {detail or 'request failed'}")

    if resp.content:
        return resp.json()
    return None


def _auth_headers(user_id: str | None) -> dict[str, str]:
    uid = (user_id or "").strip()
    return {"X-User-Id": uid} if uid else {}


def _format_notebook_choices(items: list[dict[str, Any]]):
    return [
        (f"{item.get('name', 'Untitled')} [{str(item.get('notebook_id', ''))[:8]}]", item.get("notebook_id"))
        for item in items
    ]


def _format_citations(citations: list[dict[str, Any]] | None) -> str:
    if not citations:
        return ""
    lines = ["", "Citations:"]
    for c in citations:
        lines.append(
            f"- [Source: {c.get('source_name', 'unknown')} | Type: {c.get('source_type', 'unknown')} | Location: {c.get('location', 'unknown')}]"
        )
    return "\n".join(lines)


def _messages_to_chatbot(messages: list[dict[str, Any]]):
    history = []
    pending_user = None
    for msg in messages:
        role = msg.get("role")
        content = str(msg.get("content", ""))
        if role == "user":
            pending_user = content
        elif role == "assistant":
            assistant_text = content + _format_citations(msg.get("citations"))
            if pending_user is None:
                history.append({"role": "assistant", "content": assistant_text})
            else:
                history.append({"role": "user", "content": pending_user})
                history.append({"role": "assistant", "content": assistant_text})
                pending_user = None
    if pending_user is not None:
        history.append({"role": "user", "content": pending_user})
    return history


def load_notebooks(user_id: str):
    user_id = (user_id or "").strip()
    if not user_id:
        return gr.Dropdown(choices=[], value=None), [], gr.JSON(value={"sources": []}), [], "Enter a user ID."
    try:
        items = _api_request("GET", "/api/notebooks/", params={"user_id": user_id}, headers=_auth_headers(user_id))
        choices = _format_notebook_choices(items)
        selected = choices[0][1] if choices else None
        chat_history = []
        sources_payload = {"sources": []}
        if selected:
            chat = _api_request(
                "GET",
                f"/api/notebooks/{selected}/chat",
                params={"user_id": user_id},
                headers=_auth_headers(user_id),
            )
            chat_history = _messages_to_chatbot(chat.get("messages", []))
            sources_payload = _api_request(
                "GET",
                f"/api/notebooks/{selected}/sources",
                params={"user_id": user_id},
                headers=_auth_headers(user_id),
            )
        return gr.Dropdown(choices=choices, value=selected), items, gr.JSON(value=sources_payload), chat_history, ""
    except Exception as exc:
        return gr.Dropdown(choices=[], value=None), [], gr.JSON(value={"sources": []}), [], str(exc)


def create_notebook(user_id: str, notebook_name: str):
    user_id = (user_id or "").strip()
    notebook_name = (notebook_name or "").strip() or "Untitled Notebook"
    if not user_id:
        return gr.Dropdown(choices=[], value=None), [], "", "Enter a user ID first."
    try:
        _api_request(
            "POST",
            "/api/notebooks/",
            json_body={"user_id": user_id, "name": notebook_name},
            headers=_auth_headers(user_id),
        )
        dropdown, notebooks, _sources, _chat, _status = load_notebooks(user_id)
        return dropdown, notebooks, "", f"Created notebook '{notebook_name}'."
    except Exception as exc:
        return gr.Dropdown(), gr.skip(), notebook_name, str(exc)


def rename_notebook(user_id: str, notebook_id: str, notebook_name: str):
    user_id = (user_id or "").strip()
    notebook_name = (notebook_name or "").strip()
    if not user_id or not notebook_id or not notebook_name:
        return gr.Dropdown(), gr.skip(), "Provide user, notebook, and new name."
    try:
        _api_request(
            "PATCH",
            f"/api/notebooks/{notebook_id}",
            json_body={"user_id": user_id, "name": notebook_name},
            headers=_auth_headers(user_id),
        )
        items = _api_request("GET", "/api/notebooks/", params={"user_id": user_id}, headers=_auth_headers(user_id))
        choices = _format_notebook_choices(items)
        return gr.Dropdown(choices=choices, value=notebook_id), items, f"Renamed notebook to '{notebook_name}'."
    except Exception as exc:
        return gr.Dropdown(), gr.skip(), str(exc)


def delete_notebook(user_id: str, notebook_id: str):
    user_id = (user_id or "").strip()
    if not user_id or not notebook_id:
        return gr.Dropdown(choices=[], value=None), [], gr.JSON(value={"sources": []}), [], "Select a notebook to delete."
    try:
        _api_request(
            "DELETE",
            f"/api/notebooks/{notebook_id}",
            params={"user_id": user_id},
            headers=_auth_headers(user_id),
        )
        dropdown, notebooks, sources_json, chat_history, _ = load_notebooks(user_id)
        return dropdown, notebooks, sources_json, chat_history, "Deleted notebook."
    except Exception as exc:
        return gr.Dropdown(), gr.skip(), gr.skip(), gr.skip(), str(exc)


def on_notebook_change(user_id: str, notebook_id: str):
    user_id = (user_id or "").strip()
    if not user_id or not notebook_id:
        return gr.JSON(value={"sources": []}), [], ""
    try:
        sources_payload = _api_request(
            "GET",
            f"/api/notebooks/{notebook_id}/sources",
            params={"user_id": user_id},
            headers=_auth_headers(user_id),
        )
        chat = _api_request(
            "GET",
            f"/api/notebooks/{notebook_id}/chat",
            params={"user_id": user_id},
            headers=_auth_headers(user_id),
        )
        return gr.JSON(value=sources_payload), _messages_to_chatbot(chat.get("messages", [])), ""
    except Exception as exc:
        return gr.JSON(value={"sources": []}), [], str(exc)


def upload_source(user_id: str, notebook_id: str, file_path: str):
    user_id = (user_id or "").strip()
    if not user_id or not notebook_id or not file_path:
        return gr.JSON(value={"sources": []}), "Choose a file and notebook first."
    try:
        with open(file_path, "rb") as f:
            payload = _api_request(
                "POST",
                f"/api/notebooks/{notebook_id}/sources/upload",
                data={"user_id": user_id},
                files={"file": (os.path.basename(file_path), f)},
                headers=_auth_headers(user_id),
                timeout=180,
            )
        sources_payload = _api_request(
            "GET",
            f"/api/notebooks/{notebook_id}/sources",
            params={"user_id": user_id},
            headers=_auth_headers(user_id),
        )
        return gr.JSON(value=sources_payload), f"Ingested file: {payload.get('source_name', os.path.basename(file_path))}"
    except Exception as exc:
        return gr.JSON(value={"sources": []}), str(exc)


def ingest_url_source(user_id: str, notebook_id: str, url: str):
    user_id = (user_id or "").strip()
    url = (url or "").strip()
    if not user_id or not notebook_id or not url:
        return gr.JSON(value={"sources": []}), "Provide a URL and select a notebook."
    try:
        payload = _api_request(
            "POST",
            f"/api/notebooks/{notebook_id}/sources/url",
            json_body={"user_id": user_id, "url": url},
            headers=_auth_headers(user_id),
            timeout=180,
        )
        sources_payload = _api_request(
            "GET",
            f"/api/notebooks/{notebook_id}/sources",
            params={"user_id": user_id},
            headers=_auth_headers(user_id),
        )
        return gr.JSON(value=sources_payload), f"Ingested URL: {payload.get('source_name', url)}"
    except Exception as exc:
        return gr.JSON(value={"sources": []}), str(exc)


def _source_choices_from_payload(payload: dict[str, Any]) -> list[tuple[str, str]]:
    sources = payload.get("sources", []) if isinstance(payload, dict) else []
    choices: list[tuple[str, str]] = []
    for src in sources:
        source_id = str(src.get("source_id", ""))
        if not source_id:
            continue
        name = str(src.get("source_name", source_id))
        choices.append((f"{name} [{source_id[:8]}]", source_id))
    return choices


def _enabled_source_ids_from_payload(payload: dict[str, Any]) -> list[str]:
    if not isinstance(payload, dict):
        return []
    enabled_ids: list[str] = []
    for src in payload.get("sources", []):
        source_id = str(src.get("source_id", ""))
        if source_id and bool(src.get("enabled", True)):
            enabled_ids.append(source_id)
    return enabled_ids


def refresh_source_controls(sources_payload: dict[str, Any]):
    choices = _source_choices_from_payload(sources_payload or {"sources": []})
    enabled_values = _enabled_source_ids_from_payload(sources_payload or {"sources": []})
    return gr.CheckboxGroup(choices=choices, value=enabled_values, interactive=bool(choices))


def apply_source_selection(
    user_id: str,
    notebook_id: str,
    enabled_source_ids: list[str] | None,
    sources_payload: dict[str, Any],
):
    user_id = (user_id or "").strip()
    if not user_id or not notebook_id:
        payload = {"sources": []}
        return gr.JSON(value=payload), payload, gr.CheckboxGroup(choices=[], value=[]), "Select a notebook first."
    selected = set(enabled_source_ids or [])
    try:
        current_sources = (sources_payload or {}).get("sources", [])
        changed = 0
        for src in current_sources:
            source_id = str(src.get("source_id", "")).strip()
            if not source_id:
                continue
            current_enabled = bool(src.get("enabled", True))
            target_enabled = source_id in selected
            if current_enabled == target_enabled:
                continue
            _api_request(
                "PATCH",
                f"/api/notebooks/{notebook_id}/sources/{source_id}",
                json_body={"user_id": user_id, "enabled": target_enabled},
                headers=_auth_headers(user_id),
            )
            changed += 1

        sources_payload = _api_request(
            "GET",
            f"/api/notebooks/{notebook_id}/sources",
            params={"user_id": user_id},
            headers=_auth_headers(user_id),
        )
        source_group = refresh_source_controls(sources_payload)
        return gr.JSON(value=sources_payload), sources_payload, source_group, f"Updated {changed} source setting(s)."
    except Exception as exc:
        payload = {"sources": []}
        return gr.JSON(value=payload), payload, gr.CheckboxGroup(choices=[], value=[]), str(exc)


def send_message(message: str, history, user_id: str, notebook_id: str):
    message = (message or "").strip()
    user_id = (user_id or "").strip()
    history = history or []
    if not message:
        return "", history, ""
    if not user_id or not notebook_id:
        return "", history, "Provide user ID and select a notebook first."
    try:
        resp = _api_request(
            "POST",
            f"/api/notebooks/{notebook_id}/chat",
            json_body={"user_id": user_id, "message": message, "top_k": 5},
            headers=_auth_headers(user_id),
            timeout=180,
        )
        assistant_text = str(resp.get("answer", "")) + _format_citations(resp.get("citations"))
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": assistant_text},
        ]
        return "", history, ""
    except Exception as exc:
        return "", history, str(exc)


def _empty_artifacts_payload() -> dict[str, Any]:
    return {"reports": [], "quizzes": [], "podcasts": []}


def _artifact_outputs_from_payload(payload: dict[str, Any]):
    reports = payload.get("reports") or []
    quizzes = payload.get("quizzes") or []
    podcasts = payload.get("podcasts") or []

    latest_report = reports[-1].get("path") if reports else None
    latest_quiz = quizzes[-1].get("path") if quizzes else None

    latest_transcript = None
    latest_audio = None
    if podcasts:
        last = podcasts[-1] or {}
        transcript = last.get("transcript") or {}
        audio = last.get("audio") or {}
        latest_transcript = transcript.get("path")
        latest_audio = audio.get("path")
    return latest_report, latest_quiz, latest_transcript, latest_audio


def refresh_artifacts(user_id: str, notebook_id: str):
    user_id = (user_id or "").strip()
    if not user_id or not notebook_id:
        payload = _empty_artifacts_payload()
        return gr.JSON(value=payload), None, None, None, None, "Select a notebook first."
    try:
        payload = _api_request(
            "GET",
            f"/api/notebooks/{notebook_id}/artifacts",
            params={"user_id": user_id},
            headers=_auth_headers(user_id),
        )
        report, quiz, transcript, audio = _artifact_outputs_from_payload(payload)
        return gr.JSON(value=payload), report, quiz, transcript, audio, ""
    except Exception as exc:
        payload = _empty_artifacts_payload()
        return gr.JSON(value=payload), None, None, None, None, str(exc)


def sync_artifacts_on_notebook_change(user_id: str, notebook_id: str):
    payload_json, report, quiz, transcript, audio, _status = refresh_artifacts(user_id, notebook_id)
    return payload_json, report, quiz, transcript, audio


def generate_report_artifact(user_id: str, notebook_id: str, artifact_prompt: str):
    user_id = (user_id or "").strip()
    if not user_id or not notebook_id:
        payload = _empty_artifacts_payload()
        return gr.JSON(value=payload), None, None, None, None, "Select a notebook first."
    try:
        resp = _api_request(
            "POST",
            f"/api/notebooks/{notebook_id}/artifacts/report",
            json_body={"user_id": user_id, "prompt": (artifact_prompt or "").strip() or None},
            headers=_auth_headers(user_id),
            timeout=180,
        )
        payload_json, report, quiz, transcript, audio, _ = refresh_artifacts(user_id, notebook_id)
        return payload_json, report, quiz, transcript, audio, str(resp.get("message", "Generated report."))
    except Exception as exc:
        payload = _empty_artifacts_payload()
        return gr.JSON(value=payload), None, None, None, None, str(exc)


def generate_quiz_artifact(user_id: str, notebook_id: str, artifact_prompt: str, num_questions: float):
    user_id = (user_id or "").strip()
    if not user_id or not notebook_id:
        payload = _empty_artifacts_payload()
        return gr.JSON(value=payload), None, None, None, None, "Select a notebook first."
    try:
        resp = _api_request(
            "POST",
            f"/api/notebooks/{notebook_id}/artifacts/quiz",
            json_body={
                "user_id": user_id,
                "prompt": (artifact_prompt or "").strip() or None,
                "num_questions": int(num_questions),
            },
            headers=_auth_headers(user_id),
            timeout=180,
        )
        payload_json, report, quiz, transcript, audio, _ = refresh_artifacts(user_id, notebook_id)
        return payload_json, report, quiz, transcript, audio, str(resp.get("message", "Generated quiz."))
    except Exception as exc:
        payload = _empty_artifacts_payload()
        return gr.JSON(value=payload), None, None, None, None, str(exc)


def generate_podcast_artifact(user_id: str, notebook_id: str, artifact_prompt: str):
    user_id = (user_id or "").strip()
    if not user_id or not notebook_id:
        payload = _empty_artifacts_payload()
        return gr.JSON(value=payload), None, None, None, None, "Select a notebook first."
    try:
        resp = _api_request(
            "POST",
            f"/api/notebooks/{notebook_id}/artifacts/podcast",
            json_body={"user_id": user_id, "prompt": (artifact_prompt or "").strip() or None},
            headers=_auth_headers(user_id),
            timeout=240,
        )
        payload_json, report, quiz, transcript, audio, _ = refresh_artifacts(user_id, notebook_id)
        return payload_json, report, quiz, transcript, audio, str(resp.get("message", "Generated podcast."))
    except Exception as exc:
        payload = _empty_artifacts_payload()
        return gr.JSON(value=payload), None, None, None, None, str(exc)
    
def greet_user(profile: gr.OAuthProfile | None) -> tuple[str | None, str]:
    if profile is None:
        return None, "User not logged in."
    return profile.username, f"Welcome {profile.username}"


with gr.Blocks(title="MemoriaLM") as demo:
    notebook_state = gr.State([])
    sources_payload_state = gr.State({"sources": []})

    gr.Markdown("# MemoriaLM")
    gr.Markdown("NotebookLM-style RAG app (Phase 4 UI wired to backend APIs)")
    gr.LoginButton()
    user_id = gr.State()
    login_message = gr.Markdown()
    demo.load(greet_user, outputs=[user_id, login_message])

    if user_id is None:
        gr.Markdown("Please log in to access your notebooks.")
        gr.update(visible=False)

    with gr.Row():
        backend_url_info = gr.Textbox(label="Backend URL", value=BACKEND_URL, interactive=False)

    status_box = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            with gr.Accordion("Notebook Management", open=True, elem_classes=["card"]):
                notebook_name = gr.Textbox(label="Notebook name")
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    create_btn = gr.Button("Create")
                notebook_selector = gr.Dropdown(label="Notebook", choices=[], value=None)
                with gr.Row():
                    rename_btn = gr.Button("Rename")
                    delete_btn = gr.Button("Delete")
    
            gr.Markdown("## Sources")
            upload_file = gr.File(label="Upload PDF/PPTX/TXT", type="filepath")
            upload_btn = gr.Button("Ingest File")
            url_input = gr.Textbox(label="Web URL", placeholder="https://...")
            url_btn = gr.Button("Ingest URL")
            sources_json = gr.JSON(label="Ingested Sources", value={"sources": []})
            source_enabled_group = gr.CheckboxGroup(label="Enabled sources for RAG", choices=[], value=[], interactive=False)
            toggle_source_btn = gr.Button("Apply Source Selection")

        with gr.Column(scale=2, min_width=420):
            with gr.Tabs():
                with gr.Tab("Chat"):
                    gr.Markdown("## Chat")
                    chatbot = gr.Chatbot(height=480)
                    with gr.Row():
                        message = gr.Textbox(label="Message", placeholder="Ask about your sources...", scale=4)
                        send_btn = gr.Button("Send", scale=1)
                with gr.Tab("Artifacts"):
                    gr.Markdown("## Artifacts")
                    artifact_prompt = gr.Textbox(
                        label="Artifact focus prompt (optional)",
                        placeholder="Focus on topic X and how it relates to topic Y",
                    )
                    quiz_questions = gr.Slider(minimum=3, maximum=15, step=1, value=8, label="Quiz questions")
                    with gr.Row():
                        refresh_artifacts_btn = gr.Button("Refresh Artifacts")
                        report_btn = gr.Button("Generate Report")
                    with gr.Row():
                        quiz_btn = gr.Button("Generate Quiz")
                        podcast_btn = gr.Button("Generate Podcast")
                    artifacts_json = gr.JSON(label="Generated Artifacts", value=_empty_artifacts_payload())
                    latest_report_file = gr.File(label="Latest Report (.md)", interactive=False)
                    latest_quiz_file = gr.File(label="Latest Quiz (.md)", interactive=False)
                    latest_podcast_transcript_file = gr.File(label="Latest Podcast Transcript (.md)", interactive=False)
                    latest_podcast_audio = gr.Audio(label="Latest Podcast Audio (.mp3)", type="filepath", interactive=False)

    refresh_evt = refresh_btn.click(
        load_notebooks,
        inputs=[user_id],
        outputs=[notebook_selector, notebook_state, sources_json, chatbot, status_box],
    )
    refresh_evt.then(
        lambda payload: payload,
        inputs=[sources_json],
        outputs=[sources_payload_state],
    ).then(
        refresh_source_controls,
        inputs=[sources_payload_state],
        outputs=[source_enabled_group],
    )
    user_change_evt = user_id.change(
        load_notebooks,
        inputs=[user_id],
        outputs=[notebook_selector, notebook_state, sources_json, chatbot, status_box],
    )
    user_change_evt.then(
        lambda payload: payload,
        inputs=[sources_json],
        outputs=[sources_payload_state],
    ).then(
        refresh_source_controls,
        inputs=[sources_payload_state],
        outputs=[source_enabled_group],
    )

    create_btn.click(
        create_notebook,
        inputs=[user_id, notebook_name],
        outputs=[notebook_selector, notebook_state, notebook_name, status_box],
    )
    rename_btn.click(
        rename_notebook,
        inputs=[user_id, notebook_selector, notebook_name],
        outputs=[notebook_selector, notebook_state, status_box],
    )
    delete_btn.click(
        delete_notebook,
        inputs=[user_id, notebook_selector],
        outputs=[notebook_selector, notebook_state, sources_json, chatbot, status_box],
    )

    notebook_selector.change(
        on_notebook_change,
        inputs=[user_id, notebook_selector],
        outputs=[sources_json, chatbot, status_box],
    )
    notebook_selector.change(
        lambda payload: payload,
        inputs=[sources_json],
        outputs=[sources_payload_state],
    ).then(
        refresh_source_controls,
        inputs=[sources_payload_state],
        outputs=[source_enabled_group],
    )
    notebook_selector.change(
        sync_artifacts_on_notebook_change,
        inputs=[user_id, notebook_selector],
        outputs=[
            artifacts_json,
            latest_report_file,
            latest_quiz_file,
            latest_podcast_transcript_file,
            latest_podcast_audio,
        ],
    )

    refresh_evt.then(
        sync_artifacts_on_notebook_change,
        inputs=[user_id, notebook_selector],
        outputs=[
            artifacts_json,
            latest_report_file,
            latest_quiz_file,
            latest_podcast_transcript_file,
            latest_podcast_audio,
        ],
    )
    user_change_evt.then(
        sync_artifacts_on_notebook_change,
        inputs=[user_id, notebook_selector],
        outputs=[
            artifacts_json,
            latest_report_file,
            latest_quiz_file,
            latest_podcast_transcript_file,
            latest_podcast_audio,
        ],
    )

    upload_btn.click(
        upload_source,
        inputs=[user_id, notebook_selector, upload_file],
        outputs=[sources_json, status_box],
    )
    upload_btn.click(
        lambda payload: payload,
        inputs=[sources_json],
        outputs=[sources_payload_state],
    ).then(
        refresh_source_controls,
        inputs=[sources_payload_state],
        outputs=[source_enabled_group],
    )
    url_btn.click(
        ingest_url_source,
        inputs=[user_id, notebook_selector, url_input],
        outputs=[sources_json, status_box],
    )
    url_btn.click(
        lambda payload: payload,
        inputs=[sources_json],
        outputs=[sources_payload_state],
    ).then(
        refresh_source_controls,
        inputs=[sources_payload_state],
        outputs=[source_enabled_group],
    )

    toggle_source_btn.click(
        apply_source_selection,
        inputs=[user_id, notebook_selector, source_enabled_group, sources_payload_state],
        outputs=[sources_json, sources_payload_state, source_enabled_group, status_box],
    )

    refresh_artifacts_btn.click(
        refresh_artifacts,
        inputs=[user_id, notebook_selector],
        outputs=[
            artifacts_json,
            latest_report_file,
            latest_quiz_file,
            latest_podcast_transcript_file,
            latest_podcast_audio,
            status_box,
        ],
    )
    report_btn.click(
        generate_report_artifact,
        inputs=[user_id, notebook_selector, artifact_prompt],
        outputs=[
            artifacts_json,
            latest_report_file,
            latest_quiz_file,
            latest_podcast_transcript_file,
            latest_podcast_audio,
            status_box,
        ],
    )
    quiz_btn.click(
        generate_quiz_artifact,
        inputs=[user_id, notebook_selector, artifact_prompt, quiz_questions],
        outputs=[
            artifacts_json,
            latest_report_file,
            latest_quiz_file,
            latest_podcast_transcript_file,
            latest_podcast_audio,
            status_box,
        ],
    )
    podcast_btn.click(
        generate_podcast_artifact,
        inputs=[user_id, notebook_selector, artifact_prompt],
        outputs=[
            artifacts_json,
            latest_report_file,
            latest_quiz_file,
            latest_podcast_transcript_file,
            latest_podcast_audio,
            status_box,
        ],
    )

    send_btn.click(
        send_message,
        inputs=[message, chatbot, user_id, notebook_selector],
        outputs=[message, chatbot, status_box],
    )
    message.submit(
        send_message,
        inputs=[message, chatbot, user_id, notebook_selector],
        outputs=[message, chatbot, status_box],
    )


if __name__ == "__main__":
    _maybe_start_local_backend()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        ssr_mode=False,
    )
