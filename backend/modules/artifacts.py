import audioop
import json
import math
import os
import re
import wave
import torch
import io
import soundfile as sf
from transformers import AutoTokenizer, VitsModel
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.models.schemas import (
    ArtifactFileOut,
    ArtifactGenerateOut,
    ArtifactListOut,
    PodcastArtifactOut,
)
from backend.services.llm import llm_service
from backend.services.storage import NotebookStore

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_vits_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _artifact_dirs(store: NotebookStore, user_id: str, notebook_id: str) -> Dict[str, Path]:
    notebook_dir = store.require_notebook_dir(user_id, notebook_id)
    root = notebook_dir / "artifacts"
    dirs = {
        "report": root / "reports",
        "quiz": root / "quizzes",
        "podcast": root / "podcasts",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _next_index(artifact_dir: Path, prefix: str) -> int:
    highest = 0
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.(?:md|mp3)$")
    for path in artifact_dir.glob(f"{prefix}_*.*"):
        match = pattern.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def _artifact_file_out(path: Path) -> ArtifactFileOut:
    created = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()
    return ArtifactFileOut(name=path.name, path=str(path.as_posix()), created_at=created)


def _collect_source_texts(store: NotebookStore, user_id: str, notebook_id: str, max_chars: int = 14000) -> List[Dict[str, str]]:
    extracted_dir = store.files_extracted_dir(user_id, notebook_id)
    sources: List[Dict[str, str]] = []
    consumed = 0
    for text_path in sorted(extracted_dir.glob("*.txt")):
        source_id = text_path.stem
        meta_path = extracted_dir / f"{source_id}.meta.json"
        source_name = text_path.name
        if meta_path.exists():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                source_name = str(payload.get("source_name") or source_name)
            except Exception:
                pass

        text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        remaining = max_chars - consumed
        if remaining <= 0:
            break
        excerpt = text[:remaining]
        consumed += len(excerpt)
        sources.append({"source_name": source_name, "text": excerpt})
    return sources


def _sources_block(sources: List[Dict[str, str]]) -> str:
    blocks = []
    for source in sources:
        blocks.append(f"[Source: {source['source_name']}]\n{source['text']}")
    return "\n\n".join(blocks)


def _llm_or_fallback(prompt: str, fallback_text: str) -> str:
    try:
        result = llm_service.generate(prompt).strip()
        if result:
            return result
    except Exception:
        pass
    return fallback_text


def _report_fallback(sources: List[Dict[str, str]], extra_prompt: Optional[str]) -> str:
    lines = [
        "# Study Report",
        "",
        "## Overview",
        "This report was generated from your notebook sources.",
    ]
    if extra_prompt:
        lines.extend(["", "## Focus", extra_prompt.strip()])
    lines.extend(["", "## Key Source Notes"])
    for src in sources[:5]:
        snippet = src["text"].replace("\n", " ")[:280].strip()
        lines.append(f"- **{src['source_name']}**: {snippet}")
    lines.extend(["", "## Conclusion", "Use the chat tab to ask follow-up questions with citations."])
    return "\n".join(lines).strip() + "\n"


def _quiz_fallback(sources: List[Dict[str, str]], num_questions: int) -> str:
    questions = []
    answers = []
    for i in range(1, num_questions + 1):
        src = sources[(i - 1) % len(sources)]
        snippet = src["text"].replace("\n", " ")[:220].strip()
        questions.append(f"{i}. Which source contains this idea: \"{snippet}\"?")
        answers.append(f"{i}. {src['source_name']}")
    return "\n".join(
        [
            "# Quiz",
            "",
            "## Questions",
            *questions,
            "",
            "## Answer Key",
            *answers,
            "",
        ]
    )


def _podcast_transcript_fallback(sources: List[Dict[str, str]], extra_prompt: Optional[str]) -> str:
    focus = extra_prompt.strip() if extra_prompt else "core concepts from the notebook"
    lines = [
        "# Podcast Transcript",
        "",
        f"Topic focus: {focus}",
        "",
        "**Host:** Welcome back. Today we cover the key ideas from your notebook.",
    ]
    for idx, src in enumerate(sources[:6], start=1):
        snippet = src["text"].replace("\n", " ")[:180].strip()
        lines.append(f"**Co-Host:** From {src['source_name']}, point {idx}: {snippet}")
        lines.append(f"**Host:** Great, and why does that matter in practice?")
    lines.append("**Co-Host:** That wraps the study summary. Review the report and quiz next.")
    lines.append("")
    return "\n".join(lines)


def _encode_pcm_to_mp3(pcm_16le: bytes, sample_rate: int, channels: int) -> bytes:
    import lameenc

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(96)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(channels)
    encoder.set_quality(2)
    return encoder.encode(pcm_16le) + encoder.flush()


def _wav_bytes_to_mp3(wav_bytes: bytes) -> bytes:
    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        frames = audioop.lin2lin(frames, sample_width, 2)
    if channels not in (1, 2):
        channels = 1
    return _encode_pcm_to_mp3(frames, sample_rate, channels)


def _is_mp3(audio_bytes: bytes) -> bool:
    if not audio_bytes:
        return False
    return audio_bytes.startswith(b"ID3") or audio_bytes[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}


def clean_transcript_for_tts(transcript: str) -> str:
    lines = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove markdown formatting and speaker labels.
        line = re.sub(r"\*\*Host:\*\*", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"\*\*Co-Host:\*\*", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"[*_#>`]", " ", line)
        line = re.sub(r'[^\x00-\x7F]+', ' ', line)
        lines.append(line)
    return " ".join(lines)


def _synthesize_podcast_mp3(transcript_text: str) -> bytes:
    tts_text = clean_transcript_for_tts(transcript_text)[:1800]

    inputs = _tokenizer(tts_text, return_tensors="pt")

    with torch.no_grad():
        waveform = _vits_model(**inputs).waveform.squeeze().cpu().numpy()

    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, waveform, 16000, format="WAV")
    wav_bytes = wav_buffer.getvalue()

    return _wav_bytes_to_mp3(wav_bytes)


def _write_markdown_artifact(artifact_dir: Path, prefix: str, content: str) -> Path:
    idx = _next_index(artifact_dir, prefix)
    path = artifact_dir / f"{prefix}_{idx}.md"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def generate_report(
    store: NotebookStore,
    *,
    user_id: str,
    notebook_id: str,
    prompt: Optional[str] = None,
) -> ArtifactGenerateOut:
    dirs = _artifact_dirs(store, user_id, notebook_id)
    sources = _collect_source_texts(store, user_id, notebook_id)
    if not sources:
        raise ValueError("No ingested sources available. Upload or ingest sources first.")

    focus = prompt.strip() if prompt else "Produce a complete study report with clear sections."
    source_context = _sources_block(sources)
    llm_prompt = (
        "Create a markdown study report grounded only in SOURCES.\n"
        "Include: title, summary, core concepts, examples, and a short conclusion.\n"
        "Cite source names inline when useful.\n\n"
        f"FOCUS:\n{focus}\n\n"
        f"SOURCES:\n{source_context}\n"
    )
    content = _llm_or_fallback(llm_prompt, _report_fallback(sources, prompt))
    out_path = _write_markdown_artifact(dirs["report"], "report", content)
    return ArtifactGenerateOut(
        artifact_type="report",
        message=f"Generated {out_path.name}",
        markdown_path=str(out_path.as_posix()),
        audio_path=None,
        created_at=_now(),
    )


def generate_quiz(
    store: NotebookStore,
    *,
    user_id: str,
    notebook_id: str,
    prompt: Optional[str] = None,
    num_questions: int = 8,
) -> ArtifactGenerateOut:
    dirs = _artifact_dirs(store, user_id, notebook_id)
    sources = _collect_source_texts(store, user_id, notebook_id)
    if not sources:
        raise ValueError("No ingested sources available. Upload or ingest sources first.")

    questions = max(3, min(15, int(num_questions)))
    focus = prompt.strip() if prompt else "Create a mixed-difficulty study quiz."
    llm_prompt = (
        "Create a markdown quiz from SOURCES.\n"
        f"Include exactly {questions} questions and a final 'Answer Key' section.\n"
        "Ground each question in source content.\n\n"
        f"FOCUS:\n{focus}\n\n"
        f"SOURCES:\n{_sources_block(sources)}\n"
    )
    content = _llm_or_fallback(llm_prompt, _quiz_fallback(sources, questions))
    out_path = _write_markdown_artifact(dirs["quiz"], "quiz", content)
    return ArtifactGenerateOut(
        artifact_type="quiz",
        message=f"Generated {out_path.name}",
        markdown_path=str(out_path.as_posix()),
        audio_path=None,
        created_at=_now(),
    )


def generate_podcast(
    store: NotebookStore,
    *,
    user_id: str,
    notebook_id: str,
    prompt: Optional[str] = None,
) -> ArtifactGenerateOut:
    dirs = _artifact_dirs(store, user_id, notebook_id)
    sources = _collect_source_texts(store, user_id, notebook_id)
    if not sources:
        raise ValueError("No ingested sources available. Upload or ingest sources first.")

    focus = prompt.strip() if prompt else "Generate a conversational podcast between two hosts."
    llm_prompt = (
        "Write a markdown podcast transcript with a two-person conversation.\n"
        "Use **Host:** and **Co-Host:** labels.\n"
        "Keep it factual and grounded in SOURCES.\n\n"
        f"FOCUS:\n{focus}\n\n"
        f"SOURCES:\n{_sources_block(sources)}\n"
    )
    transcript = _llm_or_fallback(llm_prompt, _podcast_transcript_fallback(sources, prompt))

    idx = _next_index(dirs["podcast"], "podcast")
    transcript_path = dirs["podcast"] / f"podcast_{idx}.md"
    audio_path = dirs["podcast"] / f"podcast_{idx}.mp3"

    transcript_path.write_text(transcript.strip() + "\n", encoding="utf-8")
    audio_bytes = _synthesize_podcast_mp3(transcript)
    audio_path.write_bytes(audio_bytes)

    return ArtifactGenerateOut(
        artifact_type="podcast",
        message=f"Generated podcast_{idx}.md and podcast_{idx}.mp3",
        markdown_path=str(transcript_path.as_posix()),
        audio_path=str(audio_path.as_posix()),
        created_at=_now(),
    )


def list_artifacts(store: NotebookStore, *, user_id: str, notebook_id: str) -> ArtifactListOut:
    dirs = _artifact_dirs(store, user_id, notebook_id)

    reports = [_artifact_file_out(p) for p in sorted(dirs["report"].glob("report_*.md"))]
    quizzes = [_artifact_file_out(p) for p in sorted(dirs["quiz"].glob("quiz_*.md"))]

    podcast_indices: set[int] = set()
    for path in dirs["podcast"].glob("podcast_*.*"):
        match = re.match(r"podcast_(\d+)\.(?:md|mp3)$", path.name)
        if match:
            podcast_indices.add(int(match.group(1)))

    podcasts: List[PodcastArtifactOut] = []
    for idx in sorted(podcast_indices):
        transcript_path = dirs["podcast"] / f"podcast_{idx}.md"
        audio_path = dirs["podcast"] / f"podcast_{idx}.mp3"
        podcasts.append(
            PodcastArtifactOut(
                transcript=_artifact_file_out(transcript_path) if transcript_path.exists() else None,
                audio=_artifact_file_out(audio_path) if audio_path.exists() else None,
            )
        )

    return ArtifactListOut(reports=reports, quizzes=quizzes, podcasts=podcasts)


def resolve_artifact_path(
    store: NotebookStore,
    *,
    user_id: str,
    notebook_id: str,
    artifact_type: str,
    filename: str,
) -> Path:
    if not filename or "/" in filename or "\\" in filename:
        raise ValueError("Invalid filename")

    dirs = _artifact_dirs(store, user_id, notebook_id)
    kind = artifact_type.strip().lower()
    if kind not in dirs:
        raise ValueError("Unsupported artifact type")

    path = dirs[kind] / filename
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("Artifact file not found")
    return path

