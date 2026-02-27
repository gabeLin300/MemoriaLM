from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.modules.ingestion import (
    chunk_text,
    extract_file_segments,
    fetch_url_text,
    format_extracted_text,
    ingest_uploaded_bytes,
)
from backend.models.schemas import NotebookCreate
from backend.services.storage import NotebookStore


def test_chunk_text_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith("ij")
    assert chunks[-1]


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_extract_txt_segments(tmp_path: Path):
    p = tmp_path / "sample.txt"
    p.write_text("hello\nworld", encoding="utf-8")
    source_type, segments = extract_file_segments(p)
    assert source_type == "txt"
    assert len(segments) == 1
    assert "hello" in segments[0]["text"]


def test_fetch_url_text(monkeypatch):
    pytest.importorskip("bs4")
    class DummyResponse:
        text = "<html><head><title>Example</title></head><body><h1>Hello</h1><script>x=1</script><p>World</p></body></html>"

        def raise_for_status(self):
            return None

    def fake_get(url, timeout, headers):
        assert url == "https://example.com"
        return DummyResponse()

    monkeypatch.setattr("backend.modules.ingestion.requests", SimpleNamespace(get=fake_get))
    title, text = fetch_url_text("https://example.com")
    assert title == "Example"
    assert "Hello" in text
    assert "World" in text
    assert "x=1" not in text

# ADDED tests for empty documents, upon empty document return error
def test_format_extracted_text_for_pdf_or_pptx_includes_locations():
    result = format_extracted_text(
        [
            {"location": "page 1", "text": "Intro\n\n\ntext"},
            {"location": "page 2", "text": "Details"},
        ],
        "pdf",
    )
    assert "[page 1]" in result
    assert "[page 2]" in result
    assert "\n\n\n" not in result


def test_format_extracted_text_for_txt_keeps_plain_text():
    result = format_extracted_text(
        [{"location": "full text", "text": "Hello\r\nWorld\r\n\r\n\r\nDone"}],
        "txt",
    )
    assert result == "Hello\nWorld\n\nDone"


def test_ingest_uploaded_bytes_rejects_empty_text(tmp_path: Path):
    store = NotebookStore(base_dir=str(tmp_path))
    nb = store.create(NotebookCreate(user_id="u1", name="N1"))

    with pytest.raises(ValueError, match="No extractable text found"):
        ingest_uploaded_bytes(
            store,
            user_id="u1",
            notebook_id=nb.notebook_id,
            filename="empty.txt",
            content=b"",
        )
