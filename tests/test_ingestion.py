from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.modules.ingestion import chunk_text, extract_file_segments, fetch_url_text


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
