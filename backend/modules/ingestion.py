from pathlib import Path
from typing import Iterable, List


ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".txt"}


def validate_file(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += chunk_size - overlap
    return chunks


def embed_chunks(chunks: Iterable[str]) -> List[list]:
    # Placeholder for Chroma embeddings
    return [[0.0] * 1536 for _ in chunks]
