from datetime import datetime, timezone
import re
import json
from typing import Any, Dict, List

from backend.services.embeddings import embedding_service
from backend.services.llm import llm_service
from backend.modules.ingestion import enabled_source_ids
from backend.services.storage import NotebookStore
from backend.services.vector_store import ChromaNotebookStore


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def citation_label(metadata: Dict[str, Any]) -> str:
    return (
        f"[Source: {metadata.get('source_name', 'unknown')} | "
        f"Type: {metadata.get('source_type', 'unknown')} | "
        f"Location: {metadata.get('location', 'unknown')}]"
    )


def build_rag_prompt(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    source_blocks: List[str] = []
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {}) or {}
        source_blocks.append(
            "\n".join(
                [
                    citation_label(meta),
                    chunk.get("text", ""),
                ]
            )
        )

    sources_section = "\n\n".join(source_blocks) if source_blocks else "(no sources retrieved)"
    return (
        "You are a study assistant. Answer using ONLY the provided sources.\n"
        "If you use a fact, cite it in brackets exactly like:\n"
        "[Source: <name> | Type: <type> | Location: <location>]\n"
        "If the answer is not in the provided sources, say you don't know.\n\n"
        f"SOURCES:\n{sources_section}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:\n"
    )


def retrieve_notebook_chunks(
    store: NotebookStore,
    *,
    user_id: str,
    notebook_id: str,
    query: str,
    top_k: int = 5,
    retrieval_mode: str = "topk",
) -> List[Dict[str, Any]]:
    query_vecs = embedding_service.embed_texts([query])
    if not query_vecs:
        return []
    enabled_ids = enabled_source_ids(store, user_id=user_id, notebook_id=notebook_id)
    if not enabled_ids:
        return []
    chroma = ChromaNotebookStore(store.chroma_dir(user_id, notebook_id))
    mode = retrieval_mode.strip().lower()
    if mode == "topk":
        rows = chroma.query(query_vecs[0], k=max(top_k * 4, top_k))
        return _filter_enabled_rows(rows, enabled_ids)[: max(1, top_k)]
    if mode == "rerank":
        # Retrieve a wider pool first, then rerank using lexical overlap + vector signal.
        pool_size = max(top_k * 4, top_k)
        pool = _filter_enabled_rows(chroma.query(query_vecs[0], k=pool_size), enabled_ids)
        return rerank_chunks(query=query, candidate_chunks=pool, top_k=top_k)
    raise ValueError("Unsupported retrieval_mode")


def _filter_enabled_rows(rows: List[Dict[str, Any]], enabled_ids: set[str]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        source_id = str((row.get("metadata") or {}).get("source_id", ""))
        if source_id in enabled_ids:
            out.append(row)
    return out


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-zA-Z0-9]+", text.lower()) if tok}


def _vector_relevance_from_distance(distance: Any) -> float:
    if distance is None:
        return 0.0
    try:
        dist = float(distance)
    except (TypeError, ValueError):
        return 0.0
    # Chroma cosine distance: lower is better; map to 0..1 relevance-like value.
    return 1.0 / (1.0 + max(0.0, dist))


def rerank_chunks(
    *,
    query: str,
    candidate_chunks: List[Dict[str, Any]],
    top_k: int,
    alpha: float = 0.65,
) -> List[Dict[str, Any]]:
    if not candidate_chunks:
        return []
    query_tokens = _tokenize(query)
    ranked: List[tuple[float, Dict[str, Any]]] = []
    for chunk in candidate_chunks:
        text = str(chunk.get("text", ""))
        chunk_tokens = _tokenize(text)
        lexical = (len(query_tokens & chunk_tokens) / len(query_tokens)) if query_tokens else 0.0
        vector_rel = _vector_relevance_from_distance(chunk.get("distance"))
        score = (alpha * vector_rel) + ((1.0 - alpha) * lexical)
        enriched = dict(chunk)
        enriched["rerank_score"] = score
        ranked.append((score, enriched))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in ranked[: max(1, top_k)]]


def _citation_objects(retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    citations: List[Dict[str, str]] = []
    seen: set[str] = set()
    for chunk in retrieved_chunks:
        meta = chunk.get("metadata", {}) or {}
        chunk_id = str(chunk.get("chunk_id", ""))
        key = f"{meta.get('source_name')}|{meta.get('location')}|{chunk_id}"
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source_name": str(meta.get("source_name", "unknown")),
                "source_type": str(meta.get("source_type", "unknown")),
                "location": str(meta.get("location", "unknown")),
                "chunk_id": chunk_id,
            }
        )
    return citations


def answer_notebook_question(
    store: NotebookStore,
    *,
    user_id: str,
    notebook_id: str,
    message: str,
    top_k: int = 5,
    retrieval_mode: str = "topk",
) -> Dict[str, Any]:
    # Ensures the notebook exists and belongs to the user before retrieving.
    store.require_notebook_dir(user_id, notebook_id)
    created_at = _now()

    csv_lookup = _csv_row_lookup(store, user_id=user_id, notebook_id=notebook_id, message=message)
    if csv_lookup is not None:
        store.append_chat_message(
            user_id,
            notebook_id,
            {
                "role": "user",
                "content": message,
                "created_at": created_at,
            },
        )
        store.append_chat_message(
            user_id,
            notebook_id,
            {
                "role": "assistant",
                "content": csv_lookup["answer"],
                "created_at": created_at,
                "citations": csv_lookup["citations"],
                "used_chunks": csv_lookup["used_chunks"],
            },
        )
        return csv_lookup

    retrieved_chunks = retrieve_notebook_chunks(
        store,
        user_id=user_id,
        notebook_id=notebook_id,
        query=message,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )
    prompt = build_rag_prompt(message, retrieved_chunks)
    answer = llm_service.generate(prompt)
    citations = _citation_objects(retrieved_chunks)

    store.append_chat_message(
        user_id,
        notebook_id,
        {
            "role": "user",
            "content": message,
            "created_at": created_at,
        },
    )
    store.append_chat_message(
        user_id,
        notebook_id,
        {
            "role": "assistant",
            "content": answer,
            "created_at": created_at,
            "citations": citations,
            "used_chunks": len(retrieved_chunks),
        },
    )

    return {
        "answer": answer,
        "citations": citations,
        "used_chunks": len(retrieved_chunks),
    }


def _csv_row_lookup(
    store: NotebookStore,
    *,
    user_id: str,
    notebook_id: str,
    message: str,
) -> Dict[str, Any] | None:
    match = re.search(r"\brow\s+(\d+)\b", message.lower())
    if not match:
        return None
    requested_row = int(match.group(1))

    extracted_dir = store.files_extracted_dir(user_id, notebook_id)
    citations: List[Dict[str, str]] = []
    row_values: List[str] = []

    for meta_path in sorted(extracted_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if str(meta.get("source_type", "")).lower() != "csv":
            continue
        if not bool(meta.get("enabled", True)):
            continue

        source_id = str(meta.get("source_id", ""))
        source_name = str(meta.get("source_name", "unknown"))
        extracted_path = extracted_dir / f"{source_id}.txt"
        if not extracted_path.exists():
            continue
        text = extracted_path.read_text(encoding="utf-8", errors="ignore")

        row_pattern = re.compile(
            rf"\[row {requested_row}\]\n(?P<value>.*?)(?:\n\n\[row \d+\]\n|\Z)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        row_match = row_pattern.search(text)
        if not row_match:
            continue
        value = row_match.group("value").strip()
        if not value:
            continue
        row_values.append(f"{source_name}: {value}")
        citations.append(
            {
                "source_name": source_name,
                "source_type": "csv",
                "location": f"row {requested_row}",
                "chunk_id": f"{source_id}:row-{requested_row}",
            }
        )

    if not row_values:
        return None

    answer_lines = [f"Row {requested_row} value(s):"] + [f"- {v}" for v in row_values]
    return {
        "answer": "\n".join(answer_lines),
        "citations": citations,
        "used_chunks": len(citations),
    }


def get_chat_history(
    store: NotebookStore,
    *,
    user_id: str,
    notebook_id: str,
) -> List[Dict[str, Any]]:
    store.require_notebook_dir(user_id, notebook_id)
    return store.read_chat_messages(user_id, notebook_id)
