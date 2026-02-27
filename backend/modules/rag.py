from datetime import datetime, timezone
from typing import Any, Dict, List

from backend.services.embeddings import embedding_service
from backend.services.llm import llm_service
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
) -> List[Dict[str, Any]]:
    query_vecs = embedding_service.embed_texts([query])
    if not query_vecs:
        return []
    chroma = ChromaNotebookStore(store.chroma_dir(user_id, notebook_id))
    return chroma.query(query_vecs[0], k=top_k)


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
) -> Dict[str, Any]:
    # Ensures the notebook exists and belongs to the user before retrieving.
    store.require_notebook_dir(user_id, notebook_id)

    retrieved_chunks = retrieve_notebook_chunks(
        store,
        user_id=user_id,
        notebook_id=notebook_id,
        query=message,
        top_k=top_k,
    )
    prompt = build_rag_prompt(message, retrieved_chunks)
    answer = llm_service.generate(prompt)
    citations = _citation_objects(retrieved_chunks)
    created_at = _now()

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


def get_chat_history(
    store: NotebookStore,
    *,
    user_id: str,
    notebook_id: str,
) -> List[Dict[str, Any]]:
    store.require_notebook_dir(user_id, notebook_id)
    return store.read_chat_messages(user_id, notebook_id)
