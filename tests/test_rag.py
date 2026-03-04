from backend.models.schemas import NotebookCreate
from backend.modules import rag
from backend.services.storage import NotebookStore


def test_build_rag_prompt_includes_citation_format():
    prompt = rag.build_rag_prompt(
        "What is this?",
        [
            {
                "chunk_id": "c1",
                "text": "Chunk body text",
                "metadata": {
                    "source_name": "sample.pdf",
                    "source_type": "pdf",
                    "location": "page 2",
                },
            }
        ],
    )
    assert "Answer using ONLY the provided sources" in prompt
    assert "[Source: sample.pdf | Type: pdf | Location: page 2]" in prompt
    assert "QUESTION:\nWhat is this?" in prompt


def test_answer_notebook_question_persists_messages(monkeypatch, tmp_path):
    store = NotebookStore(base_dir=str(tmp_path))
    nb = store.create(NotebookCreate(user_id="u1", name="Test"))

    monkeypatch.setattr(
        rag,
        "retrieve_notebook_chunks",
        lambda *args, **kwargs: [
            {
                "chunk_id": "chunk-1",
                "text": "Fact text",
                "metadata": {
                    "source_name": "a.txt",
                    "source_type": "txt",
                    "location": "full text",
                },
            }
        ],
    )
    monkeypatch.setattr(rag.llm_service, "generate", lambda prompt: "Answer [Source: a.txt | Type: txt | Location: full text]")

    result = rag.answer_notebook_question(
        store,
        user_id="u1",
        notebook_id=nb.notebook_id,
        message="Question?",
        top_k=3,
    )

    assert result["used_chunks"] == 1
    assert result["citations"][0]["source_name"] == "a.txt"

    messages = store.read_chat_messages("u1", nb.notebook_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_rerank_chunks_prefers_lexically_relevant_chunk():
    candidates = [
        {
            "chunk_id": "a",
            "text": "general overview and intro",
            "metadata": {},
            "distance": 0.15,
        },
        {
            "chunk_id": "b",
            "text": "transformer attention heads and query key value details",
            "metadata": {},
            "distance": 0.20,
        },
    ]
    reranked = rag.rerank_chunks(
        query="explain transformer attention",
        candidate_chunks=candidates,
        top_k=1,
    )
    assert len(reranked) == 1
    assert reranked[0]["chunk_id"] == "b"
