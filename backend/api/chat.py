from fastapi import APIRouter, HTTPException, Query

from backend.models.schemas import ChatHistoryOut, ChatMessageOut, ChatRequest, ChatResponseOut
from backend.modules.rag import answer_notebook_question, get_chat_history
from backend.services.storage import NotebookStore

router = APIRouter()
store = NotebookStore(base_dir="data")


@router.get("/{notebook_id}/chat", response_model=ChatHistoryOut)
def chat_history(notebook_id: str, user_id: str = Query(...)) -> ChatHistoryOut:
    try:
        messages = get_chat_history(store, user_id=user_id, notebook_id=notebook_id)
        return ChatHistoryOut(messages=[ChatMessageOut(**m) for m in messages])
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{notebook_id}/chat", response_model=ChatResponseOut)
def chat(notebook_id: str, payload: ChatRequest) -> ChatResponseOut:
    try:
        result = answer_notebook_question(
            store,
            user_id=payload.user_id,
            notebook_id=notebook_id,
            message=payload.message,
            top_k=payload.top_k,
        )
        return ChatResponseOut(**result)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
