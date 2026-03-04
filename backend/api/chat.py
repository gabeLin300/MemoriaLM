from fastapi import APIRouter, Depends, HTTPException

from backend.models.schemas import ChatHistoryOut, ChatMessageOut, ChatRequest, ChatResponseOut
from backend.modules.rag import answer_notebook_question, get_chat_history
from backend.services.auth import User, enforce_user_match, get_current_user
from backend.services.storage import NotebookStore

router = APIRouter()
store = NotebookStore(base_dir="data")


@router.get("/{notebook_id}/chat", response_model=ChatHistoryOut)
def chat_history(notebook_id: str, current_user: User = Depends(get_current_user)) -> ChatHistoryOut:
    try:
        messages = get_chat_history(store, user_id=current_user.user_id, notebook_id=notebook_id)
        return ChatHistoryOut(messages=[ChatMessageOut(**m) for m in messages])
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{notebook_id}/chat", response_model=ChatResponseOut)
def chat(
    notebook_id: str,
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
) -> ChatResponseOut:
    try:
        enforce_user_match(current_user, payload.user_id)
        result = answer_notebook_question(
            store,
            user_id=current_user.user_id,
            notebook_id=notebook_id,
            message=payload.message,
            top_k=payload.top_k,
            retrieval_mode=payload.retrieval_mode,
        )
        return ChatResponseOut(**result)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Notebook not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc
