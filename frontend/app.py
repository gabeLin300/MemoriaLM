import json
import os
import shutil
import time

import gradio as gr

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def _user_root(user_id):
    return os.path.join(DATA_ROOT, user_id)


def _index_path(user_id):
    return os.path.join(_user_root(user_id), "notebook_index.json")


def _load_index(user_id):
    index_path = _index_path(user_id)
    if not os.path.exists(index_path):
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "notebooks" in data:
        notebooks = data["notebooks"]
        if isinstance(notebooks, list):
            ids = []
            for item in notebooks:
                if isinstance(item, dict) and "id" in item:
                    ids.append(str(item["id"]))
                else:
                    ids.append(str(item))
            return ids
    return []


def _save_index(user_id, notebook_ids):
    os.makedirs(_user_root(user_id), exist_ok=True)
    index_path = _index_path(user_id)
    payload = {"notebooks": [{"id": nid} for nid in notebook_ids]}
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _notebook_paths(user_id, notebook_id):
    base = os.path.join(_user_root(user_id), notebook_id)
    return {
        "base": base,
        "sources": os.path.join(base, "sources"),
        "vector_store": os.path.join(base, "vector_store"),
        "artifacts": os.path.join(base, "artifacts"),
        "chats": os.path.join(base, "chats.json"),
    }


def _ensure_notebook_structure(user_id, notebook_id):
    paths = _notebook_paths(user_id, notebook_id)
    os.makedirs(paths["sources"], exist_ok=True)
    os.makedirs(paths["vector_store"], exist_ok=True)
    os.makedirs(paths["artifacts"], exist_ok=True)
    if not os.path.exists(paths["chats"]):
        with open(paths["chats"], "w", encoding="utf-8") as f:
            json.dump([], f)


def _load_chat_history(user_id, notebook_id):
    if not user_id or not notebook_id:
        return []
    paths = _notebook_paths(user_id, notebook_id)
    if not os.path.exists(paths["chats"]):
        return []
    try:
        with open(paths["chats"], "r", encoding="utf-8") as f:
            messages = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    history = []
    for item in messages:
        if isinstance(item, dict):
            # Convert old format if needed
            if "role" in item and "content" in item:
                history.append(item)
            elif "user" in item and "assistant" in item:
                if item["user"]:
                    history.append({"role": "user", "content": item["user"]})
                if item["assistant"]:
                    history.append({"role": "assistant", "content": item["assistant"]})
    return history


def _save_chat_history(user_id, notebook_id, history):
    paths = _notebook_paths(user_id, notebook_id)
    # Save as list of dicts with 'role' and 'content'
    with open(paths["chats"], "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_notebooks(user_id):
    if not user_id:
        return gr.Dropdown(choices=[], value=None)
    notebook_ids = _load_index(user_id)
    value = notebook_ids[0] if notebook_ids else None
    return gr.Dropdown(choices=notebook_ids, value=value)


def create_notebook(user_id, new_id):
    if not user_id:
        return gr.Dropdown(choices=[], value=None), ""
    notebook_id = (new_id or "").strip()
    if not notebook_id:
        notebook_id = f"notebook_{int(time.time())}"
    notebook_ids = _load_index(user_id)
    if notebook_id not in notebook_ids:
        notebook_ids.append(notebook_id)
    _save_index(user_id, notebook_ids)
    _ensure_notebook_structure(user_id, notebook_id)
    return gr.Dropdown(choices=notebook_ids, value=notebook_id), ""


def remove_notebook(user_id, notebook_id):
    if not user_id or not notebook_id:
        return gr.Dropdown(choices=[], value=None), []
    notebook_ids = _load_index(user_id)
    if notebook_id in notebook_ids:
        notebook_ids.remove(notebook_id)
    _save_index(user_id, notebook_ids)
    paths = _notebook_paths(user_id, notebook_id)
    if os.path.exists(paths["base"]):
        shutil.rmtree(paths["base"])
    value = notebook_ids[0] if notebook_ids else None
    history = _load_chat_history(user_id, value) if value else []
    return gr.Dropdown(choices=notebook_ids, value=value), history


def switch_notebook(user_id, notebook_id):
    history = _load_chat_history(user_id, notebook_id)
    return history


def respond(message, history, user_id, notebook_id):
    if not message or not user_id or not notebook_id:
        return "", history
    # Ensure history is a list of dicts with 'role' and 'content'
    if not history or not isinstance(history, list):
        history = []
    # Add user message
    history = history + [{"role": "user", "content": message}]
    # Add assistant response (stub)
    assistant_response = "Stub response. Integrate backend here."
    history = history + [{"role": "assistant", "content": assistant_response}]
    _save_chat_history(user_id, notebook_id, history)
    return "", history


with gr.Blocks(title="RAG NotebookLM Clone") as demo:
    with gr.Row():
        gr.Markdown("**User Login**")
        # gr.LoginButton()
        user_id = gr.Textbox(label="User ID", placeholder="e.g. user_123")

    gr.Markdown("# RAG NotebookLM Clone")

    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("**Notebook Management**")
            new_id = gr.Textbox(label="New notebook ID")
            new_notebook = gr.Button("New notebook")
            notebook_selector = gr.Dropdown(label="Notebooks", choices=[], value=None)
            delete_notebook = gr.Button("Delete notebook")

        with gr.Column(scale=2, min_width=400):
            gr.Markdown("**Chat**")
            chatbot = gr.Chatbot(height=420)
            with gr.Row():
                message = gr.Textbox(
                    placeholder="Type your message...",
                    show_label=False,
                    container=True,
                )
                send = gr.Button("Send")

            send.click(
                respond,
                inputs=[message, chatbot, user_id, notebook_selector],
                outputs=[message, chatbot],
            )
            message.submit(
                respond,
                inputs=[message, chatbot, user_id, notebook_selector],
                outputs=[message, chatbot],
            )

    user_id.change(load_notebooks, inputs=[user_id], outputs=[notebook_selector])
    new_notebook.click(
        create_notebook,
        inputs=[user_id, new_id],
        outputs=[notebook_selector, new_id],
    )
    delete_notebook.click(
        remove_notebook,
        inputs=[user_id, notebook_selector],
        outputs=[notebook_selector, chatbot],
    )
    notebook_selector.change(
        switch_notebook,
        inputs=[user_id, notebook_selector],
        outputs=[chatbot],
    )


demo.launch(server_name="0.0.0.0", server_port=7860)