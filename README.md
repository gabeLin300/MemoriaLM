---
title: MEMORIALM
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: frontend/app.py
pinned: false
hf_oauth: true
---
# MEMORIALM

📚 Retrieval-Augmented Generation (RAG) NotebookLM Clone

## Overview
MemoriaLM is a full-stack RAG application inspired by NotebookLM. It allows authenticated users to upload documents or URLs, chat with their content using retrieval-based AI, and generate study artifacts (reports, quizzes, podcasts).

### Main Features
- Upload documents (`.pdf`, `.pptx`, `.txt`) and URLs
- Chat with uploaded content using RAG
- Generate study artifacts (reports, quizzes, podcasts)
- Retrieval modes for comparison: `topk` and `rerank`

### Architecture
- **Frontend**: Gradio UI (frontend/app.py)
- **Backend**: FastAPI (backend/app.py) with modular services for ingestion, RAG, storage, authentication, embeddings, and vector store
- **Vector Store**: ChromaDB for semantic search
- **LLM**: Hugging Face Inference API (configurable)

## Setup Instructions
1. **Clone the repository**
	```sh
	git clone <repo-url>
	cd MemoriaLM
	```
2. **Create and activate a Python virtual environment**
	```sh
	python -m venv .venv
	.venv\Scripts\activate  # Windows
	source .venv/bin/activate  # macOS/Linux
	```
3. **Install dependencies**
	```sh
	pip install -r requirements.txt
	pip install -r requirements-dev.txt  # for testing
	```
4. **Set environment variables** (see below)
5. **Run the backend**
	```sh
	uvicorn backend.app:app --reload
	```
6. **Run the frontend**
	```sh
	python frontend/app.py
	```

## Required Environment Variables
Set these in a `.env.local` file at the project root (auto-loaded):

- `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`: Hugging Face API token for LLM inference
- `HF_INFERENCE_MODEL`: Model name (default: `openai/gpt-oss-20b`)
- `BACKEND_URL`: Backend API URL (default: `http://127.0.0.1:8000`)
- `DISABLE_AUTO_BACKEND`: Set to `1` to prevent frontend from auto-starting backend

## Project Structure
- `backend/` - FastAPI backend, API routes, modules, services
- `frontend/` - Gradio UI
- `docs/` - Documentation
- `tests/` - Pytest test suite

## Data Flow
1. User uploads file/URL → Backend extracts and chunks text
2. Chunks embedded → Stored in ChromaDB
3. User query embedded → Top-K chunks retrieved
4. Chunks + query sent to LLM → Response returned

## Testing
Run all tests:
```sh
pytest tests/
```

## RAG Technique Comparison
See `docs/rag_techniques.md` for:
- implemented retrieval modes (`topk` vs `rerank`)
- benchmark command (`scripts/rag_benchmark.py`)
- results table template for the project deliverable
