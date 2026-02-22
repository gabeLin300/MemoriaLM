# System Overview

This project is a full-stack Retrieval-Augmented Generation (RAG) application inspired by NotebookLM. The system allows authenticated users to upload documents or URLs, chat with their content using retrieval-based AI, and generate study artifacts such as reports, quizzes, and podcast transcripts with audio.

## Features
- Upload documents (`.pdf`, `.pptx`, `.txt`) and URL links to the chatbot
- Generate artifacts from uploaded content (podcasts with transcripts, quizzes, reports)

## Component/Module Responsibilities
- Front End / UI (Gradio or Streamlit)
  - Handles Hugging Face OAuth via Gradio login
  - Provides UI for document and URL upload
- Notebook Management Module (FastAPI CRUD)
  - Create, Read, Update, Delete notebooks
  - Persists data per-user in local storage
- Document Ingestion Module
  - Preprocessing: sanitize inputs and validate file types
  - Chunking: split into overlapping chunks
  - Vector Embeddings: build Chroma collection
- RAG Module
  - Embed user query
  - Retrieve top-K chunks from vector store
  - Augment prompt and call LLM
  - Return response with citations

## Data Model
```
/data/
  /{user_id}/
    notebook_index.json
    /{notebook_id}/
      sources/
      vector_store/
      chats.json
      artifacts/
        report_1.md
        quiz_1.md
        podcast_1.md
        podcast_1.mp3
```

## End-to-End Flow
### Ingestion
1. User uploads file or URL
2. Backend extracts raw text
3. Text is chunked
4. Chunks are embedded
5. Embeddings stored in Chroma index
6. Index stored inside user notebook directory

### Retrieval / Chat
1. User submits a query
2. Query is embedded
3. Chroma returns top-K chunks
4. Retrieved chunks passed to LLM
5. Augmented response returned to UI

## Security Plan
- Hugging Face OAuth required login to access files
- User credentials create a unique user directory
- Paths verified/normalized to ensure access only within user directory

## Milestones
1. UI with authentication
2. Notebook creation (CRUD API)
3. Document uploads (PDF, TXT, PPTX)
4. Ingestion (chunking + index creation)
5. Basic RAG/Chat (retrieval + citations)
6. Artifact generation
7. CI/CD (GitHub Actions)

## Key Risks and Mitigations
- User data leakage: enforce path normalization + user session validation
- Large file uploads: limit file size to 3MB
- Large embeddings cost: dynamically set top-K based on file size
