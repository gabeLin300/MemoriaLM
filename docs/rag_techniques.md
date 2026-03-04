# RAG Techniques Comparison

This project currently supports two retrieval modes in chat:

1. `topk`: baseline dense retrieval from Chroma using cosine distance.
2. `rerank`: retrieves a larger candidate pool, then re-scores candidates using:
   - vector relevance (from Chroma distance)
   - lexical overlap with query terms

## Where It Is Implemented

- Request model: `backend/models/schemas.py` (`ChatRequest.retrieval_mode`)
- Retrieval logic: `backend/modules/rag.py`
- Chat API: `backend/api/chat.py`

## How To Run Benchmark

Prerequisites:
- Backend running
- Notebook has at least one ingested source

Example:

```bash
python scripts/rag_benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --user-id <user_id> \
  --notebook-id <notebook_id> \
  --query "Explain the key ideas in my notes" \
  --top-k 5 \
  --runs 5
```

The script prints JSON with average/min/max latency and citation/chunk stats for both modes.

## Report Template

Use this table in your class deliverable:

| Query | Mode | Avg Latency (ms) | Avg Citations | Notes on Retrieved Context |
|---|---:|---:|---:|---|
| Q1 | topk |  |  |  |
| Q1 | rerank |  |  |  |
| Q2 | topk |  |  |  |
| Q2 | rerank |  |  |  |

Recommended write-up points:
- How different the retrieved chunks were between `topk` and `rerank`
- Which mode produced less redundant context
- Latency tradeoff (rerank usually slightly slower)
