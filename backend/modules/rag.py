from typing import List


def retrieve_top_k(query_embedding: List[float], index: List[List[float]], k: int = 5) -> List[int]:
    # Placeholder retrieval using cosine similarity (not implemented)
    return list(range(min(k, len(index))))


def generate_response(query: str, contexts: List[str]) -> str:
    # Placeholder for LLM response
    return f"Answer based on {len(contexts)} context chunks. Query: {query}"
