from pathlib import Path
from typing import Dict, List


class ChromaNotebookStore:
    def __init__(self, persist_dir: Path, collection_name: str = "chunks") -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            if self._client is None:
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def upsert_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")

        collection = self._get_collection()
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = dict(chunk["metadata"])
            metadata["chunk_index"] = int(metadata.get("chunk_index", 0))
            metadatas.append(metadata)

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def count(self) -> int:
        return self._get_collection().count()

    def query(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        if not query_embedding:
            return []
        collection = self._get_collection()
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, k),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        rows: List[Dict] = []
        for i, chunk_id in enumerate(ids):
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) and metas[i] else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return rows
