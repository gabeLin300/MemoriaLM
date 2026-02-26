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
