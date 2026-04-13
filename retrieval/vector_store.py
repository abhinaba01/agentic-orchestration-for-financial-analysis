from typing import Optional

import chromadb
from chromadb.config import Settings

from config import Config
from preprocessing.chunker import Chunk
from retrieval.embedder import TextEmbedder


class VectorStore:
    """Vector store wrapper for ChromaDB persistent collections."""

    def __init__(self, config: Config, embedder: TextEmbedder) -> None:
        self.client = chromadb.PersistentClient(
    path=config.CHROMA_PERSIST_DIR,
)
       
        self.collection = self.client.get_or_create_collection(name="financial_documents")
        self.embedder = embedder

    def index_document(self, doc_id: str, chunks: list[Chunk]) -> None:
        """Embed and index chunks into the ChromaDB collection."""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"Indexed {len(chunks)} chunks for doc_id: {doc_id}")

    def query(self, query_text: str, top_k: int = 5, doc_ids: list[str] = None) -> list[dict]:
        """Query the vector store and return top matching chunks."""
        embedding = self.embedder.embed_query(query_text)
        where_filter = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[dict] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for doc_text, metadata, distance in zip(documents, metadatas, distances):
            score = 1.0 - distance
            chunks.append(
                {
                    "chunk_text": doc_text,
                    "doc_id": metadata.get("doc_id", ""),
                    "section": metadata.get("section", ""),
                    "score": float(score),
                }
            )
        return chunks

    def delete_document(self, doc_id: str) -> None:
        """Delete all indexed chunks for a specific document id."""
        self.collection.delete(where={"doc_id": doc_id})
        print(f"Deleted all chunks for doc_id: {doc_id}")

    def get_collection_stats(self) -> dict:
        """Return collection statistics such as total number of chunks."""
        total_chunks = self.collection.count()
        return {"total_chunks": total_chunks, "collection_name": self.collection.name}
