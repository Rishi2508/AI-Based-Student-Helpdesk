"""Document indexing and retrieval module using LlamaIndex.

Responsible for ingesting college policy documents (Markdown, PDF, Text),
generating embeddings, persisting vector indices to storage, and providing
retrieval interfaces for student queries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore


class CollegeDocumentIndexer:
    """Manages document ingestion, vector index construction, persistence, and retrieval."""

    def __init__(
        self,
        docs_dir: str | Path = "./data/documents",
        storage_dir: str | Path = "./storage",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.storage_dir = Path(storage_dir)
        self.llm_model = os.getenv("OPENAI_MODEL_NAME", llm_model)
        self.embedding_model = os.getenv("EMBEDDING_MODEL_NAME", embedding_model)

        self._configure_settings()
        self.index: Optional[VectorStoreIndex] = None

    def _configure_settings(self) -> None:
        """Configures global LlamaIndex LLM and Embedding settings with offline fallbacks."""
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "your_openai_api_key_here":
            try:
                from llama_index.llms.openai import OpenAI
                from llama_index.embeddings.openai import OpenAIEmbedding

                Settings.llm = OpenAI(model=self.llm_model, temperature=0.1)
                Settings.embed_model = OpenAIEmbedding(model_name=self.embedding_model)
                return
            except Exception:
                pass

        # Offline fallback for testing and development without live API keys
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding
        from llama_index.core.llms.mock import MockLLM

        Settings.llm = MockLLM()
        Settings.embed_model = MockEmbedding(embed_dim=16)

    def load_documents(self) -> List[Document]:
        """Loads all Markdown, PDF, and text documents from the documents directory."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory '{self.docs_dir}' does not exist.")

        reader = SimpleDirectoryReader(
            input_dir=str(self.docs_dir),
            required_exts=[".pdf", ".md", ".txt"],
            recursive=True,
        )
        documents = reader.load_data()
        return documents

    def build_or_load_index(self, force_rebuild: bool = False) -> VectorStoreIndex:
        """Handles indexing and persistence.

        Checks if an existing index exists in the storage directory; if found,
        reloads it via load_index_from_storage(StorageContext.from_defaults(persist_dir=...)).
        If no existing index is found, builds a VectorStoreIndex and persists it to storage.
        """
        docstore_path = self.storage_dir / "docstore.json"
        if not force_rebuild and docstore_path.exists():
            storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
            self.index = load_index_from_storage(storage_context)
            return self.index

        documents = self.load_documents()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True,
        )
        self.index.storage_context.persist(persist_dir=str(self.storage_dir))
        return self.index

    def get_retriever(self, similarity_top_k: int = 3) -> VectorIndexRetriever:
        """Returns the raw LlamaIndex retriever object against the vector store."""
        if self.index is None:
            self.build_or_load_index()
        assert self.index is not None
        return self.index.as_retriever(similarity_top_k=similarity_top_k)

    def retrieve_nodes(self, query: str, top_k: int = 3) -> List[NodeWithScore]:
        """Retrieves raw scored context nodes for a given query, augmented with keyword relevance."""
        retriever = self.get_retriever(similarity_top_k=top_k)
        retrieved_nodes = retriever.retrieve(query)

        # In offline/mock mode or as hybrid re-ranking, boost nodes matching query terms
        api_key = os.getenv("OPENAI_API_KEY")
        is_mock_mode = not api_key or api_key == "your_openai_api_key_here"

        if is_mock_mode and hasattr(self.index, "docstore") and self.index.docstore.docs:
            stop_words = {
                "what", "is", "the", "for", "if", "a", "an", "can", "how", "are",
                "of", "in", "to", "be", "from", "by", "many", "do", "does", "on"
            }
            keywords = [
                w.strip("?,.:;\"'()")
                for w in query.lower().split()
                if w.strip("?,.:;\"'()") and w.strip("?,.:;\"'()") not in stop_words
            ]
            if keywords:
                scored_nodes = []
                for doc_id, doc in self.index.docstore.docs.items():
                    text = doc.get_content().lower()
                    fn = doc.metadata.get("file_name", "").lower()
                    kw_score = sum(text.count(kw) + (15 if kw in fn else 0) for kw in keywords)
                    if kw_score > 0:
                        scored_nodes.append((kw_score, NodeWithScore(node=doc, score=float(kw_score))))

                if scored_nodes:
                    scored_nodes.sort(key=lambda x: x[0], reverse=True)
                    seen_texts = set()
                    final_nodes = []
                    for _, node in scored_nodes:
                        txt = node.node.get_content()
                        if txt not in seen_texts:
                            seen_texts.add(txt)
                            final_nodes.append(node)
                        if len(final_nodes) >= top_k:
                            break
                    if final_nodes:
                        return final_nodes

        return retrieved_nodes

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Runs a retriever against the vector store and returns the relevant context as a cleanly concatenated text string."""
        nodes = self.retrieve_nodes(query, top_k=top_k)
        if not nodes:
            return "No relevant college policy context found."

        chunks = []
        for i, node in enumerate(nodes, start=1):
            file_name = getattr(node, "metadata", {}).get("file_name", "College Policy Document")
            content = node.node.get_content() if hasattr(node, "node") else getattr(node, "text", str(node))
            chunks.append(f"=== [Document Chunk {i} | Source: {file_name}] ===\n{content.strip()}")

        return "\n\n".join(chunks)


# Alias for backward compatibility
DocumentIndexer = CollegeDocumentIndexer


def get_default_indexer() -> CollegeDocumentIndexer:
    """Convenience factory function for the default CollegeDocumentIndexer."""
    docs_dir = os.getenv("DOCUMENTS_DIR", "./data/documents")
    storage_dir = os.getenv("STORAGE_DIR", "./storage")
    return CollegeDocumentIndexer(docs_dir=docs_dir, storage_dir=storage_dir)


if __name__ == "__main__":
    test_query = "What is the procedure for applying for a bonafide certificate?"
    print(f"=== Verifying Document Indexing & Retrieval ===")
    indexer = get_default_indexer()

    print(f"\n1. Ingesting documents from: {indexer.docs_dir.resolve()}")
    docs = indexer.load_documents()
    print(f"   Loaded {len(docs)} document chunk(s).")

    print(f"\n2. Building or loading vector index from: {indexer.storage_dir.resolve()}")
    indexer.build_or_load_index()
    print("   Vector index ready and persisted.")

    print(f"\n3. Running test retrieval for query:\n   \"{test_query}\"\n")
    retrieved_chunk = indexer.retrieve_context(test_query, top_k=3)

    print("--- Retrieved Context Chunk(s) ---")
    print(retrieved_chunk)
    print("----------------------------------")
