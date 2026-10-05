"""Document indexing and retrieval module using LlamaIndex.

Responsible for ingesting college policy documents, generating embeddings,
persisting vector indices, and providing retrieval capabilities.
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
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


class DocumentIndexer:
    """Manages document ingestion, vector index construction, and storage."""

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
        """Configures global LlamaIndex LLM and Embedding settings."""
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
        """Loads all markdown and text documents from the documents directory."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Documents directory '{self.docs_dir}' does not exist.")

        reader = SimpleDirectoryReader(
            input_dir=str(self.docs_dir),
            required_exts=[".md", ".txt"],
            recursive=True,
        )
        documents = reader.load_data()
        return documents

    def build_or_load_index(self, force_rebuild: bool = False) -> VectorStoreIndex:
        """Loads an existing index from storage or builds a new one from documents."""
        if not force_rebuild and (self.storage_dir / "docstore.json").exists():
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

    def get_retriever(self, similarity_top_k: int = 4) -> VectorIndexRetriever:
        """Returns a vector index retriever for fetching context nodes."""
        if self.index is None:
            self.build_or_load_index()
        assert self.index is not None
        return self.index.as_retriever(similarity_top_k=similarity_top_k)

    def retrieve_context(self, query: str, top_k: int = 4) -> List[NodeWithScore]:
        """Retrieves top scoring nodes matching the input query."""
        retriever = self.get_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query)


def get_default_indexer() -> DocumentIndexer:
    """Convenience factory function for the default DocumentIndexer."""
    docs_dir = os.getenv("DOCUMENTS_DIR", "./data/documents")
    storage_dir = os.getenv("STORAGE_DIR", "./storage")
    return DocumentIndexer(docs_dir=docs_dir, storage_dir=storage_dir)


if __name__ == "__main__":
    indexer = get_default_indexer()
    print(f"Checking documents in: {indexer.docs_dir.resolve()}")
    docs = indexer.load_documents()
    print(f"Loaded {len(docs)} document chunks.")
    for d in docs:
        print(f" - {d.metadata.get('file_name', 'Unknown')}")
