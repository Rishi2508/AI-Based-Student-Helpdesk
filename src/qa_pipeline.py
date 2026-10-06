"""LangChain-powered QA and Answer Synthesis Pipeline.

Combines retrieved context from LlamaIndex with grounded prompt templates to
generate accurate, concise, and citation-rich student helpdesk answers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path when script is executed directly
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.indexer import CollegeDocumentIndexer, get_default_indexer


STUDENT_HELPDESK_SYSTEM_PROMPT = """You are the official AI Academic Advisor and Student Helpdesk Assistant for Apex Institute of Technology & Sciences.

Your mission is to provide accurate, factual, and reassuring guidance to students regarding university policies, procedures, regulations, deadlines, and administrative services.

Strict Grounding Guidelines:
1. Grounding: Answer based STRICTLY and ONLY on the provided Official College Document Context below. Do NOT extrapolate or guess.
2. Factuality & Conciseness: Be direct, structured, and concise. Present the essential points, required IDs, forms, deadlines, fees, and office counters in 2 to 4 bullet points or short paragraphs.
3. Information Absence: If the provided context does not contain the answer, politely state that the information is not available in the official college documents, and advise contacting the relevant administrative office.

Official College Document Context:
---------------------
{context}
---------------------
"""


class StudentHelpdeskQA:
    """Interfaces with language models and LlamaIndex context for grounded student helpdesk Q&A."""

    def __init__(
        self,
        indexer: Optional[CollegeDocumentIndexer] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ) -> None:
        self.indexer = indexer or get_default_indexer()
        self.model_name = os.getenv("OPENAI_MODEL_NAME", model_name)
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY", "")

        self.chain = self.build_chain()

    def build_chain(self) -> Any:
        """Constructs the runnable LangChain QA chain using Ollama local models or OpenAI."""
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", STUDENT_HELPDESK_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

        provider = os.getenv("LLM_PROVIDER", "ollama").lower()

        # 1. Ollama local models (Default)
        if provider == "ollama":
            try:
                from langchain_ollama import ChatOllama

                ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                ollama_model = os.getenv("OLLAMA_MODEL_NAME", "llama3")

                self.llm = ChatOllama(
                    model=ollama_model,
                    base_url=ollama_url,
                    temperature=self.temperature,
                    num_predict=220,   # Concise answers for fast CPU inference
                    num_ctx=2048,      # Compact context window for fast CPU generation
                    request_timeout=120.0,
                )
                return self.prompt | self.llm | StrOutputParser()
            except Exception:
                pass

        # 2. OpenAI cloud models (fallback)
        if provider == "openai" and self.api_key and self.api_key != "your_openai_api_key_here":
            try:
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    openai_api_key=self.api_key,
                )
                return self.prompt | self.llm | StrOutputParser()
            except Exception:
                pass

        self.llm = None
        return None

    def format_context(self, nodes: List[Any]) -> str:
        """Formats context nodes into a consolidated text block with source tags."""
        context_blocks = []
        for i, node in enumerate(nodes, start=1):
            file_name = getattr(node, "metadata", {}).get("file_name", "Official College Policy")
            text = node.node.get_content() if hasattr(node, "node") else getattr(node, "text", str(node))
            context_blocks.append(f"--- [Source {i}: {file_name}] ---\n{text.strip()}")
        return "\n\n".join(context_blocks)

    def answer_question(self, query: str) -> Dict[str, Any]:
        """Accepts a student query, retrieves context, and runs the QA chain.

        Returns a dictionary containing:
        - 'query': the original question
        - 'answer': the generated response string
        - 'context': the retrieved document snippets used for the answer
        """
        # Retrieve context from indexer (top_k=2 provides precise policy clauses without context bloat)
        if hasattr(self.indexer, "retrieve_nodes"):
            nodes = self.indexer.retrieve_nodes(query, top_k=2)
            context = self.format_context(nodes) if nodes else self.indexer.retrieve_context(query, top_k=2)
            sources = [
                node.metadata.get("file_name", "Unknown")
                for node in nodes
                if hasattr(node, "metadata")
            ]
        else:
            context = self.indexer.retrieve_context(query, top_k=2)
            sources = []

        if self.chain is not None:
            raw_response = self.chain.invoke({"context": context, "question": query})
            answer = str(raw_response)
        else:
            # Fallback when running without an active LLM provider
            answer = (
                "[OFFLINE MODE - LLM not configured]\n"
                f"Retrieved college policy context for: '{query}'\n\n"
                f"Context Preview:\n{context[:450]}..."
            )

        return {
            "query": query,
            "answer": answer,
            "context": context,
            "sources": list(dict.fromkeys(sources)),
        }

    def stream_question(self, query: str, context: Optional[str] = None):
        """Yields answer chunks in real-time for responsive streaming UI."""
        if context is None:
            if hasattr(self.indexer, "retrieve_nodes"):
                nodes = self.indexer.retrieve_nodes(query, top_k=2)
                context = self.format_context(nodes) if nodes else self.indexer.retrieve_context(query, top_k=2)
            else:
                context = self.indexer.retrieve_context(query, top_k=2)

        if self.chain is not None and hasattr(self, "llm") and self.llm is not None:
            formatted_prompt = self.prompt.format_messages(context=context, question=query)
            try:
                for chunk in self.llm.stream(formatted_prompt):
                    content = getattr(chunk, "content", str(chunk))
                    if content:
                        yield content
                return
            except Exception:
                pass

        # Fallback to direct answer if streaming fails
        res = self.answer_question(query)
        yield res["answer"]

    def answer_query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """Compatibility wrapper for answer_question returning legacy keys."""
        res = self.answer_question(question)
        return {
            "question": question,
            "query": question,
            "answer": res["answer"],
            "context": res["context"],
            "sources": res.get("sources", []),
            "retrieved_nodes_count": len(res.get("sources", [])) or (1 if res["context"] else 0),
        }


# Aliases for backward compatibility
StudentHelpdeskQAPipeline = StudentHelpdeskQA


def get_default_pipeline() -> StudentHelpdeskQA:
    """Convenience factory function."""
    return StudentHelpdeskQA()


if __name__ == "__main__":
    qa = StudentHelpdeskQA()
    test_queries = [
        "What is the procedure for applying for a bonafide certificate?",
        "What is the minimum attendance requirement?",
    ]

    print("=" * 75)
    print("🎓 Verifying StudentHelpdeskQA Pipeline (LangChain + LlamaIndex)")
    print("=" * 75)

    for idx, query in enumerate(test_queries, start=1):
        print(f"\n[Test Query {idx}]: {query}\n")
        response = qa.answer_question(query)

        print("--- Answer ---")
        print(response["answer"])

        print("\n--- Retrieved Context Snippet ---")
        preview = response["context"][:350].strip()
        print(f"{preview}...\n")
        print("-" * 75)
