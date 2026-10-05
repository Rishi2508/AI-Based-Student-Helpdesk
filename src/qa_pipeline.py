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
1. Grounding: Answer based STRICTLY and ONLY on the provided Official College Document Context below. Do NOT extrapolate, speculate, or invent policies not stated in the context.
2. Factuality & Conciseness: Keep responses concise, practical, and factual. Always cite official Document IDs (e.g., AITS-ADM-POL-014), required identification documents, deadlines, fees, and office counters/locations where applicable.
3. Information Absence: If the provided context does not contain enough information to answer the question, politely and clearly state that the information is not available in the official college documents, and advise the student to contact the relevant administrative office.

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
        """Constructs the runnable LangChain QA chain."""
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", STUDENT_HELPDESK_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )

        if self.api_key and self.api_key != "your_openai_api_key_here":
            try:
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    openai_api_key=self.api_key,
                )
                return self.prompt | self.llm | StrOutputParser()
            except Exception:
                self.llm = None
                return None

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
        # Retrieve context from indexer
        if hasattr(self.indexer, "retrieve_nodes"):
            nodes = self.indexer.retrieve_nodes(query, top_k=3)
            context = self.format_context(nodes) if nodes else self.indexer.retrieve_context(query, top_k=3)
            sources = [
                node.metadata.get("file_name", "Unknown")
                for node in nodes
                if hasattr(node, "metadata")
            ]
        else:
            context = self.indexer.retrieve_context(query, top_k=3)
            sources = []

        if self.chain is not None:
            raw_response = self.chain.invoke({"context": context, "question": query})
            answer = str(raw_response)
        else:
            # Fallback when running without an active OpenAI API key
            answer = (
                "[MOCK MODE - OPENAI_API_KEY not configured]\n"
                f"Retrieved college policy context for: '{query}'\n\n"
                f"Context Preview:\n{context[:450]}..."
            )

        return {
            "query": query,
            "answer": answer,
            "context": context,
            "sources": list(dict.fromkeys(sources)),
        }

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
