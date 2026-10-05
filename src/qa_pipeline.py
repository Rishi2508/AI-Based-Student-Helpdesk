"""LangChain-powered QA and Answer Synthesis Pipeline.

Combines retrieved context from LlamaIndex with prompt templates to generate
grounded, comprehensive, and citation-rich student helpdesk answers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.indexer import DocumentIndexer, get_default_indexer

STUDENT_HELPDESK_SYSTEM_PROMPT = """You are the official AI Academic Advisor and Student Helpdesk Assistant for Apex Institute of Technology & Sciences.

Your mission is to provide accurate, clear, and reassuring answers to students regarding university policies, procedures, regulations, deadlines, and services.

Follow these strict guidelines:
1. Grounding: Answer ONLY based on the provided Official College Policy Context. Do NOT invent policies or assume rules not mentioned.
2. Structure your response clearly:
   - **Direct Answer**: Provide a concise summary directly answering the student's question.
   - **Key Regulations & Requirements**: Bullet points highlighting thresholds, documents needed, fees, or deadlines.
   - **Action Steps**: Step-by-step guidance on how the student should proceed (e.g., ERP portal paths, counter numbers, office locations).
   - **Official Reference & Contact**: Quote the Document ID (e.g., AITS-ADM-POL-014) and relevant contact email/office.
3. If the provided context does not contain enough information to answer completely, acknowledge what is known and advise the student to contact the relevant department directly.
4. Maintain an encouraging, professional, and empathetic tone.

Official College Policy Context:
---------------------
{context}
---------------------
"""


class StudentHelpdeskQAPipeline:
    """Orchestrates query answering using LangChain LLM chains and LlamaIndex context."""

    def __init__(
        self,
        indexer: Optional[DocumentIndexer] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> None:
        self.indexer = indexer or get_default_indexer()
        self.model_name = os.getenv("OPENAI_MODEL_NAME", model_name)
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY", "")

        self._init_chain()

    def _init_chain(self) -> None:
        """Initializes the LangChain prompt template and runnable chain."""
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", STUDENT_HELPDESK_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        if self.api_key and self.api_key != "your_openai_api_key_here":
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key,
            )
            self.chain = self.prompt | self.llm | StrOutputParser()
        else:
            self.llm = None
            self.chain = None

    def format_context(self, nodes: List[Any]) -> str:
        """Formats retrieved context nodes into a consolidated text block with source tags."""
        context_blocks = []
        for i, node in enumerate(nodes, start=1):
            file_name = getattr(node, "metadata", {}).get("file_name", "Official College Policy")
            text = getattr(node, "text", str(node))
            context_blocks.append(f"--- [Source {i}: {file_name}] ---\n{text.strip()}")
        return "\n\n".join(context_blocks)

    def answer_query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """Retrieves relevant context and generates a synthesized answer."""
        if hasattr(self.indexer, "retrieve_nodes"):
            retrieved_nodes = self.indexer.retrieve_nodes(question, top_k=top_k)
            formatted_context = self.format_context(retrieved_nodes)
            sources = [
                node.metadata.get("file_name", "Unknown")
                for node in retrieved_nodes
                if hasattr(node, "metadata")
            ]
            node_count = len(retrieved_nodes)
        else:
            formatted_context = self.indexer.retrieve_context(question, top_k=top_k)
            sources = []
            node_count = 1 if formatted_context else 0

        if self.chain is not None:
            raw_response = self.chain.invoke(
                {"context": formatted_context, "question": question}
            )
            response_text = str(raw_response)
        else:
            # Fallback when running without an active OpenAI API key
            response_text = (
                "[MOCK MODE - OPENAI_API_KEY not configured]\n"
                f"Retrieved {node_count} policy document section(s).\n\n"
                f"Top context preview:\n{formatted_context[:450]}..."
            )

        return {
            "question": question,
            "answer": response_text,
            "sources": list(dict.fromkeys(sources)),
            "retrieved_nodes_count": node_count,
            "context": formatted_context,
        }


def get_default_pipeline() -> StudentHelpdeskQAPipeline:
    """Convenience factory function."""
    return StudentHelpdeskQAPipeline()


if __name__ == "__main__":
    pipeline = get_default_pipeline()
    sample_q = "How do I apply for a bonafide certificate and what ID is needed?"
    print(f"Testing QA Pipeline with question: '{sample_q}'")
    result = pipeline.answer_query(sample_q)
    print(f"Sources: {result['sources']}")
    print(f"Answer:\n{result['answer']}")
