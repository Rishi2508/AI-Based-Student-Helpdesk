"""Main entry point for the AI-Based Student Helpdesk application.

Provides interactive and CLI execution modes combining AutoGen,
LlamaIndex, and LangChain for answering student inquiries.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.indexer import get_default_indexer
from src.qa_pipeline import get_default_pipeline
from src.agent_coordinator import get_default_coordinator


def print_banner() -> None:
    banner = """
========================================================================
     🎓 AI-BASED STUDENT HELPDESK - APEX INSTITUTE OF TECHNOLOGY
     Powered by AutoGen (Agents) + LlamaIndex (RAG) + LangChain (QA)
========================================================================
    """
    print(banner)


def list_documents() -> None:
    """Lists all available official policy documents in the knowledge base."""
    docs_dir = Path(os.getenv("DOCUMENTS_DIR", "./data/documents"))
    print(f"\n📂 Knowledge Base Directory: {docs_dir.resolve()}")
    if not docs_dir.exists():
        print("❌ Directory does not exist.")
        return

    doc_files = sorted(list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt")))
    print(f"Found {len(doc_files)} policy documents:\n")
    for idx, doc in enumerate(doc_files, start=1):
        size_kb = doc.stat().st_size / 1024
        print(f"  [{idx}] {doc.name:<35} ({size_kb:.1f} KB)")
    print()


def rebuild_index() -> None:
    """Forces rebuilding of the vector index."""
    print("\n⚙️  Rebuilding vector index from data/documents/...")
    indexer = get_default_indexer()
    indexer.build_or_load_index(force_rebuild=True)
    print("✅ Index successfully rebuilt and persisted to storage directory.\n")


def run_single_query(query: str, use_agents: bool = False) -> None:
    """Processes a single student query and prints the response."""
    print(f"\n❓ Student Inquiry: \"{query}\"\n")
    if use_agents:
        print("🤖 Invoking AutoGen Multi-Agent Coordinator...")
        coordinator = get_default_coordinator()
        result = coordinator.process_query(query)
        print("\n--- Response ---")
        print(result["summary"])
    else:
        print("🔍 Searching Knowledge Base via LlamaIndex & LangChain...")
        pipeline = get_default_pipeline()
        result = pipeline.answer_query(query)
        print("\n--- Official Answer ---")
        print(result["answer"])
        print("\n📚 Referenced Documents:")
        for src in result["sources"]:
            print(f"  • {src}")
    print()


def run_interactive_mode() -> None:
    """Starts an interactive session for the student."""
    print_banner()
    print("Welcome! Ask any question about college regulations, fees, attendance,")
    print("bonafide certificates, exams, course registration, or transfer procedures.")
    print("Type 'exit' or 'quit' to end the session.\n")

    pipeline = get_default_pipeline()

    while True:
        try:
            user_input = input("Student > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nThank you for using the Student Helpdesk. Have a great academic semester! 🎓\n")
                break

            print("\nThinking and checking college policies...")
            result = pipeline.answer_query(user_input)
            print("\n[Official Advisor Response]")
            print(result["answer"])
            if result.get("sources"):
                print("\n[References]")
                print(", ".join(result["sources"]))
            print("-" * 72 + "\n")

        except KeyboardInterrupt:
            print("\n\nSession terminated by student. Goodbye!")
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-Based Student Helpdesk (AutoGen + LlamaIndex + LangChain)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single student query non-interactively.",
    )
    parser.add_argument(
        "--agents", "-a",
        action="store_true",
        help="Use AutoGen multi-agent coordinator for query resolution.",
    )
    parser.add_argument(
        "--list-docs",
        action="store_true",
        help="List all policy documents in the knowledge base.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector index from data/documents/.",
    )

    args = parser.parse_args()

    if args.list_docs:
        list_documents()
        return

    if args.rebuild:
        rebuild_index()
        return

    if args.query:
        run_single_query(args.query, use_agents=args.agents)
        return

    # Default to interactive mode
    run_interactive_mode()


if __name__ == "__main__":
    main()
