"""Main entry point for the AI-Based Student Helpdesk application.

Provides interactive and CLI execution modes combining AutoGen,
LlamaIndex, and LangChain for answering student inquiries.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path when script is executed directly
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

    doc_files = sorted(list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.pdf")))
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


def run_single_query(query: str, use_agents: bool = True, show_context: bool = False) -> None:
    """Processes a single student query and prints the response."""
    print(f"\n❓ Student Inquiry: \"{query}\"\n")

    if use_agents:
        print("🤖 Invoking AutoGen Multi-Agent Coordinator (StudentProxy -> HelpdeskCoordinator -> RetrievalQASpecialist)...")
        coordinator = get_default_coordinator()
        result = coordinator.process_query(query)
        answer = result["answer"]
        context = result.get("context", "")
        flow = " -> ".join(result.get("agent_flow", ["StudentProxy", "HelpdeskCoordinator", "RetrievalQASpecialist"]))
        category = result.get("category", "general")

        print(f"📌 Domain Category: [{category.upper()}]")
        print(f"🔄 Agent Workflow : {flow}\n")
    else:
        print("🔍 Searching Knowledge Base via LlamaIndex & LangChain QA Pipeline...")
        pipeline = get_default_pipeline()
        result = pipeline.answer_question(query)
        answer = result["answer"]
        context = result.get("context", "")

    print("--- Official Advisor Response ---")
    print(answer)

    if show_context and context:
        print("\n--- Retrieved Grounding Context Excerpts ---")
        preview = context.strip()
        if len(preview) > 800:
            print(f"{preview[:800]}...\n[Truncated {len(preview) - 800} additional characters]")
        else:
            print(preview)

    print()


def run_interactive_mode(use_agents: bool = True, show_context: bool = False) -> None:
    """Starts an interactive session for the student."""
    print_banner()
    print("Welcome to the Apex Institute AI Student Helpdesk!")
    print("Ask any question regarding college regulations, fee refunds, attendance,")
    print("bonafide certificates, exams, course registration, or transfer procedures.")
    print("Type 'exit' or 'quit' to end the session.\n")

    coordinator = get_default_coordinator() if use_agents else None
    pipeline = get_default_pipeline() if not use_agents else None

    while True:
        try:
            user_input = input("Student > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nThank you for using the Student Helpdesk. Have a productive semester! 🎓\n")
                break

            print("\nThinking and consulting university policy documents...")

            if use_agents and coordinator is not None:
                result = coordinator.process_query(user_input)
                answer = result["answer"]
                context = result.get("context", "")
                cat = result.get("category", "general")
                print(f"[Category: {cat.upper()}]")
            else:
                assert pipeline is not None
                result = pipeline.answer_question(user_input)
                answer = result["answer"]
                context = result.get("context", "")

            print("\n[Official Advisor Response]")
            print(answer)

            if show_context and context:
                print("\n[Retrieved Context Snippets]")
                print(context[:500].strip() + ("..." if len(context) > 500 else ""))

            print("-" * 72 + "\n")

        except KeyboardInterrupt:
            print("\n\nSession terminated by student. Goodbye!")
            break


def launch_web_ui(show_context: bool = False) -> None:
    """Launches a lightweight Gradio web interface if available, or falls back gracefully to CLI."""
    try:
        import gradio as gr

        print("\n🚀 Launching Gradio Web Interface on http://localhost:7860 ...")
        coordinator = get_default_coordinator()

        def respond(message: str, chat_history: list):
            res = coordinator.process_query(message)
            answer_text = res["answer"]
            if show_context and res.get("context"):
                answer_text += f"\n\n---\n**Grounding Context:**\n{res['context'][:600]}..."
            return answer_text

        demo = gr.ChatInterface(
            fn=respond,
            title="🎓 AI-Based Student Helpdesk",
            description="Powered by AutoGen Multi-Agent Coordination, LlamaIndex Vector Retrieval, and LangChain Grounded QA.",
            examples=[
                "What is the procedure for applying for a bonafide certificate?",
                "What documents are required for examination registration?",
                "What is the attendance requirement and medical condonation limit?",
                "What is the fee refund policy if admission is cancelled?",
            ],
        )
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

    except ImportError:
        print("\nℹ️  Web UI package ('gradio') is not installed in the current environment.")
        print("💡 You can install it using: pip install gradio")
        print("➡️  Falling back gracefully to interactive CLI mode...\n")
        run_interactive_mode(use_agents=True, show_context=show_context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-Based Student Helpdesk (AutoGen + LlamaIndex + LangChain)"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Process a one-off question and exit.",
    )
    parser.add_argument(
        "--agents", "-a",
        action="store_true",
        default=True,
        help="Use the full AutoGen multi-agent workflow (default: enabled).",
    )
    parser.add_argument(
        "--no-agents",
        dest="agents",
        action="store_false",
        help="Bypass AutoGen coordinator and query LangChain QA pipeline directly.",
    )
    parser.add_argument(
        "--show-context", "-c",
        action="store_true",
        help="Display retrieved document excerpts alongside the answer.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch a lightweight web interface if available, or fall back to interactive CLI.",
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

    if args.ui:
        launch_web_ui(show_context=args.show_context)
        return

    if args.query:
        run_single_query(args.query, use_agents=args.agents, show_context=args.show_context)
        return

    # Default: Interactive terminal chatbot loop
    run_interactive_mode(use_agents=args.agents, show_context=args.show_context)


if __name__ == "__main__":
    main()
