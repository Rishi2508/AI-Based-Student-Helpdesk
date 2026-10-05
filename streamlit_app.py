"""Streamlit Web Interface for the AI-Based Student Helpdesk.

Combines AutoGen Multi-Agent Coordination, LlamaIndex Vector Retrieval,
and LangChain Grounded QA in an interactive, responsive local web UI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.indexer import get_default_indexer, CollegeDocumentIndexer
from src.qa_pipeline import get_default_pipeline, StudentHelpdeskQA
from src.agent_coordinator import get_default_coordinator, StudentHelpdeskCoordinator

# Set page configuration
st.set_page_config(
    page_title="AI Student Helpdesk | Apex Institute",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced aesthetics
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .badge-category {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .badge-flow {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .status-card {
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing LlamaIndex & Document Indexer...")
def load_indexer() -> CollegeDocumentIndexer:
    """Loads and caches the document indexer."""
    indexer = get_default_indexer()
    indexer.build_or_load_index()
    return indexer


@st.cache_resource(show_spinner="Initializing LangChain QA Pipeline...")
def load_qa_pipeline(_indexer: CollegeDocumentIndexer) -> StudentHelpdeskQA:
    """Loads and caches the QA pipeline."""
    return StudentHelpdeskQA(indexer=_indexer)


@st.cache_resource(show_spinner="Initializing AutoGen Multi-Agent Coordinator...")
def load_coordinator(_pipeline: StudentHelpdeskQA) -> StudentHelpdeskCoordinator:
    """Loads and caches the multi-agent coordinator."""
    return StudentHelpdeskCoordinator(qa_pipeline=_pipeline)


# Initialize components
indexer = load_indexer()
pipeline = load_qa_pipeline(indexer)
coordinator = load_coordinator(pipeline)

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am your **AI Academic Advisor & Student Helpdesk Assistant** at "
                "Apex Institute of Technology & Sciences.\n\n"
                "I can assist you with official policies on:\n"
                "- 📜 **Bonafide & Transfer Certificates** (forms, IDs, No-Dues clearance)\n"
                "- 📝 **Examination Regulations & Hall Tickets** (eligibility, dates, fees)\n"
                "- ⏱️ **Attendance Rules & Medical Condonation** (75% rule, CMO leave)\n"
                "- 💳 **Fee Structure, Online Payment & Refunds** (slabs, NEFT, receipts)\n"
                "- 📚 **Library Timings, Borrowing & Book Renewals** (fines, hours)\n"
                "- 📅 **Course Registration & Credit Rules** (Add/Drop windows, limits)\n\n"
                "How can I help you today?"
            ),
            "category": "welcome",
            "flow": None,
            "context": None,
        }
    ]

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/graduation-cap.png",
        width=70,
    )
    st.title("🎓 Helpdesk Control")

    st.subheader("⚙️ Engine Configuration")
    engine_mode = st.radio(
        "Workflow Engine",
        options=["AutoGen Multi-Agent", "Direct LangChain QA"],
        index=0,
        help="AutoGen routes through StudentProxy, HelpdeskCoordinator, and RetrievalQASpecialist.",
    )

    show_grounding_context = st.checkbox(
        "Show Grounding Context",
        value=True,
        help="Display the exact policy text chunks retrieved from LlamaIndex.",
    )

    st.divider()

    st.subheader("💡 Quick Sample Questions")
    sample_queries = [
        "Procedure for applying for a bonafide certificate",
        "Documents required for examination registration",
        "Attendance requirement & medical condonation",
        "How to download official fee receipt",
        "Central library timings & weekend hours",
        "Course registration add/drop deadline",
        "Transfer certificate no-dues clearance steps",
        "Fee refund policy if admission is cancelled",
    ]

    selected_sample = None
    for q in sample_queries:
        if st.button(f"📌 {q}", use_container_width=True):
            selected_sample = q

    st.divider()

    st.subheader("📂 Knowledge Base Manuals")
    doc_files = sorted([f.name for f in Path("./data/documents").glob("*.md")])
    chosen_doc = st.selectbox("Inspect Official Policy Document", options=doc_files)
    if chosen_doc:
        doc_path = Path("./data/documents") / chosen_doc
        if doc_path.exists():
            with st.expander(f"📄 View {chosen_doc}"):
                st.markdown(doc_path.read_text(encoding="utf-8"))

    st.divider()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [st.session_state.messages[0]]
            st.rerun()

    with col_btn2:
        if st.button("🔄 Re-index", use_container_width=True):
            with st.spinner("Rebuilding Vector Index..."):
                indexer.build_or_load_index(force_rebuild=True)
            st.success("Index refreshed!")


# ----------------- MAIN CHAT AREA -----------------
st.markdown('<div class="main-title">🎓 AI-Based Student Helpdesk</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Apex Institute of Technology & Sciences — '
    'Grounding: <strong>LlamaIndex (RAG)</strong> | Reasoning: <strong>LangChain</strong> | '
    'Coordination: <strong>AutoGen Multi-Agent</strong></div>',
    unsafe_allow_html=True,
)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("category") and msg["category"] != "welcome":
            st.markdown(
                f'<span class="badge-category">📌 Category: {msg["category"].upper()}</span>',
                unsafe_allow_html=True,
            )
        if msg.get("flow"):
            st.markdown(
                f'<span class="badge-flow">🔄 Flow: {" → ".join(msg["flow"])}</span>',
                unsafe_allow_html=True,
            )

        st.markdown(msg["content"])

        if show_grounding_context and msg.get("context"):
            with st.expander("📚 Retrieved Grounding Context & Source Clauses"):
                st.markdown(msg["context"])


# Function to process and respond to queries
def process_user_query(user_text: str):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Consulting official university regulations..."):
            if engine_mode == "AutoGen Multi-Agent":
                result = coordinator.process_query(user_text)
                answer_text = result["answer"]
                context_text = result.get("context", "")
                category = result.get("category", "general")
                flow = result.get("agent_flow", ["StudentProxy", "HelpdeskCoordinator", "RetrievalQASpecialist"])
            else:
                result = pipeline.answer_question(user_text)
                answer_text = result["answer"]
                context_text = result.get("context", "")
                category = "direct_qa"
                flow = ["User", "StudentHelpdeskQA"]

            if category and category != "welcome":
                st.markdown(
                    f'<span class="badge-category">📌 Category: {category.upper()}</span>',
                    unsafe_allow_html=True,
                )
            if flow:
                st.markdown(
                    f'<span class="badge-flow">🔄 Flow: {" → ".join(flow)}</span>',
                    unsafe_allow_html=True,
                )

            st.markdown(answer_text)

            if show_grounding_context and context_text:
                with st.expander("📚 Retrieved Grounding Context & Source Clauses"):
                    st.markdown(context_text)

    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "category": category,
            "flow": flow,
            "context": context_text,
        }
    )


# Handle quick question click from sidebar
if selected_sample:
    process_user_query(selected_sample)

# Handle chat input from user
prompt_input = st.chat_input("Ask a question about college regulations, fees, attendance, certificates...")
if prompt_input:
    process_user_query(prompt_input)
