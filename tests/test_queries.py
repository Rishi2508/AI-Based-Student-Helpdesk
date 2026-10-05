"""Test suite verifying knowledge base documents, indexer, and query pipelines."""

from pathlib import Path
import pytest

from src.indexer import CollegeDocumentIndexer, DocumentIndexer
from src.qa_pipeline import StudentHelpdeskQA, StudentHelpdeskQAPipeline
from src.agent_coordinator import HelpdeskAgentCoordinator, StudentHelpdeskCoordinator


DOCS_PATH = Path("./data/documents")

EXPECTED_DOCUMENTS = [
    "bonafide_certificate_policy.md",
    "exam_regulations.md",
    "attendance_rules.md",
    "fee_structure_and_refund.md",
    "library_guide.md",
    "course_registration.md",
    "transfer_certificate_policy.md",
]


class TestDocumentKnowledgeBase:
    """Verifies that all required college policy documents exist and contain required clauses."""

    @pytest.mark.parametrize("doc_filename", EXPECTED_DOCUMENTS)
    def test_document_exists(self, doc_filename: str):
        file_path = DOCS_PATH / doc_filename
        assert file_path.exists(), f"Document {doc_filename} missing in {DOCS_PATH}"
        content = file_path.read_text(encoding="utf-8")
        assert len(content) > 300, f"Document {doc_filename} appears too short or empty."

    def test_bonafide_policy_content(self):
        content = (DOCS_PATH / "bonafide_certificate_policy.md").read_text(encoding="utf-8")
        assert "AITS-ADM-POL-014" in content
        assert "Smart Card" in content or "Student ID" in content
        assert "2 working days" in content
        assert "Tatkal" in content or "Urgent" in content

    def test_exam_regulations_content(self):
        content = (DOCS_PATH / "exam_regulations.md").read_text(encoding="utf-8")
        assert "AITS-EXAM-REG-022" in content
        assert "Hall Ticket" in content or "Admit Card" in content
        assert "75%" in content
        assert "Controller of Examinations" in content

    def test_attendance_rules_content(self):
        content = (DOCS_PATH / "attendance_rules.md").read_text(encoding="utf-8")
        assert "75%" in content
        assert "65%" in content
        assert "Medical Condonation" in content or "medical leave" in content.lower()
        assert "Chief Medical Officer" in content or "CMO" in content

    def test_fee_structure_and_refund_content(self):
        content = (DOCS_PATH / "fee_structure_and_refund.md").read_text(encoding="utf-8")
        assert "B.Tech" in content
        assert "UPI" in content or "NEFT" in content
        assert "Caution Deposit" in content
        assert "100%" in content
        assert "Refund" in content

    def test_library_guide_content(self):
        content = (DOCS_PATH / "library_guide.md").read_text(encoding="utf-8")
        assert "Dr. APJ Abdul Kalam" in content
        assert "Overdue Fines" in content or "₹5" in content
        assert "14 Days" in content or "borrowing" in content.lower()
        assert "Renew" in content

    def test_course_registration_content(self):
        content = (DOCS_PATH / "course_registration.md").read_text(encoding="utf-8")
        assert "Choice-Based Credit System" in content or "CBCS" in content
        assert "Add / Drop" in content
        assert "Minimum Credits" in content
        assert "Day 10" in content

    def test_transfer_certificate_content(self):
        content = (DOCS_PATH / "transfer_certificate_policy.md").read_text(encoding="utf-8")
        assert "AITS-REG-TC-019" in content
        assert "No-Dues" in content
        assert "Surrendered" in content or "surrendered" in content
        assert "Caution Deposit" in content


class TestIndexerAndComponents:
    """Tests loading documents through the LlamaIndex document reader."""

    def test_indexer_loads_all_documents(self):
        indexer = DocumentIndexer(docs_dir=DOCS_PATH)
        docs = indexer.load_documents()
        assert len(docs) >= len(EXPECTED_DOCUMENTS)
        loaded_files = {d.metadata.get("file_name") for d in docs}
        for expected in EXPECTED_DOCUMENTS:
            assert expected in loaded_files

    def test_qa_pipeline_context_formatting(self):
        pipeline = StudentHelpdeskQAPipeline()

        class DummyNode:
            def __init__(self, text: str, file_name: str):
                self.text = text
                self.metadata = {"file_name": file_name}

        dummy_nodes = [
            DummyNode("Policy 1 details here", "attendance_rules.md"),
            DummyNode("Policy 2 details here", "exam_regulations.md"),
        ]
        context = pipeline.format_context(dummy_nodes)
        assert "attendance_rules.md" in context
        assert "exam_regulations.md" in context
        assert "Policy 1 details here" in context

    def test_agent_coordinator_initialization(self):
        coordinator = HelpdeskAgentCoordinator()
        assert coordinator.student_proxy is not None
        assert coordinator.triage_agent is not None
        assert coordinator.policy_advisor is not None
        assert callable(coordinator.lookup_tool)

    def test_college_document_indexer_build_and_retrieve_context(self, tmp_path):
        indexer = CollegeDocumentIndexer(docs_dir=DOCS_PATH, storage_dir=tmp_path / "storage")
        index = indexer.build_or_load_index()
        assert index is not None

        retriever = indexer.get_retriever(similarity_top_k=2)
        assert retriever is not None

        context = indexer.retrieve_context("What is the attendance requirement?", top_k=2)
        assert isinstance(context, str)
        assert len(context) > 50
        assert "Document Chunk" in context

    def test_student_helpdesk_qa_answer_question(self):
        qa = StudentHelpdeskQA()
        result = qa.answer_question("What is the procedure for applying for a bonafide certificate?")
        assert isinstance(result, dict)
        assert "query" in result
        assert "answer" in result
        assert "context" in result
        assert result["query"] == "What is the procedure for applying for a bonafide certificate?"
        assert len(result["context"]) > 50
        assert len(result["answer"]) > 10

    def test_student_helpdesk_coordinator_process_query(self):
        coordinator = StudentHelpdeskCoordinator()
        query = "What documents are required for examination registration?"
        result = coordinator.process_query(query)
        assert isinstance(result, dict)
        assert result["query"] == query
        assert "answer" in result and len(result["answer"]) > 10
        assert "context" in result and len(result["context"]) > 10
        assert result["agent_flow"] == ["StudentProxy", "HelpdeskCoordinator", "RetrievalQASpecialist"]
        assert result["category"] == "exams"
