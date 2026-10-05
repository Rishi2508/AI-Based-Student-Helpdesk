"""AutoGen multi-agent coordination layer for the Student Helpdesk.

Implements the multi-agent workflow:
Student Query -> AutoGen Agent Coordinator -> LangChain/LlamaIndex QA Retrieval -> Final Response.
Designated Roles:
- StudentProxy (UserProxyAgent): Represents the student query input and terminal handoff.
- HelpdeskCoordinator (QueryRouterAgent): Parses the query category (fees, attendance, certificates, exams, library, general).
- RetrievalQASpecialist: Equipped with the tool invoking StudentHelpdeskQA to fetch verified document answers.
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

from src.qa_pipeline import StudentHelpdeskQA, get_default_pipeline


class OfflineHelpdeskAgent:
    """Lightweight representation of an AutoGen agent role for offline/testing mode."""

    def __init__(self, name: str, role: str, description: str = "") -> None:
        self.name = name
        self.role = role
        self.description = description

    def __repr__(self) -> str:
        return f"<OfflineHelpdeskAgent name={self.name!r} role={self.role!r}>"


# Support both classic autogen (0.2.x) and modern autogen-agentchat (0.4.x+)
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent
    HAS_AUTOGEN = True
    AUTOGEN_FLAVOR = "classic"
except ImportError:
    try:
        from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
        import autogen_agentchat as autogen
        HAS_AUTOGEN = True
        AUTOGEN_FLAVOR = "agentchat"
    except ImportError:
        HAS_AUTOGEN = False
        AUTOGEN_FLAVOR = "none"


class StudentHelpdeskCoordinator:
    """Coordinates multi-agent workflows connecting student queries to the LangChain/LlamaIndex QA pipeline."""

    SUPPORTED_CATEGORIES = [
        "fees",
        "attendance",
        "certificates",
        "exams",
        "library",
        "course_registration",
        "general",
    ]

    def __init__(
        self,
        qa_pipeline: Optional[StudentHelpdeskQA] = None,
        llm_model: str = "gpt-4o-mini",
    ) -> None:
        self.qa_pipeline = qa_pipeline or get_default_pipeline()
        self.llm_model = os.getenv("OPENAI_MODEL_NAME", llm_model)
        self.api_key = os.getenv("OPENAI_API_KEY", "")

        self._init_llm_config()
        self._init_agents()

    def _init_llm_config(self) -> None:
        """Sets up AutoGen configuration list."""
        if self.api_key and self.api_key != "your_openai_api_key_here":
            self.llm_config = {
                "config_list": [
                    {
                        "model": self.llm_model,
                        "api_key": self.api_key,
                    }
                ],
                "temperature": 0.2,
                "timeout": 120,
            }
        else:
            self.llm_config = False

    def parse_query_category(self, query: str) -> str:
        """Parses the student query category into one of: fees, attendance, certificates, exams, library, course_registration, general."""
        q_lower = query.lower()
        if any(w in q_lower for w in ["fee", "tuition", "refund", "receipt", "payment", "bank", "dd", "caution deposit"]):
            return "fees"
        elif any(w in q_lower for w in ["attend", "attendance", "medical leave", "condonation", "on-duty", "od", "absent", "debarred"]):
            return "attendance"
        elif any(w in q_lower for w in ["bonafide", "transfer certificate", "tc", "migration", "certificate", "no-dues"]):
            return "certificates"
        elif any(w in q_lower for w in ["exam", "examination", "hall ticket", "admit card", "arrear", "revaluation", "malpractice"]):
            return "exams"
        elif any(w in q_lower for w in ["library", "book", "borrow", "renew", "fine", "overdue", "journal"]):
            return "library"
        elif any(w in q_lower for w in ["course", "registration", "credit", "add/drop", "elective", "cbcs", "prerequisite"]):
            return "course_registration"
        return "general"

    def _lookup_college_policy(self, query: str) -> Dict[str, Any]:
        """Tool function invoked by the RetrievalQASpecialist to fetch verified document answers."""
        return self.qa_pipeline.answer_question(query)

    def _init_agents(self) -> None:
        """Initializes the three designated multi-agent roles."""
        self.lookup_tool = self._lookup_college_policy

        if not self.llm_config or AUTOGEN_FLAVOR != "classic":
            # Resilient offline/mock representation when no live API key is configured
            self.student_proxy = OfflineHelpdeskAgent(
                name="StudentProxy",
                role="Student Query Proxy",
                description="Represents student input and terminal handoff.",
            )
            self.router_agent = OfflineHelpdeskAgent(
                name="HelpdeskCoordinator",
                role="Query Router",
                description="Parses query category (fees, attendance, certificates, exams, library, general).",
            )
            self.qa_specialist = OfflineHelpdeskAgent(
                name="RetrievalQASpecialist",
                role="Retrieval QA Specialist",
                description="Invokes StudentHelpdeskQA to fetch grounded college policy context and answers.",
            )
            # Backward-compatible references
            self.triage_agent = self.router_agent
            self.policy_advisor = self.qa_specialist
            return

        # Classic AutoGen live multi-agent configuration
        self.student_proxy = UserProxyAgent(
            name="StudentProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            is_termination_msg=lambda x: "TERMINATE" in (x.get("content", "") or ""),
            code_execution_config=False,
            system_message="You are the StudentProxy. You represent the student's question and complete the handoff when an authoritative answer is received.",
        )

        self.router_agent = AssistantAgent(
            name="HelpdeskCoordinator",
            llm_config=self.llm_config,
            system_message="""You are the HelpdeskCoordinator (QueryRouterAgent) at Apex Institute.
Your responsibilities:
1. Greet the student cordially and identify the query category (fees, attendance, certificates, exams, library, course_registration, or general).
2. Delegate the query to the RetrievalQASpecialist to fetch verified policy details from the knowledge base.
3. Once the RetrievalQASpecialist provides verified policy context, present a concise and structured answer to the student.
4. Conclude your final response with 'TERMINATE' to end the session.""",
        )

        self.qa_specialist = AssistantAgent(
            name="RetrievalQASpecialist",
            llm_config=self.llm_config,
            system_message="""You are the RetrievalQASpecialist equipped with direct access to official college manuals.
Your job is to look up official rules, fee schedules, forms, and procedures using your lookup_policy tool.
Always cite the Document ID and office contacts in your findings.""",
        )

        # Register retrieval tool with AutoGen agents
        if hasattr(autogen, "agentchat") and hasattr(autogen.agentchat, "register_function"):
            autogen.agentchat.register_function(
                self.lookup_tool,
                caller=self.router_agent,
                executor=self.student_proxy,
                name="lookup_college_policy",
                description="Query the official college document knowledge base for rules, steps, and policies.",
            )

        # Backward-compatible references
        self.triage_agent = self.router_agent
        self.policy_advisor = self.qa_specialist

    def process_query(self, query: str) -> Dict[str, Any]:
        """Initiates the multi-agent conversation/task execution and extracts the final validated answer.

        Returns a dictionary adhering to:
        {
          "query": query,
          "answer": final_answer_text,
          "context": retrieved_context,
          "agent_flow": ["StudentProxy", "HelpdeskCoordinator", "RetrievalQASpecialist"]
        }
        """
        category = self.parse_query_category(query)
        agent_flow = ["StudentProxy", "HelpdeskCoordinator", "RetrievalQASpecialist"]

        if not self.llm_config or not isinstance(self.student_proxy, (getattr(autogen, "UserProxyAgent", object),)):
            # Offline / Fallback multi-agent simulation
            qa_result = self.qa_pipeline.answer_question(query)
            final_answer = qa_result["answer"]
            retrieved_context = qa_result["context"]

            return {
                "query": query,
                "answer": final_answer,
                "context": retrieved_context,
                "category": category,
                "agent_flow": agent_flow,
                # Backward-compatible convenience keys
                "summary": final_answer,
                "sources": qa_result.get("sources", []),
                "student_query": query,
                "status": "completed",
            }

        # Live AutoGen execution
        init_message = (
            f"Student Query: '{query}'\n"
            f"Detected Category: [{category.upper()}].\n"
            f"Please coordinate with RetrievalQASpecialist to fetch official policies and provide guidance."
        )

        chat_result = self.student_proxy.initiate_chat(
            recipient=self.router_agent,
            message=init_message,
        )

        # Retrieve ground truth context directly for inclusion in response
        qa_data = self.qa_pipeline.answer_question(query)
        final_answer = chat_result.summary if hasattr(chat_result, "summary") and chat_result.summary else qa_data["answer"]

        return {
            "query": query,
            "answer": str(final_answer),
            "context": qa_data["context"],
            "category": category,
            "agent_flow": agent_flow,
            "summary": str(final_answer),
            "sources": qa_data.get("sources", []),
            "student_query": query,
            "status": "completed",
        }


# Aliases for backward compatibility
HelpdeskAgentCoordinator = StudentHelpdeskCoordinator


def get_default_coordinator() -> StudentHelpdeskCoordinator:
    """Convenience factory function."""
    return StudentHelpdeskCoordinator()


if __name__ == "__main__":
    coordinator = StudentHelpdeskCoordinator()
    sample_queries = [
        "What documents are required for examination registration?",
        "How can a fee receipt be obtained?",
    ]

    print("=" * 75)
    print("🤖 AI-Based Student Helpdesk: AutoGen Multi-Agent Coordinator")
    print("=" * 75)

    for idx, query in enumerate(sample_queries, start=1):
        print(f"\n[Test Query {idx}]: \"{query}\"")
        result = coordinator.process_query(query)

        print(f"Category Routed : {result.get('category')}")
        print(f"Agent Flow      : {' -> '.join(result.get('agent_flow', []))}")
        print("\n--- Final Validated Answer ---")
        print(result["answer"])
        print("\n--- Retrieved Grounding Context Preview ---")
        print(result["context"][:300].strip() + "...\n")
        print("-" * 75)
