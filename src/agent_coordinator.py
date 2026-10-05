"""AutoGen multi-agent coordination system for the Student Helpdesk.

Orchestrates multi-agent interactions between:
- Student Proxy Agent (represents user/student queries)
- Helpdesk Triage Agent (categorizes intent, manages workflow, ensures clarity)
- Policy Advisor Agent (domain specialist equipped with the LlamaIndex/LangChain QA tools)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()


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

from src.qa_pipeline import StudentHelpdeskQAPipeline, get_default_pipeline


class HelpdeskAgentCoordinator:
    """Coordinates AutoGen agents for multi-perspective student problem resolution."""

    def __init__(
        self,
        qa_pipeline: Optional[StudentHelpdeskQAPipeline] = None,
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

    def _lookup_college_policy(self, query: str) -> str:
        """Looks up verified information from the college knowledge base."""
        result = self.qa_pipeline.answer_query(query)
        return (
            f"--- Verified Policy Answer ---\n"
            f"{result['answer']}\n"
            f"Source Documents: {', '.join(result['sources'])}"
        )

    def _init_agents(self) -> None:
        """Initializes the multi-agent roles."""
        self.lookup_tool = self._lookup_college_policy

        # Fallback / offline representation when no API key or agentchat without client
        if not self.llm_config:
            self.student_proxy = OfflineHelpdeskAgent(
                name="Student_Proxy",
                role="Student Proxy",
                description="Presents student inquiries and receives helpdesk advice.",
            )
            self.triage_agent = OfflineHelpdeskAgent(
                name="Helpdesk_Triage_Agent",
                role="Front-Desk Triage Coordinator",
                description="Identifies domain of inquiry and coordinates policy specialists.",
            )
            self.policy_advisor = OfflineHelpdeskAgent(
                name="Policy_Advisor_Agent",
                role="Senior Policy Advisor",
                description="Consults knowledge base and provides authoritative policy guidance.",
            )
            return

        if AUTOGEN_FLAVOR == "classic":
            self.student_proxy = UserProxyAgent(
                name="Student_Proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=3,
                is_termination_msg=lambda x: "TERMINATE" in (x.get("content", "") or ""),
                code_execution_config=False,
                system_message="You are the Student Proxy. You present the student's question and terminate once an authoritative answer is received.",
            )

            self.triage_agent = AssistantAgent(
                name="Helpdesk_Triage_Agent",
                llm_config=self.llm_config,
                system_message="""You are the Front-Desk Student Services Coordinator at Apex Institute.
Your responsibilities:
1. Greet the student cordially and identify the exact domain of the inquiry.
2. Ask the Policy_Advisor_Agent to consult the knowledge base for verified regulations and exact procedures.
3. Once the Policy_Advisor_Agent provides the policy details, present a complete, supportive, and formatted summary for the student.
4. Conclude your final response with 'TERMINATE' once the query is satisfactorily addressed.""",
            )

            self.policy_advisor = AssistantAgent(
                name="Policy_Advisor_Agent",
                llm_config=self.llm_config,
                system_message="""You are the Senior Academic Policy Advisor with direct access to all official college policy manuals.
Your job is to answer queries by referencing verified regulations, fee structures, deadlines, and required documentation.
Always cite the Document ID and office locations when providing advice.""",
            )

            if hasattr(autogen, "agentchat") and hasattr(autogen.agentchat, "register_function"):
                autogen.agentchat.register_function(
                    self.lookup_tool,
                    caller=self.triage_agent,
                    executor=self.student_proxy,
                    name="lookup_college_policy",
                    description="Query the official college document knowledge base for rules, steps, and policies.",
                )
        else:
            # Modern autogen_agentchat initialization when client is configured
            self.student_proxy = OfflineHelpdeskAgent(
                name="Student_Proxy",
                role="Student Proxy",
                description="Presents student inquiries.",
            )
            self.triage_agent = OfflineHelpdeskAgent(
                name="Helpdesk_Triage_Agent",
                role="Triage Coordinator",
                description="Front-desk triage coordinator.",
            )
            self.policy_advisor = OfflineHelpdeskAgent(
                name="Policy_Advisor_Agent",
                role="Policy Advisor",
                description="Consults knowledge base.",
            )

    def process_query(self, student_query: str) -> Dict[str, Any]:
        """Runs the collaborative multi-agent workflow to solve a student query."""
        if not self.llm_config or not isinstance(self.student_proxy, (getattr(autogen, "UserProxyAgent", object),)):
            # Running in dry-run/mock mode when OPENAI_API_KEY is not configured
            direct_result = self.qa_pipeline.answer_query(student_query)
            return {
                "status": "completed",
                "mode": "standalone_qa_direct",
                "student_query": student_query,
                "summary": direct_result["answer"],
                "sources": direct_result["sources"],
                "chat_history": [
                    {"sender": "Student_Proxy", "content": student_query},
                    {
                        "sender": "Policy_Advisor_Agent",
                        "content": direct_result["answer"],
                    },
                ],
            }

        chat_result = self.student_proxy.initiate_chat(
            recipient=self.triage_agent,
            message=f"A student is inquiring: '{student_query}'. Please verify our official policies and provide guidance.",
        )

        return {
            "status": "completed",
            "mode": "autogen_multi_agent",
            "student_query": student_query,
            "chat_history": chat_result.chat_history,
            "summary": chat_result.summary if hasattr(chat_result, "summary") else str(chat_result),
        }


def get_default_coordinator() -> HelpdeskAgentCoordinator:
    """Convenience factory function."""
    return HelpdeskAgentCoordinator()


if __name__ == "__main__":
    coordinator = get_default_coordinator()
    test_q = "What is the minimum attendance required to write the semester exams?"
    print(f"Running agent coordinator for: {test_q}")
    res = coordinator.process_query(test_q)
    print("\nResult:")
    print(res["summary"])
