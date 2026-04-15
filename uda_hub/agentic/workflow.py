import os
from typing import Literal
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Import existing state and nodes
from uda_hub.agentic.tools.udahub_state import UDAHubState
from uda_hub.agentic.agents.request_enricher import data_enricher_node
from uda_hub.agentic.agents.classifier import classifier_node
from uda_hub.agentic.agents.researcher import researcher_node
from uda_hub.agentic.agents.drafter import drafter_node
from uda_hub.agentic.agents.policy_checker import policy_checker_node
from uda_hub.agentic.agents.archivist import archivist_node
from uda_hub.agentic.agents.clarificator import clarification_node
from uda_hub.agentic.agents.escalator import escalation_node
from uda_hub.agentic.agents.supervisor import supervisor_router
from uda_hub.agentic.agents.greater import greeter_node
from uda_hub.agentic.agents.closure_node import closer_node

load_dotenv()

# ==========================================
# 1. ROUTING & BRIDGE LOGIC
# ==========================================


def qa_router(state: UDAHubState) -> Literal["drafter", "archivist"]:
    """
    Determines if the draft passed the policy check.
    """
    if state.get("policy_grade") == "FAIL":
        return "drafter"  # Loop back to fix the hallucination
    return "archivist"


def entry_bridge_node(state: UDAHubState):
    """
    Prepares the state for a new turn.
    Clears temporary decision fields while preserving the conversation history.
    """
    messages = state.get("messages", [])
    latest_input = messages[-1].content if messages else ""

    return {
        "latest_input": latest_input,
        "research_facts": None,  # Reset for new research pass
        "policy_grade": None,  # Clear previous audit
        "policy_feedback": None,  # Clear previous feedback
        "ai_response": None,  # Clear previous draft
    }


# ==========================================
# 2. GRAPH DEFINITION
# ==========================================

workflow = StateGraph(UDAHubState)

# Add Nodes
workflow.add_node("entry_bridge", entry_bridge_node)
workflow.add_node("enricher", data_enricher_node)
workflow.add_node("classifier", classifier_node)
workflow.add_node("greeter", greeter_node)
workflow.add_node("closer", closer_node)
workflow.add_node("clarifier", clarification_node)
workflow.add_node("escalator", escalation_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("drafter", drafter_node)
workflow.add_node("policy_checker", policy_checker_node)
workflow.add_node("archivist", archivist_node)

# --- Define the Connections ---

# Initial Pipeline
workflow.add_edge(START, "entry_bridge")
workflow.add_edge("entry_bridge", "enricher")
workflow.add_edge("enricher", "classifier")

# Supervisor Decision (Inbound Traffic)
workflow.add_conditional_edges(
    "classifier",
    supervisor_router,
    {
        "greeter": "greeter",
        "closer": "closer",
        "researcher": "researcher",
        "escalator": "escalator",
        "clarifier": "clarifier",
    },
)

# Automation & Quality Loop
workflow.add_edge("researcher", "drafter")
workflow.add_edge("drafter", "policy_checker")

workflow.add_conditional_edges(
    "policy_checker", qa_router, {"drafter": "drafter", "archivist": "archivist"}
)

# All terminal paths flow to Archivist to persist authenticated data
workflow.add_edge("greeter", "archivist")
workflow.add_edge("closer", "archivist")
workflow.add_edge("clarifier", "archivist")
workflow.add_edge("escalator", "archivist")

# The Final Finish
workflow.add_edge("archivist", END)

# ==========================================
# 3. COMPILATION & EXPORT
# ==========================================

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Optional: Generate visualization
# try:
#     png_data = app.get_graph().draw_mermaid_png()
#     with open("uda_hub_workflow.png", "wb") as f:
#         f.write(png_data)
#     print("🎨 Workflow diagram updated: 'uda_hub_workflow.png'")
# except Exception:
#     print("⚠️  Could not generate diagram (pygraphviz/mermaid issues), skipping...")

# ==========================================
# 4. INTEGRATION TEST
# ==========================================

if __name__ == "__main__":
    print("\n🚀 Starting UDA-Hub End-to-End Integration Test\n" + "=" * 50)

    initial_state = {
        "messages": [
            HumanMessage(
                content="I can't attend my Christ the Redeemer booking. Refund?. My email is alice.kingsley@wonderland.com."
            )
        ]
    }

    config = {
        "configurable": {"thread_id": f"test_{datetime.now().strftime('%Y%m%d_%H%M')}"}
    }

    for event in app.stream(initial_state, config):
        for node, values in event.items():
            print(f"\n📍 NODE: {node}")

            # Identity Check
            if "user_id" in values and values["user_id"]:
                print(
                    f"👤 Identity: {values['user_id']} | Tier: {values.get('customer_tier')}"
                )

            # Intent Check
            if "category" in values:
                print(f"🏷️  Intent: {values['category'].upper()}")

            # Research Facts
            if "research_facts" in values and values["research_facts"]:
                print(f"🔍 Facts: {values['research_facts']}...")

            # Response & QA Loop
            if "ai_response" in values:
                grade = values.get("policy_grade", "PENDING")
                print(f"📝 Draft Status: {grade}")
                if values.get("policy_feedback"):
                    print(f"⚠️  Feedback: {values['policy_feedback']}")
                else:
                    print(f"📩 Message: {values['ai_response']}...")

            # Archive Result
            if "archive_summary" in values:
                print(f"🗄️  Archived: {values['archive_summary']}")

    print("\n" + "=" * 50 + "\n✅ Integration Test Complete")
