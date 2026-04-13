import os
from typing import Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. Import your State and Agents
from uda_hub.agentic.tools.udahub_state import UDAHubState
from uda_hub.agentic.agents.request_enricher import data_enricher_node
from uda_hub.agentic.agents.classifier import classifier_node
from uda_hub.agentic.agents.researcher import researcher_node
from uda_hub.agentic.agents.drafter import drafter_node
from uda_hub.agentic.agents.policy_checker import policy_checker_node
from uda_hub.agentic.agents.archivist import archivist_node
from uda_hub.agentic.agents.clarificator import clarification_node
from uda_hub.agentic.agents.escalator import escalation_node

load_dotenv()

# ==========================================
# 2. ROUTING LOGIC (The Supervisor)
# ==========================================


def supervisor_router(
    state: UDAHubState,
) -> Literal["researcher", "escalate", "clarify"]:
    """
    Determines if we have enough data to proceed or if we need a human/more info.
    """
    category = state.get("category")
    urgency = state.get("urgency")
    user_id = state.get("user_id")

    # If it's a critical issue or VIP, bypass automation
    if urgency == "high":
        return "escalate"

    # If we are missing identity for a specific account issue
    if not user_id and category in ["billing", "technical_issue"]:
        return "clarify"

    return "researcher"


def qa_router(state: UDAHubState) -> Literal["drafter", "archivist"]:
    """
    Determines if the draft passed the policy check.
    """
    if state.get("policy_grade") == "FAIL":
        return "drafter"  # Loop back to fix the hallucination
    return "archivist"


# ==========================================
# 3. BUILD THE COMPLETE GRAPH
# ==========================================

workflow = StateGraph(UDAHubState)

# Add every single agent as a node
workflow.add_node("enricher", data_enricher_node)
workflow.add_node("classifier", classifier_node)
workflow.add_node("clarifier", clarification_node)
workflow.add_node("escalator", escalation_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("drafter", drafter_node)
workflow.add_node("policy_checker", policy_checker_node)
workflow.add_node("archivist", archivist_node)

# --- Define the Connections ---

workflow.add_edge(START, "enricher")
workflow.add_edge("enricher", "classifier")

# The Supervisor decides the path
workflow.add_conditional_edges(
    "classifier",
    supervisor_router,
    {"researcher": "researcher", "escalator": "escalator", "clarifier": "clarifier"},
)

# Terminal Paths for non-automated issues
workflow.add_edge("clarifier", END)
workflow.add_edge("escalator", END)

# The Main Automation Loop
workflow.add_edge("researcher", "drafter")
workflow.add_edge("drafter", "policy_checker")

workflow.add_conditional_edges(
    "policy_checker", qa_router, {"drafter": "drafter", "archivist": "archivist"}
)

workflow.add_edge("archivist", END)

# 4. Compile
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Extract the binary data from the graph
png_data = app.get_graph().draw_mermaid_png()

# Save to the current directory
with open("uda_hub_workflow.png", "wb") as f:
    f.write(png_data)

print("Workflow diagram saved as 'uda_hub_workflow.png'")

# ==========================================
# 4. COMPILE & EXPORT
# ==========================================

# Memory keeps track of the conversation threads
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    # Final Integration Test
    initial_state = {
        "ticket_text": "I can't attend my Christ the Redeemer booking. Refund?",
        "user_email": "alice.kingsley@wonderland.com",
    }

    config = {"configurable": {"thread_id": "test_run_001"}}

    for event in app.stream(initial_state, config):
        for node, values in event.items():
            print(f"\n--- Node: {node} ---")
            # We print small snippets of the state updates
            if "research_facts" in values:
                print(f"Facts found: {values['research_facts'][:100]}...")
            if "ai_response" in values:
                print(f"Draft generated!")
            if "policy_grade" in values:
                print(f"QA Result: {values['policy_grade']}")
