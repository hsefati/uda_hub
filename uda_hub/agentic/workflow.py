from typing import Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage
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


def entry_bridge_node(state: UDAHubState):
    """
    Extracts the latest message from the history to populate the internal
    processing state.
    """
    # Grab the content of the very last message from the chat UI
    latest_input = state["messages"][-1].content

    # We return the updates to our custom state keys
    return {
        "ticket_text": latest_input,
        # We can also clear previous turn facts to ensure fresh research
        "research_facts": None,
    }


workflow = StateGraph(UDAHubState)

# Add every single agent as a node
workflow.add_node("entry_bridge", entry_bridge_node)
workflow.add_node("enricher", data_enricher_node)
workflow.add_node("classifier", classifier_node)
workflow.add_node("clarifier", clarification_node)
workflow.add_node("escalator", escalation_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("drafter", drafter_node)
workflow.add_node("policy_checker", policy_checker_node)
workflow.add_node("archivist", archivist_node)

# --- Define the Connections ---
workflow.add_edge(START, "entry_bridge")
workflow.add_edge("entry_bridge", "enricher")
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

# Activate only when there is a need to generate the workflow diagram (e.g. after edits to the graph structure)
# # Extract the binary data from the graph
# png_data = app.get_graph().draw_mermaid_png()

# # Save to the current directory
# with open("uda_hub_workflow.png", "wb") as f:
#     f.write(png_data)

print("Workflow diagram saved as 'uda_hub_workflow.png'")


# Memory keeps track of the conversation threads
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


if __name__ == "__main__":
    print("🚀 Starting UDA-Hub End-to-End Integration Test\n" + "=" * 50)

    initial_state = {
        "messages": [
            HumanMessage(content="I can't attend my Christ the Redeemer booking. Refund?")
        ],
        "user_email": "alice.kingsley@wonderland.com",
    }

    config = {"configurable": {"thread_id": "test_run_alice_001"}}

    # We use stream to observe the 'Assembly Line' in action
    for event in app.stream(initial_state, config):
        for node, values in event.items():
            print(f"\n📍 NODE EXECUTED: {node}")
            print("-" * 30)

            # 1. Show Identity Enrichment
            if "user_id" in values:
                print(
                    f"👤 User Identified: {values.get('user_id')} ({values.get('customer_tier')} tier)"
                )

            # 2. Show Intent Classification
            if "category" in values:
                print(
                    f"🏷️  Classification: {values.get('category').upper()} | Urgency: {values.get('urgency')}"
                )

            # 3. Show Research Discoveries
            if "research_facts" in values and values["research_facts"]:
                print(f"🔍 Knowledge Retrieved:\n{values['research_facts']}")

            # 4. Show Draft & Policy Loop
            if "ai_response" in values:
                # If there's a policy grade, it means we just passed the QA node
                grade = values.get("policy_grade", "PENDING")
                print(f"📝 Drafted Response (QA Status: {grade})")

                if grade == "FAIL":
                    print(f"⚠️  POLICY VIOLATION: {values.get('policy_feedback')}")
                    print("🔄 Routing back to Drafter for correction...")
                else:
                    # Print a snippet of the successful response
                    print(f"📩 Final Email Preview: {values['ai_response']}")

            # 5. Show Final Archiving
            if "archive_summary" in values:
                print(f"🗄️  Archived Summary: {values['archive_summary']}")

    print("\n" + "=" * 50 + "\n✅ Test Sequence Complete")
