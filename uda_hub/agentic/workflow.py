from typing import Literal
import uuid
from dotenv import load_dotenv
from uda_hub.agentic.tools.logging import node_log, get_tool_evidence

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# Import existing state and nodes
from uda_hub.agentic.tools.udahub_state import UDAHubState
from uda_hub.agentic.agents.enricher import data_enricher_node
from uda_hub.agentic.agents.classifier import classifier_node
from uda_hub.agentic.agents.researcher import researcher_node, research_safety_router
from uda_hub.agentic.agents.drafter import drafter_node
from uda_hub.agentic.agents.policy_checker import policy_checker_node
from uda_hub.agentic.agents.archivist import archivist_node
from uda_hub.agentic.agents.escalator import escalation_node
from uda_hub.agentic.agents.supervisor import supervisor_router
from uda_hub.agentic.agents.historian import historian_node
from uda_hub.agentic.agents.concierge import concierge_node, concierge_router

load_dotenv()

# Use shared node_log from uda_hub.agentic.tools.logging

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


# ==========================================
# 2. GRAPH DEFINITION
# ==========================================

workflow = StateGraph(UDAHubState)

# Add Nodes
workflow.add_node("historian", historian_node)
workflow.add_node("enricher", data_enricher_node)
workflow.add_node("classifier", classifier_node)
workflow.add_node("concierge", concierge_node)
workflow.add_node("escalator", escalation_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("drafter", drafter_node)
workflow.add_node("policy_checker", policy_checker_node)
workflow.add_node("archivist", archivist_node)

# --- Define the Connections ---

# Initial Pipeline
workflow.add_edge(START, "concierge")

# concierge Decision (Inbound Traffic)
workflow.add_conditional_edges(
    "concierge",
    concierge_router,
    {
        "enricher": "enricher",     # Matches intent == "support_request"
        "archivist": "archivist",   # Matches intent == "closure"
        "END": END                  # Matches intent == "greeting" (Goes to START/END)
    },
)

workflow.add_edge("enricher", "historian")
workflow.add_edge("historian", "classifier")

# Supervisor Decision (Inbound Traffic)
workflow.add_conditional_edges(
    "classifier",
    supervisor_router,
    {
        "researcher": "researcher",
        "escalator": "escalator",
    },
)

# Automation & Quality Loop
workflow.add_conditional_edges(
    "researcher",
    research_safety_router,
    {"drafter": "drafter", "escalator": "escalator"},
)
workflow.add_edge("drafter", "policy_checker")

workflow.add_conditional_edges(
    "policy_checker", qa_router, {"drafter": "drafter", "archivist": "archivist"}
)

# All terminal paths flow to Archivist to persist authenticated data
workflow.add_edge("escalator", "archivist")

# The Final Finish
workflow.add_edge("archivist", END)

# ==========================================
# 3. COMPILATION & EXPORT
# ==========================================

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Optional: Generate visualization
try:
    png_data = app.get_graph().draw_mermaid_png()
    with open("uda_hub_workflow.png", "wb") as f:
        f.write(png_data)
    print("🎨 Workflow diagram updated: 'uda_hub_workflow.png'")
except Exception:
    print("⚠️  Could not generate diagram (pygraphviz/mermaid issues), skipping...")


if __name__ == "__main__":
    print("\n🚀 Starting UDA-Hub End-to-End Integration Test")
    print("=" * 60)

    # Alice returns to ask for a refund.
    # This should trigger: Concierge (Triage) -> Enricher (ID) -> Historian (Memory) -> Classifier -> Supervisor
    user_input = "I want to know what's included in my CultPass. My email is bob.stone@granite.com"
    initial_state = {
        "messages": [
            HumanMessage(
                # content="I can't attend my Christ the Redeemer booking. Refund?. My email is alice.kingsley@wonderland.com."
                # content="I'd like to know what's included in my current plan. My email is bob.stone@granite.com."
                content=user_input
            )
        ],
        "ticket_text": user_input 
    }

    config = {"configurable": {"thread_id": f"{uuid.uuid4()}"}}

    # Assuming 'app' is your compiled LangGraph
    for event in app.stream(initial_state, config):
        for node, values in event.items():
            print(f"\n📍 NODE: {node}")

            # --- 1. CONCIERGE CHECK (The Front Desk) ---
            if node.lower() == "concierge":
                intent = values.get("triage_intent", "N/A")
                is_identified = values.get("is_identified", False)
                print(f"🏷️  Triage Intent: {intent.upper()}")
                print(f"👤 Identity Status: {'IDENTIFIED' if is_identified else 'NOT IDENTIFIED'}")
                if values.get("ai_response"):
                    print(f"💬 Concierge Says: \"{values['ai_response']}\"")

            # --- 2. IDENTITY & ENRICHMENT ---
            if node.lower() == "enricher" and "user_id" in values:
                user_id = values.get("user_id")
                if user_id:
                    print(f"👤 Identity: {user_id} | Tier: {values.get('customer_tier', 'N/A')}")
                    node_log(
                        "enricher_success",
                        user_id=user_id,
                        tier=values.get("customer_tier"),
                        subscription_status=values.get("subscription_status"),
                    )
                else:
                    print(f"❌ Identity: NOT FOUND")
                    node_log("enricher_failure", reason="user_not_found")

            # --- 3. HISTORIAN CHECK (Durable Memory) ---
            if node.lower() == "historian":
                recurring = values.get("is_recurring", False)
                tid = values.get("similar_ticket_id", "None")
                status = "🚨 RECURRING ISSUE" if recurring else "✨ UNIQUE ISSUE"
                print(f"📚 Memory: {status} | Linked Ticket: {tid}")
                if values.get("history_reasoning"):
                    print(f"🔍 Reason: {values['history_reasoning']}")
                # Already logged in historian.py, but we can add workflow-level context
                node_log(
                    "workflow_historian",
                    is_recurring=recurring,
                    similar_ticket_id=tid,
                )

            # --- 4. CLASSIFIER CHECK (Domain Specialist) ---
            if node.lower() == "classifier":
                category = values.get("category", "N/A")
                urgency = values.get("urgency", "N/A")
                confidence = values.get("confidence_score", 0.0)
                print(f"🏷️  Category: {category} | Urgency: {urgency} | Confidence: {confidence}")
                node_log(
                    "workflow_classifier",
                    category=category,
                    urgency=urgency,
                    confidence=confidence,
                )

            # --- 5. ROUTING DECISION (Supervisor) ---
            if node.lower() == "classifier":
                # Log the routing decision after classifier processes
                route = "researcher" if values.get("category") else "escalator"
                print(f"🚦 Routing to: {route.upper()}")
                node_log("routing_decision", next_node=route)

            # --- 6. RESEARCHER (Knowledge Retrieval) ---
            if node.lower() == "researcher":
                conf = values.get("retrieval_confidence", 0.0)
                kb_matches = values.get("kb_matches", 0)
                res_matches = values.get("reservation_matches", 0)
                status = "✅ SUFFICIENT" if conf >= 0.6 else "🚨 INSUFFICIENT"
                print(f"📊 Retrieval Confidence: {conf} [{status}]")
                print(f"📚 KB Matches: {kb_matches} | Reservation Matches: {res_matches}")
                node_log(
                    "workflow_researcher",
                    retrieval_confidence=conf,
                    kb_matches=kb_matches,
                    reservation_matches=res_matches,
                    confidence_sufficient=(conf >= 0.6),
                )

            # --- 7. POLICY CHECK (Quality Assurance) ---
            if node.lower() == "policy_checker":
                grade = values.get("policy_grade", "N/A")
                feedback = values.get("policy_feedback", "")
                print(f"✔️  Policy Grade: {grade}")
                if feedback:
                    print(f"⚠️  Feedback: {feedback}")
                node_log(
                    "workflow_policy_check",
                    grade=grade,
                    has_feedback=bool(feedback),
                )

            # --- 8. FINAL RESPONSE & ARCHIVE ---
            if "ai_response" in values and node.lower() == "drafter":
                response_preview = values["ai_response"][:100] + "..." if len(values.get("ai_response", "")) > 100 else values.get("ai_response", "")
                print(f"📩 Final Draft: {response_preview}")
                node_log("draft_generated", response_length=len(values.get("ai_response", "")))

            if node.lower() == "archivist":
                archive_summary = values.get("archive_summary", "N/A")
                print(f"🗄️  Archived: {archive_summary}")
                node_log("workflow_archive", archive_summary=archive_summary)

    print("\n" + "=" * 60 + "\n✅ Integration Test Complete")

    # Summarize tool usage evidence captured during the run.
    evidence = get_tool_evidence()
    tool_events = [e for e in evidence if e[0].startswith("TOOL_")]
    print("\n" + "-" * 40)
    print("🔎 Tool Usage Evidence Summary")
    if tool_events:
        for tag, kw in tool_events:
            # Simple inline formatting of key=val pairs
            kvs = []
            for k, v in (kw or {}).items():
                kvs.append(f"{k}={v}")
            print(f"[{tag}] " + " ".join(kvs))
    else:
        print("No tool usage recorded during this run.")
