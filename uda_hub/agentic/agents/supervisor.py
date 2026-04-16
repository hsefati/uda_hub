from typing import Literal
from uda_hub.agentic.tools.udahub_state import UDAHubState
from uda_hub.agentic.tools.logging import node_log


# 2. The Supervisor Routing Function
def supervisor_router(
    state: UDAHubState,
) -> Literal["researcher", "escalator", "greeter", "closer"]:
    """
    Traffic Controller: Routes the user based on intent, identity,
    Durable History (Long-Term Memory), and active ticket anchors.
    """
    category = state.get("category")
    urgency = state.get("urgency")
    user_id = state.get("user_id")
    confidence = state.get("confidence_score", 0.0)
    anchor = state.get("ticket_text")

    # NEW: Long-Term Memory Fields
    suggested_action = state.get("suggested_history_action")
    is_recurring = state.get("is_recurring", False)

    # --- THE HIJACK PROTECTION (Priority 1) ---
    if category == "greeting" and not anchor:
        route = "greeter"
        node_log("supervisor", route=route, reason="greeting_no_anchor")
        return route

    # --- THE SECURITY/URGENCY GATE (Priority 2) ---
    if urgency == "high" or confidence < 0.75:
        route = "escalator"
        node_log("supervisor", route=route, reason="urgency_or_low_confidence")
        return route

    # --- THE HISTORY GATE (Priority 3) - NEW ---
    # If the Historian detected a similar ticket in the durable database,
    # we follow the 'Gatekeeper' recommendation before proceeding.
    if suggested_action == "escalate":
        print("🚨 History Check: High-risk repeat issue. Forcing Escalation.")
        route = "escalator"
        node_log("supervisor", route=route, reason="history_escalate")
        return route

    # --- THE SATISFACTION GATE (Priority 5) ---
    if category == "satisfaction":
        route = "closer"
        node_log("supervisor", route=route, reason="satisfaction_closure")
        return route

    # --- THE AUTOMATION PATH (Default) ---
    # We have an anchor, an ID, and no history blockers.
    route = "researcher"
    node_log("supervisor", route=route, reason="automation_default")
    return route


if __name__ == "__main__":
    print("🚀 --- Testing Updated Supervisor Logic ---\n" + "=" * 50)

    # Test 1: High Confidence, VIP Customer
    # VIPs trigger 'high' urgency in the Classifier, so they should be fast-tracked.
    state_vip: UDAHubState = {
        "category": "billing",
        "urgency": "high",
        "confidence_score": 0.95,
        "user_id": "user_123",
    }
    print(f"VIP Billing Issue      -> Routes to: {supervisor_router(state_vip)}")
    # Expected: escalator

    # Test 2: Standard Question, but Missing User ID
    # Billing/Account issues MUST have an ID.
    state_missing_auth: UDAHubState = {
        "category": "account_management",
        "urgency": "standard",
        "confidence_score": 0.90,
        "user_id": None,
    }
    print(
        f"Account Issue (No ID)  -> Routes to: {supervisor_router(state_missing_auth)}"
    )
    # Expected: clarifier

    # Test 3: Simple Greeting
    # New category! Should bypass research/clarification entirely.
    state_greet: UDAHubState = {
        "category": "greeting",
        "urgency": "low",
        "confidence_score": 0.99,
        "user_id": None,
    }
    print(f"Simple 'Hello'         -> Routes to: {supervisor_router(state_greet)}")
    # Expected: greeter

    # Test 4: General Question, Missing User ID
    # General inquiries (e.g., policy questions) don't need a specific user_id to research.
    state_general: UDAHubState = {
        "category": "general_inquiry",
        "urgency": "low",
        "confidence_score": 0.85,
        "user_id": None,
    }
    print(f"General Query (No ID)  -> Routes to: {supervisor_router(state_general)}")
    # Expected: researcher

    # Test 5: Perfect Automation Candidate
    state_perfect: UDAHubState = {
        "category": "technical_issue",
        "urgency": "standard",
        "confidence_score": 0.92,
        "user_id": "user_999",
    }
    print(f"Standard Tech Issue    -> Routes to: {supervisor_router(state_perfect)}")
    # Expected: researcher

    print("\n" + "=" * 50 + "\n✅ Supervisor Routing Tests Complete")
