from typing import Literal
from uda_hub.agentic.tools.udahub_state import UDAHubState


# 2. The Supervisor Routing Function
def supervisor_router(
    state: UDAHubState,
) -> Literal["researcher", "escalator", "clarifier", "greeter", "closer"]:
    """
    Traffic Controller: Routes the user based on intent, identity,
    and the presence of an active ticket anchor.
    """
    category = state.get("category")
    urgency = state.get("urgency")
    user_id = state.get("user_id")
    confidence = state.get("confidence_score", 0.0)
    anchor = state.get("ticket_text")  # The Dynamic Anchor

    # --- THE HIJACK PROTECTION (Priority 1) ---
    # We only go to the Greeter if it's a fresh 'Hello' with NO active ticket.
    if category == "greeting" and not anchor:
        return "greeter"

    # --- THE SECURITY/URGENCY GATE (Priority 2) ---
    # VIPs or low-confidence triage go straight to human help.
    if urgency == "high" or confidence < 0.75:
        return "escalator"

    # --- THE IDENTITY GATE (Priority 3) ---
    # If we have an anchor but NO user_id, we MUST stay in the clarifier loop,
    # even if the classifier mislabeled this turn as a 'greeting'.
    requires_auth = ["billing", "account_management", "technical_issue"]

    if not user_id:
        # If it's a sensitive category OR we are already tracking an anchor
        if category in requires_auth or anchor:
            return "clarifier"
        
    # If the user is satisfied, go to the closer
    if category == "satisfaction":
        return "closer"

    # --- THE AUTOMATION PATH (Default) ---
    # We have an anchor, we have a user_id, and it's safe to automate.
    return "researcher"


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
