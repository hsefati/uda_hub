from uda_hub.agentic.tools.udahub_state import UDAHubState


# 2. The Supervisor Routing Function
def supervisor_router(state: UDAHubState) -> str:
    """
    Evaluates the State and returns the name of the next node in the graph.
    This acts as LangGraph's conditional edge.
    """
    category = state.get("category")
    urgency = state.get("urgency")
    confidence = state.get("confidence_score", 0.0)
    user_id = state.get("user_id")

    # RULE 1: Missing Required Data -> Clarification Agent
    # If it's a billing or account issue, we MUST know who the user is.
    # (If it's just a "general_inquiry" like "What are your hours?", we don't need their ID).
    requires_auth = ["billing", "account_management", "technical_issue"]
    if category in requires_auth and not user_id:
        return "clarification"

    # RULE 2: High Risk or Low Confidence -> Escalation Agent
    # We don't want the AI handling angry customers or guessing if it's unsure.
    if urgency == "high" or confidence < 0.75:
        return "escalation"

    # RULE 3: Safe to Automate -> Researcher Agent
    return "researcher"


# ==========================================
# 3. ISOLATED ROUTING TESTS
# ==========================================
if __name__ == "__main__":
    print("--- Testing Supervisor Logic ---\n")

    # Test 1: High Confidence, VIP Customer (Should Escalate because VIPs trigger High Urgency)
    state_vip: UDAHubState = {
        "category": "billing", "urgency": "high", "confidence_score": 0.95,
        "user_id": "user_123", # Data is present
        # ... other fields omitted for brevity
    }
    print(f"VIP Billing Issue -> Routes to: {supervisor_router(state_vip)}")
    # Expected: escalation

    # Test 2: Standard Question, but Missing User ID (Should go to Clarification)
    state_missing_auth: UDAHubState = {
        "category": "account_management", "urgency": "standard", "confidence_score": 0.90,
        "user_id": None, # Missing!
    }
    print(f"Account Issue (No ID) -> Routes to: {supervisor_router(state_missing_auth)}")
    # Expected: clarification

    # Test 3: General Question, Missing User ID (Should go to Researcher)
    # General inquiries don't require an ID.
    state_general: UDAHubState = {
        "category": "general_inquiry", "urgency": "low", "confidence_score": 0.85,
        "user_id": None, 
    }
    print(f"General Question (No ID) -> Routes to: {supervisor_router(state_general)}")
    # Expected: researcher
    
    # Test 4: Perfect Automation Candidate
    state_perfect: UDAHubState = {
        "category": "technical_issue", "urgency": "standard", "confidence_score": 0.92,
        "user_id": "user_999", 
    }
    print(f"Standard Tech Issue -> Routes to: {supervisor_router(state_perfect)}")
    # Expected: researcher