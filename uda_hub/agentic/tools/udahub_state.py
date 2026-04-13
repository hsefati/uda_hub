from typing import TypedDict, Optional, List

class UDAHubState(TypedDict):
    # --- Initial Inputs ---
    ticket_text: str
    user_email: Optional[str]
    
    # --- Enricher & Classifier Outputs ---
    user_id: Optional[str]
    customer_tier: Optional[str]
    subscription_status: Optional[str]
    category: Optional[str]
    urgency: Optional[str]
    confidence_score: Optional[float]
    
    # --- Researcher Output ---
    research_facts: Optional[str]  # <--- MUST HAVE THIS
    
    # --- Drafter & Policy Checker Outputs ---
    ai_response: Optional[str]
    policy_grade: Optional[str]    # "PASS" or "FAIL"
    policy_feedback: Optional[str]
    
    # --- Archivist Output ---
    archive_summary: Optional[str]
    
    # --- Internal Graph Management ---
    # messages is used by the create_react_agent instances internally
    messages: List[str]