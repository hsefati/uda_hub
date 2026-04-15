from typing import TypedDict, Optional, List, Annotated
from langgraph.graph.message import add_messages

class UDAHubState(TypedDict):
    # --- Input Handling & Dynamic Anchoring ---
    # messages is the "Source of Truth" for the conversation history
    # Annotated[..., add_messages] tells LangGraph to APPEND new messages rather than overwrite
    messages: Annotated[List, add_messages] 
    
    # latest_input stores ONLY what the user said in the current turn
    latest_input: Optional[str]
    
    # ticket_text is the "Anchor" - once set, it shouldn't change
    ticket_text: Optional[str]
    
    # --- Identity & Enrichment ---
    user_email: Optional[str]
    user_id: Optional[str]
    customer_tier: Optional[str]
    subscription_status: Optional[str]
    
    # --- Classification ---
    category: Optional[str]  # e.g., 'greeting', 'billing', 'technical'
    urgency: Optional[str]
    confidence_score: Optional[float]
    
    # --- Processing ---
    research_facts: Optional[str]
    ai_response: Optional[str]
    
    # --- Quality Assurance ---
    policy_grade: Optional[str]    # "PASS" or "FAIL"
    policy_feedback: Optional[str]
    
    # --- Output & Handoff ---
    archive_summary: Optional[str] # For the database
    handoff_summary: Optional[str] # For the human agent (Escalation)
    status: Optional[str]          # 'in_progress', 'pending_human', 'pending_user'