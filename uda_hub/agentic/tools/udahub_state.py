from typing import TypedDict, Optional, List, Annotated
from langgraph.graph.message import add_messages


class UDAHubState(TypedDict):
    # --- Conversation State (Short-Term Memory) ---
    messages: Annotated[List, add_messages]
    latest_input: Optional[str]
    ticket_text: Optional[str]

    # --- Identity & Enrichment ---
    user_email: Optional[str]
    user_id: Optional[str]
    customer_tier: Optional[str]
    subscription_status: Optional[str]

    # --- Durable Context (Long-Term Memory) ---
    # user_history: Fetched via lookup_user_history tool
    user_history: Optional[str]
    # is_recurring: Semantic flag for duplicate/repeat issues
    is_recurring: Optional[bool]
    # history_reasoning: Explanation of why it's recurring or unique
    history_reasoning: Optional[str]
    # suggested_history_action: Operational advice for the Supervisor
    suggested_history_action: Optional[str]
    # similar_ticket_id: ID of a similar previously resolved ticket
    similar_ticket_id: Optional[str]

    # --- Classification & Triage ---
    category: Optional[str]
    urgency: Optional[str]
    confidence_score: Optional[float]

    # --- Knowledge Retrieval & Grounding ---
    research_facts: Optional[str]
    retrieval_confidence: Optional[float]
    needs_escalation: Optional[bool]

    # --- Generation & Quality Assurance ---
    ai_response: Optional[str]
    policy_grade: Optional[str]
    policy_feedback: Optional[str]

    # --- Archivist & Handoff ---
    archive_summary: Optional[str]
    handoff_summary: Optional[str]
    status: Optional[str]  # 'in_progress', 'pending_human', 'closed'
