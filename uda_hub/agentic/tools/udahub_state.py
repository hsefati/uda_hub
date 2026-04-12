from typing import Optional, TypedDict


class UDAHubState(TypedDict):
    ticket_text: str
    user_email: Optional[str]
    # Fields populated by the Enricher
    user_id: Optional[str]
    customer_tier: Optional[str]
    subscription_status: Optional[str]
    # Fields populated by AI later
    category: Optional[str]
    urgency: Optional[str]
