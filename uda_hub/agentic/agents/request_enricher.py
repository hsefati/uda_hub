# Assuming you have sqlalchemy or sqlite3 set up to query your DBs
import sqlite3

from uda_hub.agentic.tools.udahub_state import UDAHubState


def data_enricher_node(state: UDAHubState) -> dict:
    """
    Step 1 of LangGraph: No LLMs used here.
    Queries the databases using the email provided in the webhook payload.
    """
    email = state.get("user_email")
    if not email:
        # If no email is provided, pass state as-is to let the Clarification Agent handle it later
        return {"customer_tier": "unknown", "subscription_status": "unknown"}

    # 1. Connect to Client DB (cultpass.db)
    conn = sqlite3.connect("uda_hub/data/external/cultpass.db")
    cursor = conn.cursor()

    # Fetch User ID & Tier
    cursor.execute(
        """
        SELECT u.user_id, s.tier, s.status 
        FROM users u 
        LEFT JOIN subscriptions s ON u.user_id = s.user_id 
        WHERE u.email = ?
    """,
        (email,),
    )

    result = cursor.fetchone()
    conn.close()

    if result:
        user_id, tier, status = result
        return {
            "user_id": user_id,
            "customer_tier": tier,  # e.g., "premium", "basic"
            "subscription_status": status,  # e.g., "active", "cancelled"
        }
    else:
        return {"customer_tier": "unregistered", "subscription_status": "none"}


if __name__ == "__main__":
    print("--- Running Data Enricher Tests ---\n")

    # Test 1: An Active Premium User (Frank Ocean)
    state_frank: UDAHubState = {
        "ticket_text": "I want to book a new experience.",
        "user_email": "frank.ocean@seawaves.io",
        "user_id": None,
        "customer_tier": None,
        "subscription_status": None,
        "category": None,
        "urgency": None,
    }
    print("Test 1 (Frank Ocean):")
    print(data_enricher_node(state_frank))
    # Expected: {'user_id': 'e6376d', 'customer_tier': 'premium', 'subscription_status': 'active'}
    print("-" * 40)

    # Test 2: A Cancelled Basic User (Alice Kingsley)
    state_alice: UDAHubState = {
        "ticket_text": "I can't log in to my Cultpass account.",
        "user_email": "alice.kingsley@wonderland.com",
        "user_id": None,
        "customer_tier": None,
        "subscription_status": None,
        "category": None,
        "urgency": None,
    }
    print("Test 2 (Alice Kingsley):")
    print(data_enricher_node(state_alice))
    # Expected: {'user_id': 'a4ab87', 'customer_tier': 'basic', 'subscription_status': 'cancelled'}
    print("-" * 40)

    # Test 3: An Unregistered/Unknown Email
    state_unknown: UDAHubState = {
        "ticket_text": "Do you offer student discounts?",
        "user_email": "not_a_user@example.com",
        "user_id": None,
        "customer_tier": None,
        "subscription_status": None,
        "category": None,
        "urgency": None,
    }
    print("Test 3 (Unknown Email):")
    print(data_enricher_node(state_unknown))
    # Expected: {'customer_tier': 'unregistered', 'subscription_status': 'none'}
    print("-" * 40)

    # Test 4: Missing Email entirely
    state_missing: UDAHubState = {
        "ticket_text": "Help me!",
        "user_email": None,
        "user_id": None,
        "customer_tier": None,
        "subscription_status": None,
        "category": None,
        "urgency": None,
    }
    print("Test 4 (Missing Email):")
    print(data_enricher_node(state_missing))
    # Expected: {'customer_tier': 'unknown', 'subscription_status': 'unknown'}
