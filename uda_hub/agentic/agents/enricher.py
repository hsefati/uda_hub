# Assuming you have sqlalchemy or sqlite3 set up to query your DBs
import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from uda_hub.agentic.tools.udahub_state import UDAHubState
from uda_hub.agentic.tools.logging import node_log

from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

# Dynamically resolve the project root and database path
def _get_db_path():
    """
    Resolve the cultpass.db path regardless of where the script is run from.
    Works from notebook, workflow.py, or direct script execution.
    """
    # Get the directory of this file (enricher.py)
    current_file = Path(__file__).resolve()
    # Navigate up: enricher.py -> agents -> agentic -> uda_hub -> project_root
    project_root = current_file.parent.parent.parent.parent
    db_path = project_root / "uda_hub" / "data" / "external" / "cultpass.db"
    return str(db_path)


def lookup_user_in_cultpass(email: str = None, user_id: str = None, name: str = None):
    """
    Searches the CultPass database for a user's subscription and identity details.
    You can provide an email, a specific user_id, or a full name.
    """
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    result = None

    try:
        # 1. Primary Search: Email or Alphanumeric ID
        if email or user_id:
            id_query = """
                SELECT u.user_id, u.email, u.full_name, s.tier, s.status 
                FROM users u 
                LEFT JOIN subscriptions s ON u.user_id = s.user_id 
                WHERE u.email = ? OR u.user_id = ?
            """
            cursor.execute(id_query, (email, user_id))
            result = cursor.fetchone()

        # 2. Secondary Search: Fuzzy Name Match
        if not result and name:
            name_query = """
                SELECT u.user_id, u.email, u.full_name, s.tier, s.status 
                FROM users u 
                LEFT JOIN subscriptions s ON u.user_id = s.user_id 
                WHERE u.full_name LIKE ?
            """
            cursor.execute(name_query, (f"%{name}%",))
            result = cursor.fetchone()

        if result:
            uid, umail, uname, utier, ustatus = result
            return {
                "found": True,
                "user_id": uid,
                "user_email": umail,
                "full_name": uname,
                "customer_tier": utier or "regular",
                "subscription_status": ustatus or "active",
            }

        return {"found": False, "error": "No user matched the provided details."}

    finally:
        conn.close()


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)


class MinedIdentifiers(BaseModel):
    email: Optional[str] = Field(
        None, description="An email address found in the text."
    )
    user_id: Optional[str] = Field(
        None, description="A specific alphanumeric user ID like 'e6376d'."
    )
    name: Optional[str] = Field(None, description="The user's full or first name.")


def data_enricher_node(state: UDAHubState) -> dict:
    """
    Mines identifiers from history and uses the lookup tool to enrich the state.
    """
    latest_input = state.get("latest_input", "")
    history = state.get("messages", [])[-3:] # Use history to catch Turn 1 info

    # --- STEP 1: AI Extraction (Mining) ---
    mining_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract user identification (email, membership_id, or name) from the history."),
        *history,
        ("human", "{latest_input}")
    ])
    
    mining_llm = llm.with_structured_output(MinedIdentifiers)
    mined: MinedIdentifiers = (mining_prompt | mining_llm).invoke({"latest_input": latest_input})

    # --- STEP 2: Tool Execution ---
    # We call our new tool directly as a helper function
    user_data = lookup_user_in_cultpass(
        email=(mined.email or state.get("user_email")),
        user_id=(mined.user_id or state.get("user_id")),
        name=(mined.name or state.get("full_name"))
    )

    # --- STEP 3: State Update ---
    if user_data.get("found"):
        # We found them! Populate the state with verified DB values
        out = {
            "user_id": user_data["user_id"],
            "user_email": user_data["user_email"],
            "full_name": user_data["full_name"],
            "customer_tier": user_data["customer_tier"],
            "subscription_status": user_data["subscription_status"]
        }
        node_log(
            "enricher",
            user_id=out.get("user_id"),
            user_email=out.get("user_email"),
            customer_tier=out.get("customer_tier"),
        )
        return out

    # If not found, we still save any mined info (like the email) 
    # so the Clarifier knows what the user tried to provide.
    out = {
        "user_email": mined.email or state.get("user_email"),
        "full_name": mined.name or state.get("full_name"),
        "customer_tier": "unknown"
    }
    node_log(
        "enricher",
        user_email=out.get("user_email"),
        full_name=out.get("full_name"),
        customer_tier=out.get("customer_tier"),
    )
    return out


if __name__ == "__main__":
    print(
        "🚀 --- Running Improved Data Enricher Tests (Conversational) ---\n" + "=" * 60
    )

    # Test 1: Conversational Email Extraction (Frank Ocean)
    # Testing if the AI can pull the email out of a noisy sentence.
    state_frank: UDAHubState = {
        "latest_input": "Hey, I'm Frank and my email is frank.ocean@seawaves.io. I want to book a trip.",
        "ticket_text": None,
        "user_email": None,  # Starting with no explicit email
        "user_id": None,
    }
    print("Test 1 (Conversational Email - Frank):")
    print(data_enricher_node(state_frank))
    # Expected: Finds Frank via email mining -> returns premium tier/active status
    print("-" * 60)

    # Test 2: Name-Based Lookup (Alice Kingsley)
    # Testing the fallback to searching by name if no email is found.
    state_alice: UDAHubState = {
        "latest_input": "My name is Alice Kingsley and I can't log in.",
        "ticket_text": None,
        "user_email": None,
        "user_id": None,
    }
    print("Test 2 (Name Lookup - Alice):")
    print(data_enricher_node(state_alice))
    # Expected: Mines name 'Alice Kingsley' -> finds user_id 'a4ab87' -> basic tier
    print("-" * 60)

    # Test 3: User ID Extraction
    # Testing if it can find a user when they provide their ID directly in chat.
    state_id_search: UDAHubState = {
        "latest_input": "My membership ID is e6376d. Help me with a refund.",
        "ticket_text": None,
        "user_email": None,
        "user_id": None,
    }
    print("Test 3 (ID Extraction):")
    print(data_enricher_node(state_id_search))
    # Expected: Mines ID 'e6376d' -> returns Frank Ocean's profile
    print("-" * 60)

    # Test 4: Unknown/Garbage Input
    state_unknown: UDAHubState = {
        "latest_input": "I am a new user, my name is Batman and I live in Gotham.",
        "ticket_text": None,
        "user_email": None,
        "user_id": None,
    }
    print("Test 4 (Unknown User):")
    print(data_enricher_node(state_unknown))
    # Expected: No match in DB -> returns {'customer_tier': 'unknown'}
    print("-" * 60)

    # Test 5: Pure Greeting (No Identifiers)
    state_greeting: UDAHubState = {
        "latest_input": "Hello! Are you a robot?",
        "ticket_text": None,
        "user_email": None,
        "user_id": None,
    }
    print("Test 5 (Greeting - No Identifiers):")
    print(data_enricher_node(state_greeting))
    # Expected: No data to mine -> returns {'customer_tier': 'unknown'}

    print("\n" + "=" * 60 + "\n✅ Enricher Test Suite Complete")
