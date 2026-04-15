import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from uda_hub.agentic.tools.udahub_state import UDAHubState

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)


# 2. Define the Output Schema (Unchanged)
class TicketClassification(BaseModel):
    category: str = Field(
        ...,
        description="Categories: billing, technical_issue, account_management, general_inquiry, greeting, closure",
        json_schema_extra=[
            "billing",
            "technical_issue",
            "account_management",
            "general_inquiry",
            "greeting",
        ],
    )
    urgency: str = Field(
        ...,
        description="Urgency: low, standard, high. Note: VIP customers always default to high urgency.",
        json_schema_extra=["low", "standard", "high"],
    )
    confidence_score: float = Field(..., description="Score between 0.0 and 1.0")


# 3. The LangGraph Classifier Node
def classifier_node(state: UDAHubState) -> dict:
    """
    Classifies the current message and decides if it should be 'Anchored'
    as the main ticket_text.
    """
    # 1. Look at what was JUST said (latest_input) vs the existing anchor
    latest_input = state.get("latest_input", "")
    existing_ticket = state.get("ticket_text")
    customer_tier = state.get("customer_tier", "regular")
    # Take the last 3-4 messages for enough context without bloat
    history = state.get("messages", [])[-4:]

    routing_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Triage Classifier for UDA-Hub. Your goal is to determine the intent "
                "of the LATEST INPUT while considering the CONVERSATION HISTORY.\n\n"
                "Categories:\n"
                "- 'greeting': Only for fresh hellos with NO prior context.\n"
                "- 'closure': Indication that the user wants to end the conversation or the problem is now resolved.\n"
                "- 'billing': Money, refunds, or payment issues.\n"
                "- 'technical_issue': Site bugs or login errors.\n"
                "- 'account_management': Providing identification (email, name, ID) or profile updates.\n"
                "- 'general_inquiry': General questions.\n\n"
                "CRITICAL LOGIC:\n"
                "1. If the HISTORY shows the AI asked for ID and the LATEST INPUT provides a name or email, "
                "you MUST classify this as 'account_management', NOT 'greeting'.\n"
                "2. If an 'existing_ticket' is provided, prioritize keeping the context relevant to that issue.",
            ),
            # Pass the history list directly into the prompt
            *history,
            (
                "human",
                f"Existing Ticket Anchor: {existing_ticket}\n"
                f"Customer Tier: {customer_tier}\n"
                f"LATEST INPUT: {latest_input}",
            ),
        ]
    )

    classifier_llm = llm.with_structured_output(TicketClassification)
    chain = routing_prompt | classifier_llm

    # Classify what the user JUST said
    response = chain.invoke(
        {
            "latest_input": latest_input,
            "customer_tier": customer_tier,
            "existing_ticket": existing_ticket or "None (New Conversation)",
        }
    )

    updates = response.model_dump()

    # --- DYNAMIC ANCHORING LOGIC ---
    if not existing_ticket and updates["category"] != "greeting":
        updates["ticket_text"] = latest_input
    else:
        updates["ticket_text"] = existing_ticket

    return updates


if __name__ == "__main__":
    print("🚀 Testing Classifier with Dynamic Anchoring\n" + "=" * 50)

    # Test 1: A greeting (Should NOT anchor the ticket)
    state_greeting: UDAHubState = {
        "latest_input": "Hi there! Just wanted to say hello.",
        "ticket_text": None,  # Starting fresh
        "customer_tier": "regular",
    }

    # Test 2: A VIP user asking a question (Urgency should be HIGH)
    state_vip: UDAHubState = {
        "latest_input": "I can't log in to my account.",
        "ticket_text": None,  # Starting fresh
        "customer_tier": "vip",
    }

    # Test 3: Regular user with a problem (Should anchor the ticket)
    state_regular: UDAHubState = {
        "latest_input": "I need a refund for my canceled booking.",
        "ticket_text": None,  # Starting fresh
        "customer_tier": "regular",
    }

    # Test 4: A follow-up message (Should NOT overwrite the existing anchor)
    state_followup: UDAHubState = {
        "latest_input": "my email is alice@wonderland.com",
        "ticket_text": "I need a refund for my canceled booking.",  # Anchor already set!
        "customer_tier": "regular",
    }

    print("\n--- TEST 1: Greeting ---")
    res1 = classifier_node(state_greeting)
    print(
        f"Category: {res1.get('category')} | Anchored Ticket: {res1.get('ticket_text')}"
    )

    print("\n--- TEST 2: VIP Support Request ---")
    res2 = classifier_node(state_vip)
    print(
        f"Category: {res2.get('category')} | Urgency: {res2.get('urgency')} | Anchored Ticket: {res2.get('ticket_text')}"
    )

    print("\n--- TEST 3: Regular Support Request ---")
    res3 = classifier_node(state_regular)
    print(
        f"Category: {res3.get('category')} | Anchored Ticket: {res3.get('ticket_text')}"
    )

    print("\n--- TEST 4: Follow-up Response ---")
    res4 = classifier_node(state_followup)
    # Crucial: 'ticket_text' should NOT be in res4 if it didn't change,
    # or it should remain the original refund string.
    print(f"Input: {state_followup['latest_input']}")
    print(
        f"Category: {res4.get('category')} | Anchored Ticket: {res4.get('ticket_text', 'STAYED THE SAME')}"
    )

    print("\n" + "=" * 50)
