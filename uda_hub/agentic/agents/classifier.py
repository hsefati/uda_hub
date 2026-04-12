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
        description="Categories: billing, technical_issue, account_management, general_inquiry",
        json_schema_extra=[
            "billing",
            "technical_issue",
            "account_management",
            "general_inquiry",
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
    Reads the state (including pre-injected user context), classifies the ticket,
    and returns the updated fields for the State.
    """
    # Extract data from the current state
    ticket_text = state.get("ticket_text", "")
    customer_tier = state.get("customer_tier", "regular")

    routing_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Triage Classifier for UDA-Hub. "
                "Determine the category, urgency, and your confidence score. "
                "CRITICAL RULE: If the Customer Tier is 'vip', the urgency MUST be 'high' "
                "regardless of the ticket content.",
            ),
            (
                "human",
                "Customer Tier: {customer_tier}\n\nTicket details:\n{ticket_text}",
            ),
        ]
    )

    classifier_llm = llm.with_structured_output(TicketClassification)
    chain = routing_prompt | classifier_llm

    # Pass BOTH the text and the injected context to the LLM
    response = chain.invoke(
        {"ticket_text": ticket_text, "customer_tier": customer_tier}
    )

    # Return ONLY the state variables we are updating
    return response.model_dump()


# --- Test the Context-Aware Node ---
if __name__ == "__main__":
    # Test 1: A regular user asking a simple question
    state_regular: UDAHubState = {
        "ticket_text": "Can you tell me how to update my profile picture?",
        "user_id": "user_123",
        "customer_tier": "regular",
        "category": None,
        "urgency": None,
        "confidence_score": None,
    }

    # Test 2: A VIP user asking the EXACT same question
    state_vip: UDAHubState = {
        "ticket_text": "Can you tell me how to update my profile picture?",
        "user_id": "user_999",
        "customer_tier": "vip",
        "category": None,
        "urgency": None,
        "confidence_score": None,
    }

    print("Regular User Result:", classifier_node(state_regular))
    # Expects: urgency -> 'low' or 'standard'

    print("VIP User Result:    ", classifier_node(state_vip))
    # Expects: urgency -> 'high' (because the prompt knows they are VIP)
