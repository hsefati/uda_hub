import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from uda_hub.agentic.tools.udahub_state import UDAHubState

load_dotenv()

# Initialize the LLM (Using gpt-4o-mini is perfect for summarization)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,  # Keep at 0 for factual, non-creative summaries
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)


# The Escalation Agent Node
def escalation_node(state: UDAHubState) -> dict:
    """
    Reads the full context of the failed or high-risk ticket and packages it
    into a concise internal note for a human support agent.
    """
    # Extract relevant context from the State
    ticket_text = state.get("ticket_text", "No text provided.")
    tier = state.get("customer_tier", "Unknown")
    status = state.get("subscription_status", "Unknown")
    category = state.get("category", "Unknown")
    urgency = state.get("urgency", "Unknown")

    escalation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Escalation Coordinator for CultPass. "
                "Your job is to read a customer ticket and the system metadata, and write a "
                "brief, bulleted internal summary for the human support agent who will take over. "
                "DO NOT respond to the customer. Write an internal note.\n\n"
                "Format your response exactly like this:\n"
                "**Reason for Escalation:** [Why this needs a human]\n"
                "**Customer Context:** [Tier & Status]\n"
                "**Issue Summary:** [1-2 sentences on what they want]",
            ),
            (
                "human",
                "Metadata: Tier={tier}, Status={status}, Category={category}, Urgency={urgency}\n\n"
                "Customer Ticket: {ticket_text}",
            ),
        ]
    )

    chain = escalation_prompt | llm

    response = chain.invoke(
        {
            "tier": tier,
            "status": status,
            "category": category,
            "urgency": urgency,
            "ticket_text": ticket_text,
        }
    )

    # Return the drafted summary and change the graph status to pending_human
    return {"handoff_summary": response.content, "status": "pending_human"}


if __name__ == "__main__":
    print("--- Testing Escalation Agent ---\n")

    # Test Case: Frank Ocean (VIP Premium User) is furious about a double charge.
    # The Supervisor routed this here because Urgency = "high".
    state_vip_escalation: UDAHubState = {
        "ticket_text": "I was charged twice for the Samba Night experience! This is unacceptable, refund me immediately or I'm canceling my premium membership.",
        "user_email": "frank.ocean@seawaves.io",
        "user_id": "e6376d",
        "customer_tier": "premium",
        "subscription_status": "active",
        "category": "billing",
        "urgency": "high",
        "confidence_score": 0.88,
        "ai_response": None,
        "handoff_summary": None,
        "status": "in_progress",
    }

    result = escalation_node(state_vip_escalation)

    print("INTERNAL HUMAN HANDOFF NOTE GENERATED:\n")
    print(result["handoff_summary"])
    print(f"\n[System Status Changed To: {result['status']}]")
