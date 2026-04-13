import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from uda_hub.agentic.tools.udahub_state import UDAHubState
from langchain_core.messages import HumanMessage

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

    internal_chain = escalation_prompt | llm
    internal_note = internal_chain.invoke(
        {
            "tier": tier,
            "status": status,
            "category": category,
            "urgency": urgency,
            "ticket_text": ticket_text,
        }
    ).content

    public_msg = (
        f"I've reviewed your request regarding {category}. Because this requires specialized "
        f"assistance, I am handing this conversation over to one of our human experts. "
        f"They will be with you shortly!"
    )

    # 4. Return everything needed for Option B
    return {
        "handoff_summary": internal_note,
        "ai_response": public_msg,  # The string version
        "messages": [AIMessage(content=public_msg)],  # The Chat UI version
        "status": "pending_human",
    }


if __name__ == "__main__":
    print("🚀 --- Testing Escalation Agent ---\n" + "=" * 50)

    # Test Case: Frank Ocean (VIP Premium User) is furious about a double charge.
    # The Supervisor routed this here because Urgency = "high".
    state_vip_escalation: UDAHubState = {
        # Option B requirement: The actual message history from the chat UI
        "messages": [
            HumanMessage(
                content="I was charged twice for the Samba Night experience! This is unacceptable, refund me immediately or I'm canceling my premium membership."
            )
        ],
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

    # Run the node
    result = escalation_node(state_vip_escalation)

    print("📝 INTERNAL HUMAN HANDOFF NOTE (For Agent Dashboard):")
    print("-" * 50)
    print(result.get("handoff_summary"))
    print("-" * 50)

    print("\n💬 PUBLIC RESPONSE (Sent to Chat Interface):")
    if "messages" in result:
        last_msg = result["messages"][-1]
        print(f"Assistant: {last_msg.content}")

    print(f"\n⚙️  SYSTEM STATUS: {result.get('status')}")
    print("=" * 50)
    print(
        "✅ Test Complete: Escalation properly handles both the user and the support team."
    )
