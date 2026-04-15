import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from uda_hub.agentic.tools.udahub_state import UDAHubState
from langchain_core.messages import HumanMessage
from uda_hub.agentic.tools.logging import node_log

load_dotenv()

# Initialize the LLM (Using gpt-4o-mini is perfect for summarization)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,  # Keep at 0 for factual, non-creative summaries
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)


def escalation_node(state: UDAHubState) -> dict:
    """
    Refined Escalation Node: Now includes Knowledge Retrieval confidence
    to explain exactly why the AI is handing over the ticket.
    """
    # 1. Extract existing metadata
    ticket_text = state.get("ticket_text", "No text provided.")
    tier = state.get("customer_tier", "Unknown")
    status = state.get("subscription_status", "Unknown")
    category = state.get("category", "Unknown")
    urgency = state.get("urgency", "Unknown")

    # 2. Extract the NEW Research Metrics
    retrieval_conf = state.get("retrieval_confidence", 0.0)
    research_facts = state.get("research_facts", "No research attempted.")

    escalation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Escalation Coordinator for UDA-Hub. "
                "Your job is to provide a concise internal note for a human agent.\n\n"
                "CRITICAL: If the 'Retrieval Confidence' is low (below 0.6), explicitly state "
                "that the AI could not find a matching policy in the Knowledge Base.\n\n"
                "Format your response exactly like this:\n"
                "**Reason for Escalation:** [Urgency/VIP or Knowledge Retrieval Failure]\n"
                "**Retrieval Confidence:** [0.0 - 1.0]\n"
                "**Customer Context:** [Tier & Status]\n"
                "**Internal Research Context:** [Briefly what was found or not found]\n"
                "**Issue Summary:** [What does the user need?]",
            ),
            (
                "human",
                "Metadata: Tier={tier}, Status={status}, Category={category}, Urgency={urgency}, Confidence={conf}\n"
                "Research Findings: {facts}\n"
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
            "conf": retrieval_conf,
            "facts": research_facts,
            "ticket_text": ticket_text,
        }
    ).content

    public_msg = (
        f"I've reviewed your request regarding {category}. Because this requires specialized "
        f"assistance to ensure everything is handled correctly, I am handing this over to "
        f"one of our human experts. They will be with you shortly!"
    )

    out = {
        "handoff_summary": internal_note,
        "ai_response": public_msg,
        "messages": [AIMessage(content=public_msg)],
        "status": "pending_human",
    }

    node_log(
        "escalator",
        retrieval_confidence=retrieval_conf,
        handoff_summary=internal_note[:120] if internal_note else None,
        status=out.get("status"),
    )

    return out



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
