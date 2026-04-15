import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain.agents import create_agent
from uda_hub.agentic.tools.udahub_state import UDAHubState
from langchain_core.messages import HumanMessage
from uda_hub.agentic.tools.logging import node_log

load_dotenv()


def create_agent_wrapper(name: str, model, tools: list, system_prompt: str):
    """
    Returns a compiled LangGraph ReAct agent.
    """
    return create_agent(
        model=model,
        tools=tools,  # This will be empty for the Drafter
        system_prompt=system_prompt,
        name=name,
    )


llm = ChatOpenAI(
    model="gpt-4o",  # You could also use gpt-4o-mini here to save money, as drafting is easy
    temperature=0.4,  # Slightly higher temperature allows for more natural, empathetic writing
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)

drafter_agent = create_agent_wrapper(
    name="drafter_agent",
    model=llm,
    tools=[],  # NO TOOLS! The Drafter is not allowed to search.
    system_prompt=(
        "You are the UDA-Hub Drafter Agent. Your mission is to write a final, polished response to the customer.\n\n"
        "STRICT PROTOCOLS:\n"
        "1. ANCHORING: Your subject is the 'Anchored Ticket'. Do not get distracted by follow-up chat history.\n"
        "2. EVIDENCE-BASED: Base your response ENTIRELY on the provided 'Research Facts'.\n"
        "   - If the Researcher found a specific policy (e.g., non-refundable), cite it gently[cite: 227, 502].\n"
        "   - If a specific reservation was found (or not found), mention it clearly[cite: 730, 919].\n"
        "3. TONE: Be empathetic. Use the 'Customer Tier' to adjust formality (Concierge style for VIPs).\n"
        "4. PERSONALIZATION: If a user name is provided, address them by name.\n"
        "5. STATUS: If the subscription is 'cancelled', acknowledge it respectfully[cite: 807].\n"
        "6. NO GIBBERISH: Do not invent facts, dates, or ticket IDs."
    ),
)


def drafter_node(state: UDAHubState):
    """
    Revised Drafter: Uses full history for tone/context,
    but anchors the content to the Research Facts.
    """
    # 1. Prepare the 'Ground Truth' for this specific turn
    # This acts as the "North Star" for the agent
    evidence_instruction = (
        f"--- CURRENT GROUND TRUTH ---\n"
        f"ANCHORED GOAL: {state.get('ticket_text')}\n"
        f"RESEARCH FINDINGS: {state.get('research_facts')}\n"
        f"USER PROFILE: {state.get('user_name')} ({state.get('customer_tier')})\n"
        "----------------------------\n"
        "INSTRUCTION: Use the conversation history below to match the user's tone, "
        "but ensure your answer is strictly based on the GROUND TRUTH above."
    )

    # 2. Add Policy Feedback if we are looping back from a FAIL
    feedback = state.get("policy_feedback")
    if feedback:
        evidence_instruction += (
            f"\n\nSTRICT REVISION NEEDED: Your previous draft was rejected: {feedback}"
        )

    # 3. Construct the message list for the agent
    # We take the existing conversation history and append our 'Ground Truth' as an instruction
    full_history = state.get("messages", [])
    agent_input = full_history + [HumanMessage(content=evidence_instruction)]

    config = {"configurable": {"thread_id": state.get("user_id", "drafting_turn")}}

    # 4. Invoke the Agent
    result = drafter_agent.invoke({"messages": agent_input}, config)

    response_text = result["messages"][-1].content

    out = {
        "ai_response": response_text,
        # We append the AI's final answer to the history
        "messages": [AIMessage(content=response_text)],
        "policy_feedback": None,
    }

    node_log(
        "drafter",
        ai_response_present=bool(out.get("ai_response")),
        policy_feedback=out.get("policy_feedback"),
    )

    return out


if __name__ == "__main__":
    print("🚀 --- RUNNING DRAFTER TEST (Alice Kingsley ---")

    # Mocking the state exactly as it would appear after the Researcher node
    mock_facts = """
    - Booking Status: The user has a reservation for the 'Christ the Redeemer Experience' with the status 'reserved'.
    - Refund Policy: CultPass plans are generally non-refundable after the billing cycle starts.
    - Event Specifics: For event-specific issues (like cancellations), the policy is to offer a credit equal to the ticket value.
    """

    test_state: UDAHubState = {
        # Option B requirement: The message history from the chat UI
        "messages": [
            HumanMessage(
                content="I have a reservation for the Christ the Redeemer Experience but I can't go anymore. What is your refund policy for events?"
            )
        ],
        "ticket_text": "I have a reservation for the Christ the Redeemer Experience but I can't go anymore. What is your refund policy for events?",
        "user_email": "alice.kingsley@wonderland.com",
        "user_id": "a4ab87",
        "customer_tier": "basic",
        "subscription_status": "cancelled",
        "category": "billing",
        "urgency": "standard",
        "research_facts": mock_facts.strip(),
        "ai_response": None,
        "policy_grade": None,
        "policy_feedback": None,
    }

    print("\nOriginal Ticket:", test_state["ticket_text"])
    print("Facts Provided:", test_state["research_facts"])
    print("\nDrafting response...\n")
    print("-" * 50)

    # Run the node
    result = drafter_node(test_state)

    print("\n✅ FINAL EMAIL TO CUSTOMER:\n")
    print(result["ai_response"])

    print("\n🛠️  OPTION B VERIFICATION:")
    if "messages" in result:
        last_msg = result["messages"][-1]
        print(f"Chat UI Output: [{type(last_msg).__name__}] {last_msg.content}")

    print("-" * 50)
