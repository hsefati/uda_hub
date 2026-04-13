import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from uda_hub.agentic.tools.udahub_state import UDAHubState

load_dotenv()


# ==========================================
# 2. AGENT FACTORY (Reused for consistency)
# ==========================================
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


# ==========================================
# 3. AGENT INSTANCE
# ==========================================
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
        "You are the UDA-Hub Drafter Agent. Your ONLY job is to write the final response to the customer.\n\n"
        "RULES:\n"
        "1. You will be given the original ticket, the customer's tier, and the 'Research Facts'.\n"
        "2. BASE YOUR ANSWER ENTIRELY ON THE RESEARCH FACTS. Do not invent policies, prices, or promises.\n"
        "3. Be empathetic and professional.\n"
        "4. If the customer is 'premium', specifically thank them for being a premium member.\n"
        "5. If the subscription is 'cancelled', acknowledge that respectfully.\n"
        "6. Do not include internal system notes. Write directly to the customer."
    ),
)


# ==========================================
# 4. LANGGRAPH NODE
# ==========================================
def drafter_node(state: UDAHubState):
    """
    Takes facts and any previous feedback to draft or revise the email.
    """
    # 1. Base Context
    context_payload = (
        f"Customer Tier: {state.get('customer_tier')}\n"
        f"Subscription Status: {state.get('subscription_status')}\n"
        f"Original Ticket: {state.get('ticket_text')}\n"
        f"--- RESEARCH FACTS ---\n"
        f"{state.get('research_facts')}\n"
    )

    # 2. Injection of Policy Feedback (The "Fix-it" Instruction)
    feedback = state.get("policy_feedback")
    if feedback:
        context_payload += (
            f"\n--- ATTENTION: PREVIOUS DRAFT REJECTED ---\n"
            f"Your previous draft failed policy review for the following reason:\n"
            f"{feedback}\n"
            f"Please rewrite the response to correct these issues while remaining empathetic."
        )

    # 3. Invoke the agent
    config = {"configurable": {"thread_id": state.get("user_id", "default_thread")}}

    # We pass the context as a fresh HumanMessage to trigger a new draft
    result = drafter_agent.invoke({"messages": [("user", context_payload)]}, config)

    # 4. Clear the feedback in the return so the next QA check starts fresh
    return {
        "ai_response": result["messages"][-1].content,
        "policy_feedback": None,  # Resetting feedback after attempt
    }


# ==========================================
# 5. TEST SCRIPT
# ==========================================
if __name__ == "__main__":
    print("--- RUNNING DRAFTER TEST (Alice Kingsley) ---")

    # We mock the exact state that the Researcher just outputted in our last test!
    mock_facts = """
    - Booking Status: The user has a reservation for the 'Christ the Redeemer Experience' with the status 'reserved'.
    - Refund Policy: CultPass plans are generally non-refundable after the billing cycle starts.
    - Event Specifics: For event-specific issues (like cancellations), the policy is to offer a credit equal to the ticket value.
    """

    test_state: UDAHubState = {
        "ticket_text": "I have a reservation for the Christ the Redeemer Experience but I can't go anymore. What is your refund policy for events?",
        "user_email": "alice.kingsley@wonderland.com",
        "user_id": "a4ab87",
        "customer_tier": "basic",
        "subscription_status": "cancelled",
        "category": "billing",
        "urgency": "standard",
        "research_facts": mock_facts.strip(),
        "ai_response": None,
    }

    print("\nOriginal Ticket:", test_state["ticket_text"])
    print("Facts Provided:", test_state["research_facts"])
    print("\nDrafting response...\n")
    print("-" * 50)

    # Run the node
    result = drafter_node(test_state)

    print("\n✅ FINAL EMAIL TO CUSTOMER:\n")
    print(result["ai_response"])
    print("-" * 50)
