import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from uda_hub.agentic.tools.udahub_state import UDAHubState
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

load_dotenv()

# Initialize the LLM (Using a cheap model because drafting a question is an easy task)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,  # Slight temperature for a natural, conversational tone
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)


clarifier_agent = create_agent(
    name="clarifier_agent",
    model=llm,
    system_prompt=(
        "You are the UDA-Hub Clarifier. Your job is to resolve ambiguity or missing identity.\n\n"
        "--- SPECIAL MISSION: RECURRING ISSUES ---\n"
        "If the 'is_recurring' flag is True, you must act as a Gatekeeper:\n"
        "1. Acknowledge the user's previous history found in the database.\n"
        "2. Ask if the current request is a follow-up to the previous ticket or a brand new issue.\n"
        "3. DO NOT perform research. Your goal is only to confirm the user's intent.\n\n"
        "--- STANDARD MISSION ---\n"
        "If there is no history, focus on identifying the user (email) or clarifying a vague request."
    ),
)


def clarificator_node(state: UDAHubState):
    """
    The Multi-Memory Clarifier:
    1. Checks 'messages' (Short-term) to see the immediate conversation.
    2. Checks 'user_history' (Long-term) to handle recurring issues.
    3. Prompts the user based on which memory is more urgent.
    """
    # Short-term context
    messages = state.get("messages", [])
    anchor = state.get("ticket_text")

    # Long-term context (from Historian)
    is_recurring = state.get("is_recurring", False)
    history = state.get("user_history")
    reasoning = state.get("history_reasoning")

    # Construction of the agent's task
    # We include the last few messages so the agent doesn't repeat itself
    recent_chat = "\n".join([f"{m.type}: {m.content}" for m in messages[-3:]])

    if is_recurring:
        # Priority: Address the duplicate ticket first (Gatekeeper mode)
        prompt_task = (
            f"--- LONG-TERM CONTEXT (Recurring Issue) ---\n"
            f"REASON: {reasoning}\n"
            f"HISTORY: {history}\n\n"
            f"--- SHORT-TERM CONTEXT (Recent Chat) ---\n"
            f"{recent_chat}\n\n"
            "The user might be repeating a past request. Ask them to clarify "
            "if this is the same issue or something new."
        )
    else:
        # Standard mode: Identity or intent clarification
        prompt_task = (
            f"--- RECENT CHAT ---\n{recent_chat}\n\n"
            f"ANCHORED REQUEST: {anchor}\n"
            "Please ask the user for their email or to clarify their vague request."
        )

    # Invoke the agent
    result = clarifier_agent.invoke({"messages": [("user", prompt_task)]})

    return {
        "ai_response": result["messages"][-1].content,
        "messages": [AIMessage(content=result["messages"][-1].content)],
        "status": "pending_user",
    }


if __name__ == "__main__":
    print("🚀 --- Testing Clarification Agent ---\n" + "=" * 50)

    # Test Case: An anonymous user complaining about a billing charge
    # Note: In Option B, the 'messages' list is the source of truth for the UI
    state_missing_info: UDAHubState = {
        "messages": [
            HumanMessage(
                content="I was charged $50 yesterday but I canceled my account!"
            )
        ],
        "ticket_text": "I was charged $50 yesterday but I canceled my account!",
        "category": "billing",
        "user_email": None,
        "user_id": None,
        "customer_tier": None,
        "subscription_status": None,
        "urgency": "standard",
        "confidence_score": 0.95,
        "ai_response": None,
        "status": "in_progress",
    }

    # Run the node
    result = clarification_node(state_missing_info)

    print(f"User Input:    {state_missing_info['ticket_text']}")
    print(f"AI String:     {result['ai_response']}")

    # Verify Option B compatibility
    if "messages" in result:
        last_msg = result["messages"][-1]
        print(f"UI Message:    [{type(last_msg).__name__}] {last_msg.content}")

    # If you kept the 'status' update in your node logic:
    if "status" in result:
        print(f"Graph Status:  {result['status']}")

    print("=" * 50)
    print(
        "✅ Test Complete: The response is now properly packaged for the chat_interface."
    )
