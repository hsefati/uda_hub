import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from uda_hub.agentic.tools.udahub_state import UDAHubState
from langchain_core.messages import HumanMessage

load_dotenv()

# Initialize the LLM (Using a cheap model because drafting a question is an easy task)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,  # Slight temperature for a natural, conversational tone
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)


def clarification_node(state: UDAHubState) -> dict:
    """
    An 'Identity Specialist' node that analyzes the full history
    to ask for the most appropriate missing identifier.
    """
    # 1. Pull the full context
    messages = state.get("messages", [])
    ticket_text = state.get("ticket_text", "an issue")

    # 2. Dynamic Prompt for identity recovery
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the UDA-Hub Identity Specialist. Your goal is to help find the customer's account.\n\n"
                "INSTRUCTIONS:\n"
                "1. Analyze the conversation history. See if the user already provided an identifier.\n"
                "2. If they provided an email but the system still didn't find them, ask for a Membership ID or Phone Number instead.\n"
                "3. If they haven't provided anything, ask for their Email OR Membership ID.\n"
                "4. ACKNOWLEDGE their specific problem (found in the 'User's Issue') so they know we are listening.\n"
                "5. Keep it to 2 empathetic sentences. Do NOT try to solve the technical issue yet.",
            ),
            # Passing the actual messages list so the LLM sees the history
            *messages,
            ("human", f"User's Issue: {ticket_text}"),
        ]
    )

    chain = prompt | llm

    # We don't need to pass category/text separately if they are in history,
    # but providing 'ticket_text' specifically helps the 'Acknowledge' instruction.
    response = chain.invoke({})
    response_text = response.content

    return {
        "ai_response": response_text,
        "messages": [AIMessage(content=response_text)],
        "status": "pending_user",  # Signal that we are waiting for user input
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
