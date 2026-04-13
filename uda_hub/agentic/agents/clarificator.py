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
    Drafts a polite message asking the user for missing information (like an email).
    Ensures the chat interface can display the message via the messages key.
    """
    category = state.get("category", "general issue")
    ticket_text = state.get("ticket_text", "")

    # Focused prompt for identity recovery
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the friendly 'Front Desk' intake agent for CultPass support. "
                "A customer has reached out, but we cannot find their account in our database. "
                "Your ONLY job is to acknowledge their issue and politely ask them to provide "
                "the email address associated with their account so we can look it up. "
                "Keep it brief (1-2 sentences). Do NOT try to solve their problem.",
            ),
            ("human", "Category: {category}\nUser's Message: {ticket_text}"),
        ]
    )

    # Note: 'llm' must be defined in your file or imported
    chain = prompt | llm

    response = chain.invoke({"category": category, "ticket_text": ticket_text})

    # Extract the string content from the LLM response
    response_text = response.content

    # Return the drafted message to our internal state and the message list for the UI
    return {
        "ai_response": response_text,
        "messages": [AIMessage(content=response_text)],
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
