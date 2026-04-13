import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from uda_hub.agentic.tools.udahub_state import UDAHubState

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
    Flags the state as 'pending_user' so the system knows to wait.
    """
    category = state.get("category", "general issue")
    ticket_text = state.get("ticket_text", "")

    # We use a very focused prompt. We don't want it trying to solve the problem.
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

    chain = prompt | llm

    response = chain.invoke({"category": category, "ticket_text": ticket_text})

    # Return the drafted message and change the status
    return {"ai_response": response.content, "status": "pending_user"}


if __name__ == "__main__":
    print("--- Testing Clarification Agent ---\n")

    # Test Case: An anonymous user complaining about a billing charge
    # (The Supervisor routed them here because user_id was None)
    state_missing_info: UDAHubState = {
        "ticket_text": "I was charged $50 yesterday but I canceled my account!",
        "category": "billing",
        # Missing data:
        "user_email": None,
        "user_id": None,
        "customer_tier": None,
        "subscription_status": None,
        "urgency": "standard",
        "confidence_score": 0.95,
        "ai_response": None,
        "status": "in_progress",
    }

    result = clarification_node(state_missing_info)

    print(f"User Said:   {state_missing_info['ticket_text']}")
    print(f"AI Response: {result['ai_response']}")
    print(f"New Status:  {result['status']}")

    # Expected Output Example:
    # "I'd be happy to look into that charge for you! Could you please reply with the email address associated with your canceled account?"
