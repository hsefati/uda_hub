import os

from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from uda_hub.agentic.tools.udahub_state import UDAHubState
from uda_hub.agentic.tools.tools import ARCHIVIST_TOOLS

load_dotenv()


# ==========================================
# 2. AGENT INSTANCE
# ==========================================
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Cheap model is perfect for summarization/data extraction
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)


archivist_agent = create_agent(
    name="archivist_agent",
    model=llm,  
    tools=ARCHIVIST_TOOLS,
    system_prompt=(
        "You are the UDA-Hub Archivist. Your job is to document the final outcome of a ticket.\n"
        "1. Extract a one-sentence summary and relevant tags from the conversation.\n"
        "2. Use the 'save_ticket_summary' tool to write this to the database.\n"
        "3. Provide the ticket_id provided in the state."
    ),
)


def archivist_node(state: UDAHubState):
    # Generate a unique ticket ID if one doesn't exist in the state
    ticket_id = (
        state.get("ticket_id")
        or f"TICK_{state.get('user_id')}_{int(datetime.now().timestamp())}"
    )

    config = {"configurable": {"thread_id": f"archive_{ticket_id}"}}

    # The Archivist looks at the ticket_text, category, and the final ai_response
    context = (
        f"Ticket ID: {ticket_id}\n"
        f"Category: {state.get('category')}\n"
        f"Final Resolution: {state.get('ai_response')}"
    )

    result = archivist_agent.invoke({"messages": [("user", context)]}, config)

    return {"archive_summary": result["messages"][-1].content}


if __name__ == "__main__":
    print("--- TESTING ARCHIVIST AGENT ---")

    test_state: UDAHubState = {
        "ticket_text": "I can't attend the Samba Night, can I get a refund?",
        "user_id": "a4ab87",
        "category": "billing",
        "ai_response": "I have processed a credit to your account as per our policy since you cannot attend.",
        "archive_summary": None,
    }

    # Run node
    output = archivist_node(test_state)
    print(f"\nArchivist Result: {output['archive_summary']}")
