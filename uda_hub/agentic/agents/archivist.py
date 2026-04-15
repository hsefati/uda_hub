import os

from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from uda_hub.agentic.tools.udahub_state import UDAHubState
from uda_hub.agentic.tools.tools import ARCHIVIST_TOOLS

load_dotenv()


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
    """
    Identity-aware Archivist: Only triggers the agent if a user_id exists.
    """
    user_id = state.get("user_id")

    # 1. THE IDENTITY GATE: No ID, No Archive
    if not user_id:
        print("⏸️ Archivist: Bypassing archive (Anonymous session).")
        return {"archive_summary": "Skipped: User is anonymous."}

    # 2. Setup ID and Context
    # We use the user_id to ensure the ticket is linked correctly in the DB
    ticket_id = (
        state.get("ticket_id") or f"TICK_{user_id}_{int(datetime.now().timestamp())}"
    )

    # We give the agent the "Anchor" and the "Resolution" so the summary is accurate
    context = (
        f"--- TICKET DETAILS ---\n"
        f"Ticket ID: {ticket_id}\n"
        f"Original Request (Anchor): {state.get('ticket_text')}\n"
        f"Final Category: {state.get('category')}\n"
        f"Final Resolution Provided: {state.get('ai_response')}\n"
        f"--- FULL HISTORY ---\n"
        f"{state.get('messages')}"
    )

    # 3. Invoke the Agent
    config = {"configurable": {"thread_id": f"archive_{ticket_id}"}}
    result = archivist_agent.invoke({"messages": [("user", context)]}, config)

    # 4. Return the summary for the state
    return {
        "archive_summary": result["messages"][-1].content,
        "ticket_id": ticket_id,  # Ensure the ticket_id persists in the state
    }


from datetime import datetime

if __name__ == "__main__":
    print("🗄️ --- TESTING ARCHIVIST AGENT & IDENTITY GATE ---")
    print("=" * 60)

    # ---------------------------------------------------------
    # TEST 1: The "Success Path" (Authenticated User)
    # ---------------------------------------------------------
    state_authenticated: UDAHubState = {
        "ticket_text": "I can't attend the Samba Night, can I get a refund?",
        "user_id": "a4ab87",
        "category": "billing",
        "ai_response": "I have processed a partial credit to your account since the Samba Night is tonight.",
        "messages": [
            {
                "role": "user",
                "content": "I can't attend the Samba Night, can I get a refund?",
            },
            {
                "role": "assistant",
                "content": "I have processed a partial credit to your account since the Samba Night is tonight.",
            },
        ],
        "archive_summary": None,
    }

    # ---------------------------------------------------------
    # TEST 2: The "Identity Gate" (Anonymous User)
    # ---------------------------------------------------------
    state_anonymous: UDAHubState = {
        "ticket_text": "How do I sign up for a premium membership?",
        "user_id": None,  # Should trigger the bypass
        "category": "general_inquiry",
        "ai_response": "You can sign up via the 'Profile' tab in the app!",
        "archive_summary": None,
    }

    # ---------------------------------------------------------
    # TEST 3: The "Complex Escalation" (Verification of Tags)
    # ---------------------------------------------------------
    state_escalated: UDAHubState = {
        "ticket_text": "My payment failed three times and I'm being double charged!",
        "user_id": "vip_user_99",
        "category": "billing",
        "ai_response": "I see the duplicate charge. I'm escalating this to our finance lead immediately.",
        "status": "pending_human",
        "messages": [
            {"role": "user", "content": "I'm being double charged!"},
            {
                "role": "assistant",
                "content": "I'm escalating this to our finance lead immediately.",
            },
        ],
        "archive_summary": None,
    }

    test_scenarios = [
        ("AUTHENTICATED SUCCESS", state_authenticated),
        ("ANONYMOUS BYPASS", state_anonymous),
        ("VIP ESCALATION", state_escalated),
    ]

    for name, state in test_scenarios:
        print(f"\n▶️ SCENARIO: {name}")
        print("-" * 30)

        try:
            output = archivist_node(state)

            # Check if it was a skip or a save
            summary = output.get("archive_summary", "N/A")
            if "Skipped" in summary:
                print(f"Status: ⏸️ Logic Bypassed")
            else:
                print(f"Status: ✅ Successfully Processed")

            print(f"Agent Output: {summary}")

        except Exception as e:
            print(f"❌ Test Failed with error: {e}")

    print("\n" + "=" * 60)
    print("🏁 ARCHIVIST SUITE COMPLETE")
