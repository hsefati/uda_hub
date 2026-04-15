import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI
from uda_hub.agentic.tools.udahub_state import UDAHubState

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)


class ConciergeDecision(BaseModel):
    intent: Literal["greeting", "closure", "support_request", "need_user_response"] = Field(
        description="The nature of the message."
    )
    response: str = Field(
        description="The direct response for greetings or closures. If support_request, this can be empty."
    )
    is_identified: bool = Field(
        description="True if an email, name, or ID is found in the current message or history."
    )


def concierge_node(state: UDAHubState):
    """
    Concierge 3.0: The Identity Gatekeeper.
    Blocks support requests until user identity is established.
    """
    user_input = state.get("ticket_text", "")
    chat_history = state.get("messages", [])
    
    # Provide history so the LLM knows if it already asked for an ID
    history_context = "\n".join([f"{m.type}: {m.content}" for m in chat_history[-3:]])

    concierge_llm = llm.with_structured_output(ConciergeDecision)

    prompt = (
        "You are the UDA-Hub Concierge.\n\n"
        "GOAL: Ensure every support request is linked to an identity (Email, Name, or ID).\n\n"
        "LOGIC:\n"
        "1. GREETING/CLOSURE: Respond naturally. is_identified doesn't matter here.\n"
        "2. SUPPORT_REQUEST + ID FOUND: Set intent='support_request', is_identified=True. Response can be empty.\n"
        "3. SUPPORT_REQUEST + NO ID FOUND: Set intent='need_user_response', is_identified=False. "
        "   Response MUST be a polite request for their email or name to proceed.\n\n"
        f"History:\n{history_context}\n"
        f"User says: {user_input}"
    )

    decision: ConciergeDecision = concierge_llm.invoke(prompt)

    return {
        "ai_response": decision.response,
        "triage_intent": decision.intent,
        "status": "closed" if decision.intent == "closure" else "in_progress",
    }
    


def concierge_router(state: UDAHubState) -> Literal["enricher", "archivist", "end"]:
    """
    Determines if we continue the support workflow or end the turn.
    """
    intent = state.get("triage_intent")

    if intent == "support_request":
        return "enricher"

    if intent in ["closure"]:
        return "archivist"
    
    if intent == "greeting":
        return "end"


if __name__ == "__main__":
    test_cases = [
        # Turn 1: Support request with NO ID
        {"text": "I need a refund for my trip.", "history": []},
        # Turn 2: Providing the ID (Simulation of a follow-up)
        {
            "text": "My email is alice@wonderland.com",
        },
        # Turn 3: Casual chatter (Should not ask for ID)
        {"text": "Thanks for your help!", "history": []},
    ]

    print("\n" + "=" * 60)
    print("🧪 TESTING CONCIERGE 2.0 (Identity-Aware Triage)")
    print("=" * 60 + "\n")

    for i, case in enumerate(test_cases, 1):
        # Convert simple history dicts to message objects if needed by your state
        state: UDAHubState = {"ticket_text": case["text"]}

        result = concierge_node(state)

        print(f"TEST CASE #{i}: '{case['text']}'")
        print(f"  ├─ Intent:        {result['triage_intent'].upper()}")
        print(f"  ├─ Status:        {result['status']}")

        if result["ai_response"]:
            print(f'  └─ AI Response:   "{result["ai_response"]}"')
        else:
            print(f"  └─ AI Response:   (None - Identification Secured. Proceeding.)")
        print("-" * 40)
