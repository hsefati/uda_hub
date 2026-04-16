import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_openai import ChatOpenAI
from uda_hub.agentic.tools.udahub_state import UDAHubState
from uda_hub.agentic.tools.logging import node_log

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)


class ConciergeDecision(BaseModel):
    intent: Literal["greeting", "closure", "support_request", "need_user_response"] = (
        Field(description="The nature of the message.")
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
    user_input = state.get("latest_input", "")
    chat_history = state.get("messages", [])

    # Provide history so the LLM knows if it already asked for an ID
    history_context = "\n".join([f"{m.type}: {m.content}" for m in chat_history[-3:]])

    concierge_llm = llm.with_structured_output(ConciergeDecision)

    prompt = (
        "You are the UDA-Hub Concierge.\n\n"
        "GOAL: Triage the user and ensure every support request is linked to an identity (Email, Name, or ID).\n\n"
        "LOGIC (Evaluate based on 'User_latest_input'):\n"
        "1. GREETING: If the user says hello or hi, respond naturally and set intent='greeting'.\n"
        "2. CLOSURE: If the user indicates they are leaving or says 'bye', 'goodbye', or 'thanks', "
        "set intent='need_user_response' and respond with a polite closing remark.\n"
        "3. SUPPORT_REQUEST + ID FOUND: If the user has a problem and an Email, Name, or ID is "
        "present in 'User_latest_input' or 'History_Context', set intent='support_request' and is_identified=True. "
        "Response can be empty.\n"
        "4. SUPPORT_REQUEST + NO ID FOUND: If the user has a problem but no identity is found, "
        "set intent='need_user_response' and is_identified=False. Response MUST politely ask for their email or name.\n\n"
        f"History_Context:\n{history_context}\n"
        f"User_latest_input: {user_input}"
    )

    decision: ConciergeDecision = concierge_llm.invoke(prompt)

    # Log the triage decision
    node_log(
        "concierge",
        triage_intent=decision.intent,
        is_identified=decision.is_identified,
        has_response=bool(decision.response),
    )

    return {
        "ai_response": decision.response,
        "triage_intent": decision.intent,
        "status": "closed" if decision.intent == "closure" else "in_progress",
    }


def concierge_router(state: UDAHubState) -> Literal["enricher", "archivist", "END"]:
    """
    Determines if we continue the support workflow or end the turn.
    """
    intent = state.get("triage_intent")

    # If it's a real issue, go to the heavy nodes
    if intent == "support_request":
        return "enricher"
    if intent == "need_user_response":
        return "END"  # We could also route to a "clarifier" node if we wanted to handle follow-up messages in the same turn
    # For greetings, closures, or "need_user_response", go to Archivist
    # to save the message and then finish.
    return "archivist"


if __name__ == "__main__":
    test_cases = [
        # Turn 1: Support request with NO ID
        {
            "text": "I can't attend my Christ the Redeemer booking. Refund?"
        },
        # Turn 2: Providing the ID (Simulation of a follow-up)
        {
            "text": "Bye",
        },
    ]

    print("\n" + "=" * 60)
    print("🧪 TESTING CONCIERGE 2.0 (Identity-Aware Triage)")
    print("=" * 60 + "\n")

    for i, case in enumerate(test_cases, 1):
        # Convert simple history dicts to message objects if needed by your state
        state: UDAHubState = {"latest_input": case["text"]}

        result = concierge_node(state)

        print(f"TEST CASE #{i}: '{case['text']}'")
        print(f"  ├─ Intent:        {result['triage_intent'].upper()}")
        print(f"  ├─ Status:        {result['status']}")

        if result["ai_response"]:
            print(f'  └─ AI Response:   "{result["ai_response"]}"')
        else:
            print(f"  └─ AI Response:   (None - Identification Secured. Proceeding.)")
        print("-" * 40)
