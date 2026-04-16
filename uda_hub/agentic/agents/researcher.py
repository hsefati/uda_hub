import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from uda_hub.agentic.tools.tools import RESEARCHER_TOOLS
from uda_hub.agentic.tools.udahub_state import UDAHubState
from uda_hub.agentic.tools.logging import node_log

from langchain.agents import create_agent

load_dotenv()


def create_agent_wrapper(
    name: str,
    model,
    tools: list,
    system_prompt: str,
    checkpointer=None,
    interrupt_before: list[str] | None = None,
):
    """
    Returns a compiled LangGraph ReAct agent.
    """
    # FIXED: Using create_react_agent
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,  # LangGraph uses state_modifier for the prompt
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        name=name,
    )


class ResearchGrade(BaseModel):
    reasoning: str = Field(
        description="Brief explanation of why the facts are or are not sufficient."
    )
    confidence_score: float = Field(
        description="Score between 0.0 and 1.0 based on how well the facts answer the ticket."
    )
    is_sufficient: bool = Field(
        description="True if we have enough info to draft a response; False if we must escalate."
    )


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)

researcher_agent = create_agent_wrapper(
    name="researcher_agent",
    model=llm,
    tools=RESEARCHER_TOOLS,
    system_prompt=(
        "You are the UDA-Hub Researcher Agent. Your mission is to find factual truth using provided tools.\n"
        "1. Focus strictly on the 'Anchored Ticket' provided in the input.\n"
        "2. Query 'udahub.db' for policies in the knowledge table (title, content).\n"
        "3. Query 'cultpass.db' to verify user reservations (join reservations and experiences tables).\n"
        "4. DO NOT reply to the customer. Output a bulleted list of facts found.\n"
        "5. If a user_id is provided, verify their tier and booking status[cite: 582]."
    ),
)


def research_safety_router(state: UDAHubState) -> Literal["drafter", "escalator"]:
    """
    Deterministic Safety Gate:
    If the Research Grader's confidence is too low, we bypass the AI
    Drafter to prevent hallucinations and route to the Escalator.
    """
    confidence = state.get("retrieval_confidence", 0.0)
    needs_esc = state.get("needs_escalation", False)

    # Reviewer's requirement: Ensure escalation when no relevant knowledge is found.
    # We use 0.6 as a standard threshold for 'sufficient information'.
    if needs_esc or confidence < 0.6:
        print(f"🚨 RESEARCH GATE: Low confidence ({confidence}). Forcing Escalation.")
        return "escalator"

    print(f"✅ RESEARCH GATE: High confidence ({confidence}). Proceeding to Drafter.")
    return "drafter"


# Constants for the node
MAX_RESEARCH_RETRIES = 2


def researcher_node(state: UDAHubState):
    """
    Improved Researcher: Features a self-correction loop.
    If the grader finds the facts insufficient, the agent is asked to
    pivot its search strategy (e.g., broader keywords) before giving up.
    """
    anchor = state.get("ticket_text")
    user_id = state.get("user_id")
    tier = state.get("customer_tier", "regular")

    # We maintain a local history for this specific research task
    research_history = [
        (
            "system",
            f"TASK: Find policies and bookings for: {anchor}\nCONTEXT: UserID={user_id}, Tier={tier}",
        )
    ]

    attempts = 0
    final_facts = ""
    final_grade = None

    while attempts <= MAX_RESEARCH_RETRIES:
        # --- STEP 1: EXECUTE AGENT ---
        # We pass the full local history so the agent remembers previous failed queries
        result = researcher_agent.invoke({"messages": research_history})
        final_facts = result["messages"][-1].content

        # --- STEP 2: GRADE THE RESEARCH ---
        grader_llm = llm.with_structured_output(ResearchGrade)
        grading_prompt = (
            "You are the UDA-Hub Research Grader. Assess if the facts are sufficient.\n\n"
            f"USER TICKET: {anchor}\n"
            f"FOUND FACTS: {final_facts}\n\n"
            "SCORING RULES:\n"
            "1. SUCCESS (0.9-1.0): The facts show the standard policy AND confirm the user's Tier (e.g., Basic/Premium). "
            "If the user is Basic, do NOT penalize for a lack of 'unique' benefits.\n"
            "2. PARTIAL (0.5-0.7): You have the policy but haven't verified the user's specific tier or reservations.\n"
            "3. FAIL (0.0-0.4): No relevant policy found or search failed.\n"
            "NOTE: If the agent confirms the user is a 'Basic' tier and provides the standard inclusions, THIS IS SUFFICIENT."
        )
        final_grade: ResearchGrade = grader_llm.invoke(grading_prompt)

        # EXIT CONDITION: High confidence or we've run out of retries
        if final_grade.is_sufficient or attempts == MAX_RESEARCH_RETRIES:
            break

        # --- STEP 3: PIVOT (Preparation for Retry) ---
        attempts += 1
        research_history.append(("assistant", final_facts))
        research_history.append(
            (
                "user",
                f"INSUFFICIENT (Confidence: {final_grade.confidence_score}). "
                f"FEEDBACK: {final_grade.reasoning}. "
                "Try a broader SQL search using LIKE or checking the 'tags' column.",
            )
        )

    # --- STEP 4: RETURN FINAL STATE ---
    return {
        "research_facts": final_facts,
        "retrieval_confidence": final_grade.confidence_score,
        "needs_escalation": not final_grade.is_sufficient,
    }


if __name__ == "__main__":
    print("--- RUNNING SUCCESSFUL RESEARCHER TEST ---")

    # 1. Mocking the exact state as it would come from the Enricher & Classifier
    # We use Alice's exact ID so the SQL tools will find real data.
    successful_state: UDAHubState = {
        "ticket_text": "I have a reservation for the Christ the Redeemer Experience but I can't go anymore. What is your refund policy for events?",
        "user_email": "alice.kingsley@wonderland.com",
        "user_id": "a4ab87",  # Alice's real ID in cultpass.db
        "customer_tier": "basic",  # Alice's real tier
        "subscription_status": "cancelled",  # Alice's real status
        "category": "billing",
        "urgency": "standard",
        "research_facts": None,
    }

    print(f"\nUser: {successful_state['user_email']}")
    print(f"Ticket: '{successful_state['ticket_text']}'\n")
    print("Agent is thinking and querying tools... (Look for the tool logs below)\n")
    print("-" * 50)

    # 2. Call the node directly
    result = researcher_node(successful_state)

    # 3. Print the final proper response
    print("-" * 50)
    print("\n✅ FINAL RESEARCH FACTS GENERATED BY AGENT:\n")
    print(result["research_facts"])
