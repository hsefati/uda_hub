import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from uda_hub.agentic.tools.tools import RESEARCHER_TOOLS
from uda_hub.agentic.tools.udahub_state import UDAHubState

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


def researcher_node(state: UDAHubState):
    """
    1. Orchestrates the Researcher Agent to find facts.
    2. Uses a structured LLM call to Grade those facts against the ticket.
    3. Returns facts, a numerical confidence score, and an escalation flag.
    """
    anchor = state.get("ticket_text")
    user_id = state.get("user_id")
    tier = state.get("customer_tier", "regular")

    # --- STEP 1: EXECUTE RESEARCH AGENT ---
    prompt_task = (
        f"ANCHORED TICKET: {anchor}\n"
        f"USER CONTEXT: ID={user_id}, Tier={tier}\n\n"
        "Please find the relevant policies and verify any associated bookings."
    )

    config = {
        "configurable": {"thread_id": state.get("user_id", "internal_research_turn")}
    }

    # Execute the agent to gather raw facts
    result = researcher_agent.invoke({"messages": [("user", prompt_task)]}, config)
    found_facts = result["messages"][-1].content

    # --- STEP 2: LLM-BASED GRADING (The "Missing Piece") ---
    # We use structured output to generate a numerical confidence score
    grader_llm = llm.with_structured_output(ResearchGrade)

    grading_prompt = (
        "You are the UDA-Hub Research Grader. Your goal is to assess if the found facts "
        "are sufficient to answer the user's ticket correctly.\n\n"
        f"USER TICKET: {anchor}\n"
        f"FOUND FACTS: {found_facts}\n\n"
        "SCORING RULES:\n"
        "1. If the facts contain the specific policy or booking info needed: 0.8 - 1.0\n"
        "2. If the facts are related but incomplete: 0.4 - 0.7\n"
        "3. If the facts say 'No information found' or are irrelevant: 0.0 - 0.3 (is_sufficient = False)"
    )

    grade: ResearchGrade = grader_llm.invoke(grading_prompt)

    # --- STEP 3: RETURN STRUCTURED STATE ---
    return {
        "research_facts": found_facts,
        "retrieval_confidence": grade.confidence_score,  # Numerical score for routing
        "needs_escalation": not grade.is_sufficient,  # Deterministic flag for routing
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
