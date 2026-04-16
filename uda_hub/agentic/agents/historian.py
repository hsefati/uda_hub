import os
from dotenv import load_dotenv
from typing import Optional

from langchain_openai import ChatOpenAI
from uda_hub.agentic.tools.tools import lookup_user_history
from uda_hub.agentic.tools.udahub_state import UDAHubState
from uda_hub.agentic.tools.logging import node_log

from pydantic import BaseModel, Field

load_dotenv()


class SimilarityGrade(BaseModel):
    is_recurring: bool = Field(
        description="True if the current ticket is semantically similar to a past ticket."
    )
    similar_ticket_id: Optional[str] = Field(
        description="The ID of the past similar ticket, if found."
    )
    reasoning: str = Field(
        description="Explanation of why this is or isn't a duplicate."
    )
    suggested_action: str = Field(
        description="Recommended next step (e.g., 'escalate', 'proceed').",
        json_schema_extra=[
            "escalate",
            "proceed",
        ],
    )


llm = ChatOpenAI(
    model="gpt-4o-mini",  # Cheap model is perfect for summarization/data extraction
    temperature=0.0,
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)


def historian_node(state: UDAHubState) -> dict:
    """
    Retrieves history and checks for duplicates or recurring issues.
    If a similar ticket is found, it flags it for the supervisor/classifier.
    """
    user_id = state.get("user_id")
    current_ticket = state.get("ticket_text")

    if not user_id:
        return {"user_history": "Anonymous user: No history available."}

    # 1. Fetch Durable Memory from the DB
    past_history_text = lookup_user_history.invoke({"user_id": user_id})

    # 2. LLM Similarity Check
    # We ask the LLM to compare current intent vs. the 'SUMMARY' blocks from the DB
    grader_llm = llm.with_structured_output(SimilarityGrade)

    grader_prompt = (
        "You are a Duplicate Ticket Detector. Compare the CURRENT TICKET with the USER HISTORY.\n"
        "If the user is asking the exact same question that was recently resolved, "
        "or if this is a follow-up to a recent ticket, flag it as recurring.\n\n"
        f"CURRENT TICKET: {current_ticket}\n"
        f"USER HISTORY:\n{past_history_text}"
    )

    grade: SimilarityGrade = grader_llm.invoke(grader_prompt)

    # Log the history analysis
    node_log(
        "historian",
        user_id=user_id,
        is_recurring=grade.is_recurring,
        similar_ticket_id=grade.similar_ticket_id,
        suggested_action=grade.suggested_action,
    )

    return {
        "user_history": past_history_text,
        "is_recurring": grade.is_recurring,
        "history_reasoning": grade.reasoning,
        "similar_ticket_id": grade.similar_ticket_id,
        # If it's recurring, we might want the Supervisor to route to 'clarifier' or 'escalator'
        "suggested_history_action": grade.suggested_action,
    }
