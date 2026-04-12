import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# Initialize the LLM (Keeping your Vocareum configuration)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,  # Keep at 0 for deterministic, analytical classification
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)


# 1. Improved Schema: Aligned with UDA-Hub Requirements
class TicketClassification(BaseModel):
    category: str = Field(
        ...,
        description=(
            "The primary category of the customer support ticket. "
            "- billing: invoices, refunds, charges, payments\n"
            "- technical_issue: bugs, login problems, software errors\n"
            "- account_management: password resets, profile updates, upgrades\n"
            "- general_inquiry: questions about features, pricing, or general info"
        ),
        json_schema_extra={
            "enum": [
                "billing",
                "technical_issue",
                "account_management",
                "general_inquiry",
            ]
        },
    )
    urgency: str = Field(
        ...,
        description=(
            "Assess the urgency based on customer sentiment and issue type. "
            "- low: general questions, no rush\n"
            "- standard: normal operational issues\n"
            "- high: angry customer, VIP, or severe blocker (e.g., cannot log in)"
        ),
        json_schema_extra={"enum": ["low", "standard", "high"]},
    )
    confidence_score: float = Field(
        ...,
        description="A score between 0.0 and 1.0 indicating how confident you are in this classification.",
    )


# 2. Improved Prompt & Logic
def classify_ticket(ticket_text: str) -> dict:
    """
    Classifies an incoming support ticket for the UDA-Hub routing system.
    """
    routing_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the primary Triage Classifier for UDA-Hub, an intelligent customer support system. "
                "Your job is to read incoming customer messages, determine their core intent, assess the "
                "urgency based on tone and content, and output strict structured data. "
                "If the user sounds highly frustrated, escalate the urgency.",
            ),
            ("human", "Ticket details to classify:\n\n{ticket_text}"),
        ]
    )

    # Bind the LLM to our specific Pydantic schema
    classifier_llm = llm.with_structured_output(TicketClassification)

    # Chain the prompt and the LLM
    chain = routing_prompt | classifier_llm

    # Execute the chain
    response = chain.invoke({"ticket_text": ticket_text})

    # Return as a dictionary so it can easily update a LangGraph State
    return response.model_dump()


# --- Test the Classifier ---
if __name__ == "__main__":
    test_ticket = "I was charged twice this month for my pro subscription!! Fix this right now or I am canceling."
    result = classify_ticket(test_ticket)
    print(result)
    # Expected output:
    # {'category': 'billing', 'urgency': 'high', 'confidence_score': 0.95}
