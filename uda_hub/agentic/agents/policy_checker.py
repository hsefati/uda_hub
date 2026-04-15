import os
import json
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from uda_hub.agentic.tools.udahub_state import UDAHubState
from uda_hub.agentic.tools.logging import node_log

load_dotenv()


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,  # CRITICAL: 0.0 for strict logic and compliance
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)


# 1. The Schema (The "What")
class PolicyGrade(BaseModel):
    grade: str = Field(description="Must be 'PASS' or 'FAIL'")
    feedback: str = Field(description="Specific instructions for the drafter if FAIL")


# 2. The Prompt Template (The "How")
# This provides the clarity and logic instructions you were asking about
qa_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are the UDA-Hub Quality Auditor. Your goal is to ensure customer responses "
                "are 100% compliant with company research and the user's specific request.\n\n"
                "STRICT RULES:\n"
                "1. If the research says 'No Refund', any promise of a refund is a FAIL.\n"
                "2. If the user's name is known, it must be used.\n"
                "3. No internal system IDs (e.g., e6376d) should be visible to the customer."
            ),
        ),
        (
            "human",
            (
                "Compare this DRAFT against these FACTS and the original GOAL.\n\n"
                "GOAL: {ticket_text}\n"
                "FACTS: {research_facts}\n"
                "DRAFT: {ai_response}"
            ),
        ),
    ]
)

# 3. The Binding
# We bind the schema to the LLM
structured_llm = llm.with_structured_output(PolicyGrade)

# 4. The Execution Chain
# This combines the logic (prompt) with the format (structured_llm)
qa_chain = qa_prompt_template | structured_llm

# We use a system prompt that enforces a JSON output structure
# policy_checker_agent = create_agent(
#     model=structured_llm,
#     tools=[],
#     system_prompt=(
#         "You are the UDA-Hub Quality Auditor. Your job is to audit the drafted response.\n\n"
#         "STRICT AUDIT CRITERIA:\n"
#         "1. ANCHOR ALIGNMENT: Does the email address the 'Anchored Ticket Goal'? If the goal was a refund, did they discuss the refund?\n"
#         "2. FACTUAL TRUTH: Compare the 'Proposed Email' against the 'Research Facts'. \n"
#         "   - If facts say 'No Refund', the email MUST NOT promise a refund.\n"
#         "   - If facts say 'Experience is at 5 PM', the email MUST NOT say 6 PM.\n"
#         "3. TIER COMPLIANCE: Ensure 'Premium' perks aren't offered to 'Basic' users.\n"
#         "4. NO INTERNAL LEAKS: Ensure no internal user IDs or raw database rows are in the draft.\n\n"
#         "If ANY criteria are failed, set grade to 'FAIL' and provide specific 'Fix-it' instructions."
#     ),
#     name="policy_checker",
# )


# def policy_checker_node(state: UDAHubState):
#     """
#     Audits the draft against the 'Ground Truth' and the 'Anchored Goal'.
#     """
#     # We provide the Auditor with the original goal and the facts to compare against the draft
#     audit_context = (
#         f"--- THE ANCHORED GOAL ---\n{state.get('ticket_text')}\n\n"
#         f"--- RESEARCH FACTS (GROUND TRUTH) ---\n{state.get('research_facts')}\n\n"
#         f"--- PROPOSED EMAIL DRAFT ---\n{state.get('ai_response')}\n\n"
#         f"--- USER CONTEXT ---\nTier: {state.get('customer_tier')}"
#     )

#     # Use a specific thread_id for the audit turn
#     config = {"configurable": {"thread_id": f"audit_{state.get('user_id', 'temp')}"}}

#     # Invoke the auditor
#     result = policy_checker_agent.invoke(
#         {"messages": [("user", audit_context)]},
#         config
#     )

#     # Because we used with_structured_output, result is a PolicyGrade object (if using direct LLM)
#     # If using create_react_agent, we access the structured output from the last message
#     # Note: For simple structured auditing, calling the LLM directly is often cleaner than a ReAct agent

#     # Alternative direct call for higher reliability in structured nodes:
#     data = structured_llm.invoke(audit_context)

#     return {
#         "policy_grade": data.grade,
#         "policy_feedback": data.feedback if data.grade == "FAIL" else None
#     }


def policy_checker_node(state: UDAHubState):
    data: PolicyGrade = qa_chain.invoke(
        {
            "ticket_text": state.get("ticket_text"),
            "research_facts": state.get("research_facts"),
            "ai_response": state.get("ai_response"),
        }
    )

    out = {
        "policy_grade": data.grade,
        "policy_feedback": data.feedback if data.grade == "FAIL" else None,
    }

    node_log(
        "policy_checker",
        policy_grade=out.get("policy_grade"),
        policy_feedback=(out.get("policy_feedback")[:120] if out.get("policy_feedback") else None),
    )

    return out


if __name__ == "__main__":
    print("🛡️ --- Running Policy Checker (Auditor) Tests ---\n" + "="*60)

    # TEST 1: The "Perfect" Draft (Should PASS)
    state_pass: UDAHubState = {
        "ticket_text": "I want a refund for my yoga session.",
        "research_facts": "POLICY: Refunds only allowed 24h before session. USER DATA: Alice booked Yoga at 5pm tomorrow.",
        "ai_response": "Hi Alice, I see your Yoga session is tomorrow at 5pm. Since that's more than 24h away, I can process that refund for you!",
        "customer_tier": "regular"
    }

    # TEST 2: The "Liar" Draft (Should FAIL - Hallucination)
    # Researcher says NO refund, but Drafter says YES.
    state_hallucination: UDAHubState = {
        "ticket_text": "Refund for my missed class.",
        "research_facts": "POLICY: No refunds for missed classes. USER DATA: Class was yesterday.",
        "ai_response": "I'm so sorry you missed it! I've gone ahead and issued a full refund to your card.",
        "customer_tier": "regular"
    }

    # TEST 3: The "Leaky" Draft (Should FAIL - Security)
    # Draft includes internal database IDs.
    state_security: UDAHubState = {
        "ticket_text": "What is my account status?",
        "research_facts": "USER: Frank Ocean, ID: e6376d, Tier: Premium.",
        "ai_response": "Hello Frank, your internal system ID is e6376d and you are a Premium member.",
        "customer_tier": "premium"
    }

    # TEST 4: The "VIP Pretender" (Should FAIL - Tier Compliance)
    # Draft offers VIP perks to a regular user.
    state_tier_mismatch: UDAHubState = {
        "ticket_text": "Can I get lounge access?",
        "research_facts": "POLICY: Lounge is for VIP only. USER: Bob is 'regular' tier.",
        "ai_response": "Hey Bob! As a valued member, I've granted you one-time access to the VIP lounge.",
        "customer_tier": "regular"
    }

    tests = [
        ("Test 1 (Valid)", state_pass),
        ("Test 2 (Hallucination)", state_hallucination),
        ("Test 3 (Security Leak)", state_security),
        ("Test 4 (Tier Violation)", state_tier_mismatch)
    ]

    for name, state in tests:
        print(f"\n▶️ {name}")
        result = policy_checker_node(state)
        grade = result.get("policy_grade")
        feedback = result.get("policy_feedback")
        
        color = "✅" if grade == "PASS" else "❌"
        print(f"Result: {color} {grade}")
        if feedback:
            print(f"Feedback: {feedback}")
    
    print("\n" + "="*60 + "\n✅ Policy Audit Suite Complete")
