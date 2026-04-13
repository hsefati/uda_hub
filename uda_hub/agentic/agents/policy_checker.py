import os
import json
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from uda_hub.agentic.tools.udahub_state import UDAHubState

load_dotenv()


# ==========================================
# 2. AGENT INSTANCE
# ==========================================
llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0.0, # CRITICAL: 0.0 for strict logic and compliance
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("VOCAREUM_API_KEY"),
)

# We use a system prompt that enforces a JSON output structure
policy_checker_agent = create_agent(
    model=llm,
    tools=[], 
    system_prompt=(
        "You are the UDA-Hub Policy Compliance Officer. Your job is to audit the drafted response.\n\n"
        "COMPARISON STEPS:\n"
        "1. Compare the 'Proposed Email' against the 'Research Facts'.\n"
        "2. Check for Hallucinations: Did the drafter promise a refund or action NOT supported by the facts?\n"
        "3. Check for Omissions: Did the drafter forget a critical instruction from the policy?\n\n"
        "OUTPUT REQUIREMENT:\n"
        "You must output your final decision in valid JSON format with these exact keys:\n"
        "{\n"
        "  \"grade\": \"PASS\" or \"FAIL\",\n"
        "  \"feedback\": \"Explanation of why it failed or an empty string if it passed.\"\n"
        "}"
    ),
    name="policy_checker"
)

# ==========================================
# 3. LANGGRAPH NODE
# ==========================================
def policy_checker_node(state: UDAHubState):
    """
    Audits the draft and determines if the graph needs to loop back or finish.
    """
    context = (
        f"--- RESEARCH FACTS ---\n{state.get('research_facts')}\n\n"
        f"--- PROPOSED EMAIL ---\n{state.get('ai_response')}"
    )

    config = {"configurable": {"thread_id": "qa_audit"}}
    result = policy_checker_agent.invoke(
        {"messages": [("user", context)]}, 
        config
    )

    # Parse the JSON output from the agent
    content = result["messages"][-1].content
    try:
        # We strip potential markdown code blocks if the LLM includes them
        clean_json = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        
        return {
            "policy_grade": data.get("grade", "FAIL"),
            "policy_feedback": data.get("feedback", "")
        }
    except Exception as e:
        # If parsing fails, we default to FAIL to be safe
        return {
            "policy_grade": "FAIL",
            "policy_feedback": f"System Error: Could not parse QA output. Raw output: {content}"
        }

# ==========================================
# 4. TEST SCRIPT (The Hallucination Catch)
# ==========================================
if __name__ == "__main__":
    print("--- TESTING POLICY CHECKER (Catching a Hallucination) ---")

    # Scenario: The Researcher said NO REFUND, but a "too helpful" Drafter promised one anyway.
    mock_facts = "Policy: All sales for Christ the Redeemer are final. No cash refunds. Credit only."
    hallucinated_draft = "I understand you can't make it! I've gone ahead and processed a full refund to your credit card."

    test_state: UDAHubState = {
        "research_facts": mock_facts,
        "ai_response": hallucinated_draft,
        "policy_grade": None,
        "policy_feedback": None
    }

    result = policy_checker_node(test_state)

    print(f"Grade:    {result['policy_grade']}")
    print(f"Feedback: {result['policy_feedback']}")
    
    # Expected: FAIL - because the draft contradicts the "No cash refunds" fact.