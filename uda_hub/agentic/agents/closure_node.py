from langchain_core.messages import AIMessage
from uda_hub.agentic.tools.udahub_state import UDAHubState


def closer_node(state: UDAHubState):
    """
    Handles the final 'You're welcome' and sets status to CLOSED.
    """
    response = "You're very welcome! I'm glad I could help. Feel free to reach out if you need anything else. Have a great day!"

    return {
        "ai_response": response,
        "messages": [AIMessage(content=response)],
        "status": "closed",
    }
