from langchain_core.messages import AIMessage

from uda_hub.agentic.tools.udahub_state import UDAHubState


def greeter_node(state: UDAHubState):
    """
    Handles pleasantries and simple hellos.
    """
    # We can use the user_name if the Enricher found it, otherwise generic
    name = state.get("user_name", "there")
    response = f"Hi {name}! I'm the UDA-Hub assistant. How can I help you with your CultPass today?"

    return {
        "ai_response": response,
        "messages": [AIMessage(content=response)],
        "status": "pending_user",
    }
