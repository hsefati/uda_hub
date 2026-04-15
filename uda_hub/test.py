from dotenv import load_dotenv
from utils import chat_interface
load_dotenv()


from agentic.workflow import app as orchestrator


chat_interface(orchestrator, "1")

list(orchestrator.get_state_history(
    config = {
        "configurable": {
            "thread_id": "1",
        }
    }
))[0].values["messages"]