# reset_udahub.py
import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from langchain_core.messages import (
    SystemMessage,
    HumanMessage, 
)
from langgraph.graph.state import CompiledStateGraph


Base = declarative_base()

def reset_db(db_path: str, echo: bool = True):
    """Drops the existing udahub.db file and recreates all tables."""

    # Remove the file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed existing {db_path}")

    # Create a new engine and recreate tables
    engine = create_engine(f"sqlite:///{db_path}", echo=echo)
    Base.metadata.create_all(engine)
    print(f"✅ Recreated {db_path} with fresh schema")


@contextmanager
def get_session(engine: Engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def model_to_dict(instance):
    """Convert a SQLAlchemy model instance to a dictionary."""
    return {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }

# def chat_interface(agent:CompiledStateGraph, ticket_id:str):
#     is_first_iteration = False
#     messages = [SystemMessage(content = f"ThreadId: {ticket_id}")]
#     while True:
#         user_input = input("User: ")
#         print("User:", user_input)
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Assistant: Goodbye!")
#             break
#         messages = [HumanMessage(content=user_input)]
#         if is_first_iteration:
#             messages.append(HumanMessage(content=user_input))
#         trigger = {
#             "messages": messages
#         }
#         config = {
#             "configurable": {
#                 "thread_id": ticket_id,
#             }
#         }
        
#         result = agent.invoke(input=trigger, config=config)
#         print("Assistant:", result["messages"][-1].content)
#         is_first_iteration = False

def chat_interface(agent: CompiledStateGraph, ticket_id: str):
    print(f"--- Session Started (Thread: {ticket_id}) ---")
    
    # Optional: Send an initial setup if it's a brand new thread
    # Otherwise, just start the loop.
    
    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Assistant: Goodbye!")
            break

        # 1. We only send the LATEST message. 
        # LangGraph's 'add_messages' reducer will append this to the history.
        trigger = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        config = {
            "configurable": {
                "thread_id": ticket_id,
            }
        }
        
        # 2. Invoke the agent
        result = agent.invoke(input=trigger, config=config)
        
        # 3. Access the last message in the updated state
        # Because result['messages'] contains the WHOLE history now, 
        # index -1 is the AI's latest response.
        if "messages" in result and result["messages"]:
            print("Assistant:", result["messages"][-1].content)
        else:
            # Fallback if your graph returned ai_response but didn't update messages
            print("Assistant:", result.get("ai_response", "I'm processing your request."))