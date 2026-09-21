from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_state: str  # IDLE | RUNNING | FINISHED | ERROR
    current_step: int
    max_steps: int
    system_prompt: str
    next_step_prompt: str
    intent: str  # plan_trip | quick_query | chitchat
