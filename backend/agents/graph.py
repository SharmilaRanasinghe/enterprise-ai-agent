
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from backend.config import settings


# Shared state
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str


# Gemini model
llm = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    google_api_key=settings.google_api_key,
    temperature=0,
    timeout=30,
    max_retries=2,
)


# Supervisor Agent
async def supervisor_agent(state: AgentState):
    return {"intent": "general_chat"}


# Response Agent
async def response_agent(state: AgentState):
    system_message = SystemMessage(
        content=(
            "You are an enterprise AI assistant for a "
            "fictional banking organization. "
            "Answer clearly and professionally. "
            "Do not claim access to internal company "
            "documents until retrieval is implemented. "
            "Never invent document references."
        )
    )

    response = await llm.ainvoke(
        [system_message, *state["messages"]]
    )

    return {"messages": [response]}


# Build LangGraph
builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor_agent)
builder.add_node("response", response_agent)

builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "response")
builder.add_edge("response", END)


# Session memory
checkpointer = InMemorySaver()

graph = builder.compile(checkpointer=checkpointer)
