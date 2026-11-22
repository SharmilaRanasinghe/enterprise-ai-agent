
import logging
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from google.genai.errors import ServerError

from backend.agents.graph import graph


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise AI Agent",
    description="LangGraph + Gemini AI Assistant",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=128)


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    intent: str
    trace_id: str
    agent_steps: list[str]


@app.get("/")
async def root():
    return {
        "application": "Enterprise AI Assistant",
        "status": "running",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    trace_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": request.session_id,
        },
        "run_name": "enterprise-chat",
        "tags": ["poc", "gemini"],
        "metadata": {
            "request_trace_id": trace_id,
        },
    }

    try:
        logger.info(
            "Chat started session=%s trace=%s",
            request.session_id,
            trace_id,
        )

        result = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(content=request.message)
                ],
                "intent": "",
            },
            config=config,
        )

        answer = result["messages"][-1].content

        return ChatResponse(
            answer=str(answer),
            session_id=request.session_id,
            intent=result["intent"],
            trace_id=trace_id,
            agent_steps=[
                "Supervisor: classified request",
                "Response Agent: generated answer",
            ],
        )

    
    except ServerError as exc:
        logger.warning(
            "Gemini service unavailable trace=%s: %s",
            trace_id,
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "error": "LLM_SERVICE_UNAVAILABLE",
                "message": (
                    "Gemini is temporarily unavailable. "
                    "Please try again shortly."
                ),
                "trace_id": trace_id,
            },
        )


    except Exception:
        logger.exception("Chat failed trace=%s", trace_id)

        raise HTTPException(
            status_code=503,
            detail=(
                "The assistant is temporarily unavailable. "
                f"Reference: {trace_id}"
            ),
        )
