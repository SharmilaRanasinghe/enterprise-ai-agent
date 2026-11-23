from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

import logging
import uuid
import asyncio
import json
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
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
    sources: list[dict]


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

        
        last_message = result["messages"][-1]
        answer = last_message.text

        if not answer:
            raise ValueError("Gemini returned no text")

        source_docs = result.get("sources", [])[:10]

        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            intent=result["intent"],
            trace_id=trace_id,
            agent_steps=result.get("agent_steps", []),
            sources=[
                {
                    "citation": i,
                    "title": doc.get("title", "Untitled"),
                    "source": doc.get("source", "Unknown"),
                    "department": doc.get("department", ""),
                    "text": doc.get("text", ""),
                    "score": doc.get("score", 0),
                }
                for i, doc in enumerate(
                    source_docs,
                    start=1,
                )
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


def sse_event(event_name: str, data: dict) -> str:
    """Encode one Server-Sent Event."""
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(data, default=str)}\n\n"
    )


def extract_answer(message) -> str:
    """Extract plain text from Gemini AIMessage."""
    text = getattr(message, "text", None)

    if callable(text):
        text = text()

    if isinstance(text, str) and text:
        return text

    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    return ""


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, request: Request):
    trace_id = str(uuid.uuid4())

    async def event_generator():
        started = time.perf_counter()
        last_update = started

        intent = "unknown"
        sources = []
        agent_steps = []
        answer = ""

        config = {
            "configurable": {
                "thread_id": payload.session_id,
            },
            "run_id": uuid.UUID(trace_id),
            "run_name": "enterprise-chat-stream",
            "tags": ["milestone-4", "streaming"],
            "metadata": {
                "session_id": payload.session_id,
                "trace_id": trace_id,
            },
        }

        yield sse_event("started", {
            "trace_id": trace_id,
            "session_id": payload.session_id,
            "message": "Agent workflow started",
        })

        try:
            async for update in graph.astream(
                {
                    "messages": [
                        HumanMessage(content=payload.message)
                    ],
                },
                config=config,
                stream_mode="updates",
            ):
                if await request.is_disconnected():
                    logger.info(
                        "Client disconnected trace=%s",
                        trace_id,
                    )
                    return

                now = time.perf_counter()

                for node_name, node_output in update.items():
                    if not isinstance(node_output, dict):
                        continue

                    duration_ms = round(
                        (now - last_update) * 1000,
                        2,
                    )
                    last_update = now

                    if "intent" in node_output:
                        intent = node_output["intent"]

                    if "sources" in node_output:
                        sources = node_output["sources"]

                    if "agent_steps" in node_output:
                        agent_steps = node_output["agent_steps"]

                    if node_name == "response":
                        messages = node_output.get(
                            "messages", []
                        )

                        if messages:
                            answer = extract_answer(
                                messages[-1]
                            )

                    yield sse_event("agent_update", {
                        "agent": node_name,
                        "duration_ms": duration_ms,
                        "intent": intent,
                        "source_count": len(sources),
                        "research_step": node_output.get(
                            "research_step"
                        ),
                        "message": (
                            agent_steps[-1]
                            if agent_steps
                            else f"{node_name} completed"
                        ),
                    })

            if not answer:
                raise RuntimeError(
                    "Workflow completed without an answer"
                )

            yield sse_event("completed", {
                "answer": answer,
                "session_id": payload.session_id,
                "intent": intent,
                "trace_id": trace_id,
                "agent_steps": agent_steps,
                "sources": [
                    {
                        "citation": i,
                        "title": doc.get(
                            "title", "Untitled"
                        ),
                        "source": doc.get(
                            "source", "Unknown"
                        ),
                        "department": doc.get(
                            "department", ""
                        ),
                        "score": doc.get("score", 0),
                    }
                    for i, doc in enumerate(
                        sources[:10], start=1
                    )
                ],
                "total_duration_ms": round(
                    (time.perf_counter() - started)
                    * 1000,
                    2,
                ),
            })

        except asyncio.CancelledError:
            logger.info(
                "Stream cancelled trace=%s",
                trace_id,
            )
            raise

        except Exception:
            logger.exception(
                "Streaming chat failed trace=%s",
                trace_id,
            )

            yield sse_event("error", {
                "trace_id": trace_id,
                "message": (
                    "The assistant could not complete "
                    "this request. Please retry."
                ),
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
