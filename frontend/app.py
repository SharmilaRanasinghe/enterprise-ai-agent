import json
import os
import uuid

import httpx
import streamlit as st


API_URL = "http://localhost:8000"


def stream_chat(message: str, session_id: str):
    """Stream and parse SSE events from FastAPI."""

    api_key = os.getenv("ENTERPRISE_API_KEY", "")

    headers = {
        "Accept": "text/event-stream",
    }

    if api_key:
        headers["X-API-Key"] = api_key

    timeout = httpx.Timeout(
        connect=10.0,
        read=180.0,
        write=10.0,
        pool=10.0,
    )

    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST",
            f"{API_URL}/api/chat/stream",
            json={
                "message": message,
                "session_id": session_id,
            },
            headers=headers,
        ) as response:

            response.raise_for_status()

            event_name = None
            data_lines = []

            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line[6:].strip()

                elif line.startswith("data:"):
                    data_lines.append(line[5:].strip())

                elif line == "":
                    if event_name and data_lines:
                        yield {
                            "event": event_name,
                            "data": json.loads(
                                "\n".join(data_lines)
                            ),
                        }

                    event_name = None
                    data_lines = []

            if event_name and data_lines:
                yield {
                    "event": event_name,
                    "data": json.loads(
                        "\n".join(data_lines)
                    ),
                }


st.set_page_config(
    page_title="Enterprise AI Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("Enterprise AI Agent")
st.caption("LangGraph + FastAPI + Gemini + LangSmith")


# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_steps" not in st.session_state:
    st.session_state.agent_steps = []


# Sidebar
with st.sidebar:
    st.header("Assistant Settings")

    st.write("Session ID")
    st.code(st.session_state.session_id)

    if st.button("New Conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.agent_steps = []
        st.rerun()

    st.divider()
    st.subheader("Agent Activity")

    # This placeholder can update during execution.
    sidebar_activity = st.empty()

    with sidebar_activity.container():
        for step in st.session_state.agent_steps:
            st.success(step)


# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander(
                f"Retrieved Sources ({len(message['sources'])})"
            ):
                for source in message["sources"]:
                    st.markdown(
                        f"**[{source['citation']}] "
                        f"{source['title']}**"
                    )
                    st.caption(
                        f"File: {source.get('source', '')} | "
                        f"Department: {source.get('department', '')}"
                    )


# Chat input
prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        final_result = None
        live_steps = []

        try:
            with st.status(
                "Agents are working...",
                expanded=True,
            ) as status:

                for event in stream_chat(
                    prompt,
                    st.session_state.session_id,
                ):
                    event_type = event["event"]
                    data = event["data"]

                    if event_type == "started":
                        st.write("🚀 Workflow started")

                    elif event_type == "agent_update":
                        agent = data.get("agent", "unknown")
                        duration = data.get(
                            "duration_ms", 0
                        )

                        step = (
                            f"{agent} completed "
                            f"({duration:.0f} ms)"
                        )

                        live_steps.append(step)

                        # Update progress in main chat.
                        st.write(f"✅ {step}")

                        # Update sidebar immediately.
                        with sidebar_activity.container():
                            for item in live_steps:
                                st.success(item)

                    elif event_type == "completed":
                        final_result = data

                        status.update(
                            label="Workflow completed",
                            state="complete",
                            expanded=False,
                        )

                    elif event_type == "error":
                        raise RuntimeError(
                            data.get(
                                "message",
                                "Agent execution failed",
                            )
                        )

                if final_result is None:
                    raise RuntimeError(
                        "Stream ended without completion"
                    )

            # Final answer
            st.markdown(final_result["answer"])

            # Sources
            sources = final_result.get("sources", [])

            if sources:
                with st.expander(
                    f"Retrieved Sources ({len(sources)})"
                ):
                    for source in sources:
                        st.markdown(
                            f"**[{source['citation']}] "
                            f"{source['title']}**"
                        )

                        st.caption(
                            f"File: {source.get('source', '')} | "
                            f"Department: "
                            f"{source.get('department', '')}"
                        )

            # Execution details
            with st.expander("Execution Details"):
                st.write(
                    "Intent:",
                    final_result.get("intent"),
                )
                st.write(
                    "Duration:",
                    final_result.get(
                        "total_duration_ms"
                    ),
                    "ms",
                )
                st.write(
                    "Trace ID:",
                    final_result.get("trace_id"),
                )

            # Save assistant message.
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_result["answer"],
                "sources": sources,
            })

            st.session_state.agent_steps = live_steps

        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            st.error(f"Request failed: {exc}")
