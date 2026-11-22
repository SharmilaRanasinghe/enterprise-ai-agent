
import uuid

import httpx
import streamlit as st


API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Enterprise AI Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("Enterprise AI Agent")
st.caption("LangGraph + FastAPI + Gemini")


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

    for step in st.session_state.agent_steps:
        st.success(step)


# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agents are working..."):
            try:
                with httpx.Client(timeout=90.0) as client:
                    response = client.post(
                        f"{API_URL}/api/chat",
                        json={
                            "message": prompt,
                            "session_id": (
                                st.session_state.session_id
                            ),
                        },
                    )

                    response.raise_for_status()
                    result = response.json()

                answer = result["answer"]
                st.markdown(answer)


                sources = result.get("sources", [])

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
                                f"File: {source['source']} | "
                                f"Department: {source['department']}"
                            )

                            st.write(source["text"])
                            st.divider()


                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                st.session_state.agent_steps = result[
                    "agent_steps"
                ]

            except httpx.HTTPError as exc:
                st.error(
                    "Unable to contact the AI backend. "
                    "Check that FastAPI is running."
                )
                st.caption(str(exc))

    st.rerun()
