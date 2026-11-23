
import json

from backend.main import sse_event, extract_answer
from langchain_core.messages import AIMessage


def test_sse_event_format():
    event = sse_event(
        "started",
        {"message": "Workflow started"},
    )

    assert event.startswith("event: started\n")
    assert "data: " in event
    assert event.endswith("\n\n")

    payload = event.split("data: ", 1)[1].strip()

    assert json.loads(payload) == {
        "message": "Workflow started"
    }


def test_extract_plain_text():
    message = AIMessage(
        content="Payment gateway incident resolved."
    )

    assert extract_answer(message) == (
        "Payment gateway incident resolved."
    )


def test_extract_structured_text():
    message = AIMessage(
        content=[
            {
                "type": "text",
                "text": "The downstream processor was slow.",
            }
        ]
    )

    assert extract_answer(message) == (
        "The downstream processor was slow."
    )
