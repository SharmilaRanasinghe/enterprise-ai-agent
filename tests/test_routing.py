
from backend.agents.graph import (
    classify_intent,
    create_research_queries,
    MAX_RESEARCH_STEPS,
)


def test_general_chat_routing():
    assert (
        classify_intent("What is a payment gateway?")
        == "general_chat"
    )


def test_knowledge_routing():
    assert (
        classify_intent("What caused incident INC-001?")
        == "knowledge_search"
    )


def test_research_routing():
    assert (
        classify_intent(
            "Investigate the payment gateway incident"
        )
        == "research"
    )


def test_research_step_limit():
    queries = create_research_queries(
        "Investigate the payment gateway incident"
    )

    assert len(queries) <= MAX_RESEARCH_STEPS
    assert len(queries) > 1
