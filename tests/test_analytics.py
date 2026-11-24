import pytest
from unittest.mock import AsyncMock, patch

from backend.agents.graph import analytics_agent
from backend.agents.graph import classify_intent
from backend.tools.analytics import analyze_transactions


def test_analytics_routing():
    assert classify_intent(
        "Show transaction statistics"
    ) == "analytics"

    assert classify_intent(
        "What is the payment success rate?"
    ) == "analytics"


def test_analytics_calculation():
    result = analyze_transactions.invoke({})

    assert result["total_transactions"] == 10
    assert result["successful_transactions"] == 7
    assert result["failed_transactions"] == 3
    assert result["success_rate_pct"] == 70.0
    assert result["total_successful_amount"] == 1370.5
    assert result["average_latency_ms"] == 1313.0


@pytest.mark.asyncio
async def test_viewer_analytics_access_denied():
    state = {
        "user_role": "viewer",
        "agent_steps": [],
    }

    with patch(
        "backend.agents.graph.analyze_transactions"
    ) as mock_tool:

        mock_tool.ainvoke = AsyncMock()

        result = await analytics_agent(state)

        assert result["analytics_result"]["error"] == (
            "Analytics access denied"
        )

        mock_tool.ainvoke.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["engineer", "admin"])
async def test_authorized_analytics_access(role):
    state = {
        "user_role": role,
        "agent_steps": [],
    }

    result = await analytics_agent(state)

    assert "error" not in result["analytics_result"]

    assert (
        result["analytics_result"]["total_transactions"]
        == 10
    )
