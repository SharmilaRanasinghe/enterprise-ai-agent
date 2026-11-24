
import pytest

from unittest.mock import AsyncMock, patch

from google.genai.errors import ServerError

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

from backend.agents.graph import invoke_with_fallback


def create_service_error():

    return ServerError(
        503,
        {
            "error": {
                "code": 503,
                "message": "Service unavailable",
                "status": "UNAVAILABLE",
            }
        },
    )


@pytest.mark.asyncio
async def test_primary_model_success():

    expected = AIMessage(
        content="Primary model response"
    )

    with (
        patch(
            "backend.agents.graph.primary_llm"
        ) as primary,
        patch(
            "backend.agents.graph.fallback_llm"
        ) as fallback,
    ):

        primary.ainvoke = AsyncMock(
            return_value=expected
        )

        fallback.ainvoke = AsyncMock()

        result = await invoke_with_fallback([
            HumanMessage(content="Hello")
        ])

        assert result.content == expected.content

        primary.ainvoke.assert_awaited_once()

        fallback.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_primary_model_failure_uses_fallback():

    expected = AIMessage(
        content="Fallback model response"
    )

    with (
        patch(
            "backend.agents.graph.primary_llm"
        ) as primary,
        patch(
            "backend.agents.graph.fallback_llm"
        ) as fallback,
    ):

        primary.ainvoke = AsyncMock(
            side_effect=create_service_error()
        )

        fallback.ainvoke = AsyncMock(
            return_value=expected
        )

        result = await invoke_with_fallback([
            HumanMessage(content="Hello")
        ])

        assert result.content == expected.content

        primary.ainvoke.assert_awaited_once()

        fallback.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_both_models_unavailable():

    with (
        patch(
            "backend.agents.graph.primary_llm"
        ) as primary,
        patch(
            "backend.agents.graph.fallback_llm"
        ) as fallback,
    ):

        primary.ainvoke = AsyncMock(
            side_effect=create_service_error()
        )

        fallback.ainvoke = AsyncMock(
            side_effect=create_service_error()
        )

        with pytest.raises(ServerError):

            await invoke_with_fallback([
                HumanMessage(content="Hello")
            ])

        primary.ainvoke.assert_awaited_once()

        fallback.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_both_models_unavailable():

    with (
        patch(
            "backend.agents.graph.primary_llm"
        ) as primary,
        patch(
            "backend.agents.graph.fallback_llm"
        ) as fallback,
    ):

        primary.ainvoke = AsyncMock(
            side_effect=create_service_error()
        )

        fallback.ainvoke = AsyncMock(
            side_effect=create_service_error()
        )

        with pytest.raises(ServerError):

            await invoke_with_fallback([
                HumanMessage(content="Hello")
            ])

        primary.ainvoke.assert_awaited_once()

        fallback.ainvoke.assert_awaited_once()
