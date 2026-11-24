
import asyncio

import pytest

from unittest.mock import Mock

from fastapi import HTTPException

from backend.security.auth import CurrentUser

from backend.security import rate_limit


@pytest.fixture(autouse=True)
def reset_rate_limiter(monkeypatch):

    monkeypatch.setattr(
        rate_limit,
        "buckets",
        {},
    )

    monkeypatch.setattr(
        rate_limit,
        "lock",
        asyncio.Lock(),
    )

    # Use a controlled clock instead of real waiting.
    clock = Mock(return_value=1000.0)

    monkeypatch.setattr(
        rate_limit,
        "time",
        Mock(monotonic=clock),
    )

    return clock

@pytest.mark.asyncio
async def test_requests_within_rate_limit():

    user = CurrentUser(
        user_id="test-user-001",
        role="engineer",
    )

    for _ in range(rate_limit.CAPACITY):

        result = await rate_limit.enforce_rate_limit(
            user
        )

        assert result == user

@pytest.mark.asyncio
async def test_rate_limit_exceeded():

    user = CurrentUser(
        user_id="test-user-001",
        role="engineer",
    )

    for _ in range(rate_limit.CAPACITY):

        await rate_limit.enforce_rate_limit(user)

    with pytest.raises(HTTPException) as exc:

        await rate_limit.enforce_rate_limit(user)

    assert exc.value.status_code == 429

    assert exc.value.detail == "Rate limit exceeded"

    assert int(
        exc.value.headers["Retry-After"]
    ) > 0

@pytest.mark.asyncio
async def test_rate_limit_user_isolation():

    user1 = CurrentUser(
        user_id="test-user-001",
        role="engineer",
    )

    user2 = CurrentUser(
        user_id="test-user-002",
        role="engineer",
    )

    for _ in range(rate_limit.CAPACITY):

        await rate_limit.enforce_rate_limit(user1)

    with pytest.raises(HTTPException):

        await rate_limit.enforce_rate_limit(user1)

    result = await rate_limit.enforce_rate_limit(
        user2
    )

    assert result == user2

@pytest.mark.asyncio
async def test_token_refill(reset_rate_limiter):

    user = CurrentUser(
        user_id="test-user-001",
        role="engineer",
    )

    for _ in range(rate_limit.CAPACITY):

        await rate_limit.enforce_rate_limit(user)

    with pytest.raises(HTTPException):

        await rate_limit.enforce_rate_limit(user)

    # Simulate six seconds passing.
    reset_rate_limiter.return_value = 1006.0

    result = await rate_limit.enforce_rate_limit(
        user
    )

    assert result == user

    # The replenished token has been consumed.
    with pytest.raises(HTTPException):

        await rate_limit.enforce_rate_limit(user)

@pytest.mark.asyncio
async def test_concurrent_rate_limiting():

    user = CurrentUser(
        user_id="test-user-001",
        role="engineer",
    )

    results = await asyncio.gather(
        *[
            rate_limit.enforce_rate_limit(user)
            for _ in range(20)
        ],
        return_exceptions=True,
    )

    accepted = [
        result
        for result in results
        if isinstance(result, CurrentUser)
    ]

    rejected = [
        result
        for result in results
        if isinstance(result, HTTPException)
        and result.status_code == 429
    ]

    assert len(accepted) == 10

    assert len(rejected) == 10