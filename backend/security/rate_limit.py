
import asyncio
import time

from fastapi import Depends, HTTPException

from backend.security.auth import (
    CurrentUser,
    get_current_user,
)


CAPACITY = 10
REFILL_RATE = CAPACITY / 60.0

buckets = {}
lock = asyncio.Lock()


async def enforce_rate_limit(
    user: CurrentUser = Depends(get_current_user),
) -> CurrentUser:

    async with lock:
        now = time.monotonic()

        tokens, last_updated = buckets.get(
            user.user_id,
            (float(CAPACITY), now),
        )

        elapsed = now - last_updated

        tokens = min(
            float(CAPACITY),
            tokens + elapsed * REFILL_RATE,
        )

        if tokens < 1:
            retry_after = max(
                1,
                int((1 - tokens) / REFILL_RATE) + 1,
            )

            buckets[user.user_id] = (
                tokens,
                now,
            )

            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(retry_after)
                },
            )

        buckets[user.user_id] = (
            tokens - 1,
            now,
        )

    return user
