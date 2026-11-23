
import secrets
from dataclasses import dataclass

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from backend.config import settings


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


@dataclass(frozen=True)
class CurrentUser:
    user_id: str
    role: str


async def get_current_user(
    api_key: str | None = Depends(api_key_header),
) -> CurrentUser:

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is required",
        )

    users = [
        (
            settings.demo_viewer_api_key,
            CurrentUser("demo-viewer", "viewer"),
        ),
        (
            settings.demo_engineer_api_key,
            CurrentUser("demo-engineer", "engineer"),
        ),
        (
            settings.demo_admin_api_key,
            CurrentUser("demo-admin", "admin"),
        ),
    ]

    for expected_key, user in users:
        if expected_key and secrets.compare_digest(
            api_key, expected_key
        ):
            return user

    raise HTTPException(
        status_code=401,
        detail="Invalid API key",
    )
