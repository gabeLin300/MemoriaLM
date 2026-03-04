from dataclasses import dataclass
import re
from typing import Optional

from fastapi import Header, HTTPException, Request


@dataclass
class User:
    user_id: str
    email: str


_USER_ID_PATTERN = re.compile(r"[A-Za-z0-9._@-]+$")


def _validate_user_id(value: str) -> str:
    if not _USER_ID_PATTERN.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid authenticated user id")
    return value


def _extract_user_id_from_request(request: Request) -> Optional[str]:
    # These headers are common in reverse-proxy / OAuth fronted deployments.
    header_candidates = [
        request.headers.get("x-user-id"),
        request.headers.get("x-hf-username"),
        request.headers.get("x-forwarded-user"),
        request.headers.get("remote-user"),
    ]
    for candidate in header_candidates:
        if candidate and candidate.strip():
            return candidate.strip()
    return None


def get_current_user(
    request: Request,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_user_email: Optional[str] = Header(default=None, alias="X-User-Email"),
) -> User:
    user_id = (x_user_id or "").strip() or _extract_user_id_from_request(request)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide authenticated user context.",
        )

    user_id = _validate_user_id(user_id)
    email = (x_user_email or f"{user_id}@local").strip()
    return User(user_id=user_id, email=email)


def enforce_user_match(current_user: User, supplied_user_id: Optional[str]) -> None:
    supplied = (supplied_user_id or "").strip()
    if supplied and supplied != current_user.user_id:
        raise HTTPException(
            status_code=403,
            detail="Authenticated user does not match supplied user_id",
        )
