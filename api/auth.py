"""Admin auth for mutating and operational endpoints. Fails closed: if
ADMIN_TOKEN is unset, every admin request is rejected."""
import hmac
import os

from fastapi import Header, HTTPException

_ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()


def require_admin(x_admin_token: str = Header(default="")):
    if not _ADMIN_TOKEN or not hmac.compare_digest(x_admin_token.encode(), _ADMIN_TOKEN.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized")
