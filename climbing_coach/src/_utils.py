"""Small helpers shared by the climbing_coach modules."""

from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
