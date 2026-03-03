from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")


def now_kst_iso() -> str:
    return datetime.now(tz=KST).isoformat()
