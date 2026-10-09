"""Phone push via ntfy.sh (no account). The topic comes from the NTFY_TOPIC environment
variable or from the project's .env file (gitignored; see .env.example). Messages are deliberately content-free: no balances, positions or
symbols - ntfy.sh topics are public to anyone who knows the name.

Never raises: a failed notification must not break a trading job.
    python -m farb.notify test
"""
from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
SERVER = "https://ntfy.sh"
BODY = "Alarm tetiklendi, dashboard'a bak."
log = logging.getLogger(__name__)


PRIORITY = {"min": 1, "low": 2, "default": 3, "high": 4, "urgent": 5}


def topic() -> str:
    t = os.environ.get("NTFY_TOPIC", "").strip()
    if t:
        return t
    try:
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("NTFY_TOPIC="):
                return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def send(title: str, body: str = BODY, priority: str = "high", tags: str = "warning") -> bool:
    """JSON publish (UTF-8 safe titles, e.g. Turkish characters)."""
    try:
        t = topic()
        if not t:
            return False
        payload = {"topic": t, "title": title, "message": body,
                   "priority": PRIORITY.get(priority, 3), "tags": [tags]}
        req = urllib.request.Request(SERVER, data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return 200 <= r.status < 300
    except Exception as e:  # noqa: BLE001 - notification is best effort
        log.warning("ntfy send failed: %r", e)
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        ok = send("Test bildirimi", "Test: bildirimler çalışıyor. Alarm olunca: Alarm tetiklendi, dashboard'a bak.",
                  priority="default", tags="white_check_mark")
        print("sent" if ok else "failed")
        sys.exit(0 if ok else 1)
    print(__doc__)
