"""Send the recommendation.

Deliberately not routed through Alertmanager. Alerts model conditions that become
true and later become false, with grouping, deduplication and a resolved
notification. A weekly recommendation is none of those things -- it is a document
that only you can resolve, by acting or not. Routed as an alert it would "resolve"
when the deadline passed whether or not you did anything, and repeat intervals
would re-send it.

More simply: monitoring tells you the system is broken; the system itself tells
you what it decided. Pipeline health goes to Alertmanager. This does not.

There is no escalation and deliberately so. Confirming whether you acted needs
the authenticated `my-team` endpoint, which this system does not use, and picks
stay private until the deadline passes. So sends happen on a fixed schedule and
some are redundant -- which is far better than a reminder that fires because it
cannot tell you already did the thing.
"""

from __future__ import annotations

import json
import os

import httpx
import structlog

log = structlog.get_logger()

#: Any endpoint accepting a JSON POST: Discord, Slack, ntfy, Gotify. Left
#: unset, recommendations are logged instead, which is the right default for a
#: system that must never fail because a chat webhook rotated.
WEBHOOK_ENV = "FPL_WEBHOOK_URL"


def _payload(url: str, text: str, title: str) -> dict:
    """Shape the body for whichever service the URL points at."""
    if "discord" in url:
        # Discord caps message content at 2000 characters.
        body = f"**{title}**\n```\n{text[:1900]}\n```"
        return {"content": body}
    if "slack" in url or "hooks.slack" in url:
        return {"text": f"*{title}*\n```\n{text}\n```"}
    return {"title": title, "message": text}


def send(text: str, *, title: str, url: str | None = None) -> bool:
    """Deliver the recommendation. Returns whether it actually went anywhere.

    Never raises. A failed notification must not fail the job -- the
    recommendation is already durably in Postgres, and losing the message is a
    smaller problem than a CrashLoopBackOff that stops tomorrow's ingest.
    """
    url = url or os.environ.get(WEBHOOK_ENV)
    if not url:
        log.info("recommendation_ready", title=title, body=text)
        return False

    try:
        response = httpx.post(url, json=_payload(url, text, title), timeout=15.0)
        if response.status_code >= 400:
            log.warning("notify_rejected", status=response.status_code, body=response.text[:200])
            return False
    except httpx.HTTPError as exc:
        log.warning("notify_failed", error=str(exc))
        return False

    log.info("notified", title=title)
    return True
