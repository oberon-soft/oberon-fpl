"""Send the recommendation.

Deliberately not routed through Alertmanager, even though Alertmanager already
sends email here. Alerts model conditions that become true and later become
false, with grouping, deduplication and a resolved notification. A weekly
recommendation is none of those: it is a document only you can resolve, by
acting or not. Routed as an alert it would "resolve" when the deadline passed
whether or not you did anything, `repeat_interval` would re-send it, and
`group_by: alertname` would merge this week's with last week's.

More simply: monitoring tells you the system is broken; the system tells you what
it decided. Pipeline health goes to Alertmanager. This does not, while sharing
its SMTP relay.

There is no escalation and deliberately so. Confirming whether you acted needs
the authenticated `my-team` endpoint, which this system does not use, and picks
stay private until a deadline passes. So sends happen on a fixed schedule and
some are redundant -- far better than a reminder that fires because it cannot
tell you already did the thing.
"""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage

import httpx
import structlog

log = structlog.get_logger()

#: Any endpoint accepting a JSON POST: Discord, Slack, ntfy, Gotify.
WEBHOOK_ENV = "FPL_WEBHOOK_URL"

SMTP_HOST_ENV = "FPL_SMTP_HOST"
SMTP_PORT_ENV = "FPL_SMTP_PORT"
SMTP_USER_ENV = "FPL_SMTP_USER"
SMTP_PASSWORD_ENV = "FPL_SMTP_PASSWORD"
EMAIL_FROM_ENV = "FPL_EMAIL_FROM"
EMAIL_TO_ENV = "FPL_EMAIL_TO"


def _webhook_payload(url: str, text: str, title: str) -> dict:
    """Shape the body for whichever service the URL points at."""
    if "discord" in url:
        # Discord caps message content at 2000 characters.
        return {"content": f"**{title}**\n```\n{text[:1900]}\n```"}
    if "slack" in url:
        return {"text": f"*{title}*\n```\n{text}\n```"}
    return {"title": title, "message": text}


def send_webhook(text: str, *, title: str, url: str) -> bool:
    try:
        response = httpx.post(url, json=_webhook_payload(url, text, title), timeout=15.0)
        if response.status_code >= 400:
            log.warning("webhook_rejected", status=response.status_code)
            return False
    except httpx.HTTPError as exc:
        log.warning("webhook_failed", error=str(exc))
        return False
    return True


def _build_email(text: str, title: str, sender: str, recipient: str) -> EmailMessage:
    """Plain text plus a monospace HTML alternative.

    The squad table is column-aligned, and mail clients render plain text in a
    proportional font by default, which turns it into a jumble. The `<pre>`
    alternative is the difference between a readable recommendation and one you
    have to squint at on a phone.
    """
    message = EmailMessage()
    message["Subject"] = title
    message["From"] = sender
    message["To"] = recipient
    message.set_content(text)

    escaped = (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    message.add_alternative(
        "<html><body>"
        '<pre style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
        'font-size:13px;line-height:1.45">'
        f"{escaped}"
        "</pre></body></html>",
        subtype="html",
    )
    return message


def send_email(text: str, *, title: str) -> bool:
    host = os.environ.get(SMTP_HOST_ENV)
    recipient = os.environ.get(EMAIL_TO_ENV)
    if not host or not recipient:
        return False

    sender = os.environ.get(EMAIL_FROM_ENV, recipient)
    port = int(os.environ.get(SMTP_PORT_ENV, "587"))
    user = os.environ.get(SMTP_USER_ENV)
    password = os.environ.get(SMTP_PASSWORD_ENV)

    try:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls()
            if user and password:
                smtp.login(user, password)
            smtp.send_message(_build_email(text, title, sender, recipient))
    except (smtplib.SMTPException, OSError) as exc:
        log.warning("email_failed", error=str(exc))
        return False

    log.info("emailed", to=recipient, title=title)
    return True


def send(text: str, *, title: str, url: str | None = None) -> bool:
    """Deliver the recommendation. Returns whether it reached anywhere.

    Tries email first, then a webhook, then logs. Never raises: the
    recommendation is already durable in Postgres before this is called, and
    losing a message is a far smaller problem than a CrashLoopBackOff that stops
    tomorrow's ingest -- the one job whose data cannot be recovered.
    """
    delivered = send_email(text, title=title)

    url = url or os.environ.get(WEBHOOK_ENV)
    if url:
        delivered = send_webhook(text, title=title, url=url) or delivered

    if not delivered:
        log.info("recommendation_ready", title=title, body=text)
    return delivered
