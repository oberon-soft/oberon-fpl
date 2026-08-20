"""Recommendation assembly, rendering, and delivery."""

from __future__ import annotations

import httpx
import pytest
import respx

from fpl import notify
from fpl.entry import Squad
from fpl.freshness import Readiness, Verdict
from fpl.phase import Phase
from fpl.optimise import Chip, Rules, solve
from fpl.recommend import MARGIN_OF_INDIFFERENCE, build, render

from tests.test_optimise import HORIZON, make_pool

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
TEAMS = {i: f"T{i:02d}" for i in range(1, 21)}


@pytest.fixture
def rules(bootstrap) -> Rules:
    return Rules.from_bootstrap(bootstrap)


def ready() -> Readiness:
    return Readiness(Verdict.READY, Phase.PRE_DEADLINE, {})


def build_for(pool, rules, current=None, **kw):
    return build(
        pool, rules, event=1, horizon=HORIZON, kind="final", readiness=ready(),
        positions=POSITIONS, teams=TEAMS, current=current, **kw
    )


def held_squad(pool, rules, free_transfers=1) -> Squad:
    base = solve(pool, rules, horizon=HORIZON)
    return Squad(
        entry_id=1, event=0, element_ids=[p.element_id for p in base.squad],
        captain=None, vice_captain=None, bank=0,
        value=sum(p.now_cost for p in base.squad), free_transfers=free_transfers,
    )


def test_from_scratch_when_no_squad_is_confirmed(rules: Rules):
    rec = build_for(make_pool(30), rules)
    assert rec.current is None
    assert any("from scratch" in n for n in rec.notes)
    assert len(rec.solution.squad) == rules.squad_size


def test_holds_when_the_squad_is_already_optimal(rules: Rules):
    pool = make_pool(31)
    rec = build_for(pool, rules, current=held_squad(pool, rules))
    assert rec.is_hold
    assert "HOLD" in render(rec)


def test_recommends_a_transfer_when_one_clearly_helps(rules: Rules):
    pool = make_pool(32)
    current = held_squad(pool, rules)
    base = solve(pool, rules, horizon=HORIZON)
    weakest = min(base.starting, key=lambda p: p.ep_horizon)

    from fpl.project import Projection

    upgrade = Projection(
        code=555, element_id=555, element_type=weakest.element_type, web_name="Upgrade",
        team_id=weakest.team_id, now_cost=weakest.now_cost,
        by_gameweek={gw: weakest.by_gameweek[gw] + 4.0 for gw in HORIZON},
    )
    rec = build_for(pool + [upgrade], rules, current=current)
    assert not rec.is_hold
    assert "TRANSFER" in render(rec)


def test_alternatives_are_reported_with_their_cost(rules: Rules):
    """A recommendation that names only one option invites more confidence than
    the model has earned."""
    pool = make_pool(33)
    current = held_squad(pool, rules)
    base = solve(pool, rules, horizon=HORIZON)
    weakest = min(base.starting, key=lambda p: p.ep_horizon)

    from fpl.project import Projection

    rivals = [
        Projection(
            code=560 + i, element_id=560 + i, element_type=weakest.element_type,
            web_name=f"Option{i}", team_id=weakest.team_id, now_cost=weakest.now_cost,
            by_gameweek={gw: weakest.by_gameweek[gw] + 4.0 - i * 0.05 for gw in HORIZON},
        )
        for i in range(3)
    ]
    rec = build_for(pool + rivals, rules, current=current)
    assert rec.alternatives
    # Alternatives are worse than the chosen move, so the delta is never positive.
    assert all(delta <= 1e-6 for _, delta in rec.alternatives)


def test_a_near_tie_is_flagged_as_a_coin_flip(rules: Rules):
    """When the margin sits inside model error the honest output says so rather
    than presenting one option as the answer."""
    pool = make_pool(34)
    current = held_squad(pool, rules)
    base = solve(pool, rules, horizon=HORIZON)
    weakest = min(base.starting, key=lambda p: p.ep_horizon)

    from fpl.project import Projection

    twins = [
        Projection(
            code=570 + i, element_id=570 + i, element_type=weakest.element_type,
            web_name=f"Twin{i}", team_id=weakest.team_id, now_cost=weakest.now_cost,
            by_gameweek={gw: weakest.by_gameweek[gw] + 4.0 - i * 0.001 for gw in HORIZON},
        )
        for i in range(2)
    ]
    rec = build_for(pool + twins, rules, current=current)
    assert abs(rec.alternatives[0][1]) < MARGIN_OF_INDIFFERENCE
    assert any("coin flip" in n for n in rec.notes)


def test_payload_is_queryable_not_prose(rules: Rules):
    """The database record must stay structured; the rendered text is separate."""
    rec = build_for(make_pool(35), rules)
    payload = rec.to_payload()
    assert payload["event"] == 1
    assert len(payload["squad"]) == rules.squad_size
    assert sum(1 for p in payload["squad"] if p["starting"]) == rules.starting_size
    assert set(payload["captains"]) == {str(gw) for gw in HORIZON}
    assert isinstance(payload["objective"], float)


def test_render_names_imputed_players(rules: Rules):
    """Rates inferred from price are a weaker claim and the output should say
    which players rest on one."""
    from fpl.project import Projection

    pool = make_pool(36)
    pool = [
        Projection(**{**p.__dict__, "imputed": True}) if i % 7 == 0 else p
        for i, p in enumerate(pool)
    ]
    rec = build_for(pool, rules)
    if any(p.imputed for p in rec.solution.squad):
        assert any("inferred from price" in n for n in rec.notes)


def test_blocked_verdict_is_visible_in_the_render(rules: Rules):
    rec = build(
        make_pool(37), rules, event=1, horizon=HORIZON, kind="plan",
        readiness=Readiness(Verdict.BLOCKED, Phase.PRE_DEADLINE, {}),
        positions=POSITIONS, teams=TEAMS,
    )
    assert "[BLOCKED]" in render(rec)


def test_chip_shown_in_the_header(rules: Rules):
    rec = build_for(make_pool(38), rules, chip=Chip.TRIPLE_CAPTAIN)
    assert "3xc" in render(rec)


# -- notification ---------------------------------------------------------


def test_without_any_channel_it_logs_rather_than_failing(monkeypatch):
    for var in ("FPL_SMTP_HOST", "FPL_EMAIL_TO", "FPL_WEBHOOK_URL"):
        monkeypatch.delenv(var, raising=False)
    assert notify.send("body", title="t", url=None) is False


def test_email_needs_both_host_and_recipient(monkeypatch):
    monkeypatch.setenv("FPL_SMTP_HOST", "smtp.test")
    monkeypatch.delenv("FPL_EMAIL_TO", raising=False)
    assert notify.send_email("body", title="t") is False


def test_email_is_sent_as_text_and_monospace_html():
    """The squad table is column-aligned. Mail clients default to a proportional
    font, which turns it into a jumble, so the HTML alternative wraps it in
    <pre> -- the difference between readable on a phone and not."""
    msg = notify._build_email("a  b\nc  d", "Title", "from@x", "to@y")
    assert msg["Subject"] == "Title"
    parts = {p.get_content_type() for p in msg.walk()}
    assert "text/plain" in parts and "text/html" in parts
    html = next(p for p in msg.walk() if p.get_content_type() == "text/html")
    body = html.get_content()
    assert "<pre" in body and "monospace" in body
    assert "white-space:pre" in body


def test_font_stack_starts_with_a_universally_supported_family():
    """Mail clients sanitise CSS and several drop an entire font-family
    declaration containing a token they do not recognise. A modern keyword
    like ui-monospace can therefore cost you the whole declaration -- and the
    symptom is subtle: numeric columns still align because they are
    space-padded, but rows with accented names shift, because those glyphs are
    not ASCII-width in a proportional font."""
    from fpl.notify import _MONOSPACE

    stack = _MONOSPACE.split("font-family:")[1].split(";")[0]
    assert stack.split(",")[0].strip("'\"") == "Courier New"
    assert "ui-monospace" not in _MONOSPACE
    assert stack.split(",")[-1].strip() == "monospace"


def test_email_escapes_html_in_the_body():
    """Player names and notes are interpolated; a stray angle bracket must not
    break the markup."""
    msg = notify._build_email("<script>x</script> & 5>3", "T", "f@x", "t@y")
    html = next(p for p in msg.walk() if p.get_content_type() == "text/html")
    body = html.get_content()
    assert "&lt;script&gt;" in body and "&amp;" in body
    assert "<script>" not in body


def test_smtp_failure_never_raises(monkeypatch):
    import smtplib

    monkeypatch.setenv("FPL_SMTP_HOST", "smtp.invalid.test")
    monkeypatch.setenv("FPL_EMAIL_TO", "to@y")

    def boom(*a, **k):
        raise smtplib.SMTPException("relay down")

    monkeypatch.setattr(smtplib, "SMTP", boom)
    assert notify.send_email("body", title="t") is False


@respx.mock
def test_webhook_receives_the_text(monkeypatch):
    monkeypatch.delenv("FPL_SMTP_HOST", raising=False)
    route = respx.post("https://example.test/hook").mock(return_value=httpx.Response(204))
    assert notify.send("the body", title="Title", url="https://example.test/hook") is True
    assert route.called


@respx.mock
def test_delivery_failure_never_raises(monkeypatch):
    monkeypatch.delenv("FPL_SMTP_HOST", raising=False)
    """The recommendation is already durable in Postgres. Losing the message is a
    smaller problem than a CrashLoopBackOff that stops tomorrow's ingest."""
    respx.post("https://example.test/hook").mock(side_effect=httpx.ConnectError("down"))
    assert notify.send("body", title="t", url="https://example.test/hook") is False


@respx.mock
def test_rejected_delivery_never_raises(monkeypatch):
    monkeypatch.delenv("FPL_SMTP_HOST", raising=False)
    respx.post("https://example.test/hook").mock(return_value=httpx.Response(403))
    assert notify.send("body", title="t", url="https://example.test/hook") is False


@respx.mock
def test_discord_payload_is_truncated_to_the_limit():
    """Discord rejects messages over 2000 characters, which a full squad render
    plus notes can approach."""
    captured = {}

    def record(request):
        captured["body"] = request.content
        return httpx.Response(204)

    respx.post("https://discord.com/api/webhooks/x").mock(side_effect=record)
    notify.send_webhook("x" * 5000, title="T", url="https://discord.com/api/webhooks/x")
    assert len(captured["body"]) < 2100


def test_style_attribute_is_not_broken_by_nested_quotes():
    """The style string is interpolated into a double-quoted HTML attribute, so
    a double-quoted font name terminates the attribute at the first quote and
    discards everything after it. Silent, and it costs the whole declaration."""
    from fpl.notify import _MONOSPACE, _build_email

    assert '"' not in _MONOSPACE

    msg = _build_email("a b", "T", "f@x", "t@y")
    html = next(p for p in msg.walk() if p.get_content_type() == "text/html")
    body = html.get_content()
    style = body.split('style="')[1].split('"')[0]
    assert style.endswith("margin:0"), f"attribute truncated: {style!r}"
    assert "Courier New" in style
