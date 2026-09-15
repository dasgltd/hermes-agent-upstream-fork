"""Tests for the in-process delivery retry sweep (``sweep_retryable``).

``sweep_recoverable`` only claims rows whose owner process is DEAD, so it
recovers crashes and nothing else. When the PLATFORM rejects the final send
(a Discord 5xx blip) the gateway stays alive, keeps owning the row, and the
answer sat ``failed``/``attempts=0`` until an operator restarted the process.

Contract asserted here:
- a live-owned ``failed`` row is invisible to ``sweep_recoverable`` (the hole)
- ``sweep_retryable`` claims it, but only after its backoff has elapsed
- claiming flips it to ``attempting`` and spends exactly one attempt
- the two sweeps partition by owner liveness and can never claim one row
- retries always carry a marker (a 5xx can hide a delivery that landed)
- the attempts cap / stale cutoff abandon poison rows, budget shared
- pid reuse by a different process is not mistaken for us
- rows upstream already owns (reconnect ``send_path_degraded`` and flood
  refusals, handled by ``sweep_failed_for_runtime`` and the flood timers)
  are never claimed here: the three sweeps partition the space
- the gateway wiring (``_retry_failed_obligations``) feeds claimed rows
  through the shared ``_redeliver_claimed_obligations`` contract with the
  retry marker
"""

import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as dl


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


def _record(oid="ob-1", platform="discord", **kw):
    dl.record_obligation(
        obligation_id=oid,
        session_key=kw.get("session_key", "agent:main:discord:channel:C1"),
        platform=platform,
        chat_id=kw.get("chat_id", "C1"),
        thread_id=kw.get("thread_id", "1539072364405330062"),
        content=kw.get("content", "the final answer"),
    )


def _row(oid):
    with dl._connect() as conn:
        r = conn.execute(
            """SELECT state, attempts, owner_pid, owner_started_at
               FROM delivery_obligations WHERE obligation_id=?""",
            (oid,),
        ).fetchone()
    return None if r is None else {
        "state": r[0], "attempts": r[1],
        "owner_pid": r[2], "owner_started_at": r[3],
    }


def _set(oid, **cols):
    sets = ", ".join(f"{k}=?" for k in cols)
    with dl._connect() as conn:
        conn.execute(
            f"UPDATE delivery_obligations SET {sets} WHERE obligation_id=?",
            (*cols.values(), oid),
        )
        conn.commit()


def _fail_and_age(oid, seconds_ago):
    """Mark failed, then backdate updated_at to simulate elapsed backoff."""
    dl.mark_failed(oid, "503 Service Unavailable")
    _set(oid, updated_at=time.time() - seconds_ago)


# --------------------------------------------------------------------------
# The hole this sweep exists to close
# --------------------------------------------------------------------------

def test_crash_sweep_ignores_failed_row_owned_by_live_process():
    """Regression for the 2026-08-21 incident: a 503 on the final send left
    the answer stranded because the OWNING gateway never died."""
    _record()
    _fail_and_age("ob-1", 3600)

    assert dl.sweep_recoverable() == []          # crash path cannot see it
    assert _row("ob-1")["attempts"] == 0         # nothing ever retried it

    claimed = dl.sweep_retryable()               # retry path does
    assert [c["obligation_id"] for c in claimed] == ["ob-1"]


def test_claim_flips_to_attempting_and_spends_one_attempt():
    _record()
    _fail_and_age("ob-1", 3600)

    claimed = dl.sweep_retryable()

    assert claimed[0]["attempts"] == 1
    assert claimed[0]["content"] == "the final answer"
    assert claimed[0]["chat_id"] == "C1"
    row = _row("ob-1")
    assert row["state"] == "attempting"
    assert row["attempts"] == 1


def test_retry_always_carries_marker():
    """A 5xx can hide a message that actually landed - never a silent dup."""
    _record()
    _fail_and_age("ob-1", 3600)

    row = dl.sweep_retryable()[0]
    assert row["needs_marker"] is True
    # The shared redelivery loop reads the marker off the row (row.get("marker", RECOVERED_MARKER)).
    assert row["marker"] == dl.RETRY_MARKER
    assert row["profile"] == "default"


def test_retry_marker_does_not_claim_a_gateway_restart():
    """The crash wording would be a lie here: the gateway never went down."""
    assert "restart" not in dl.RETRY_MARKER.lower()
    assert "reconnect" not in dl.RETRY_MARKER.lower()
    assert "duplicate" in dl.RETRY_MARKER.lower()


def test_claim_clears_stale_error_so_a_crash_mid_resend_is_marked():
    _record()
    _fail_and_age("ob-1", 3600)
    assert dl.sweep_retryable()
    with dl._connect() as conn:
        err = conn.execute(
            "SELECT last_error FROM delivery_obligations WHERE obligation_id=?", ("ob-1",)
        ).fetchone()[0]
    assert err is None


# --------------------------------------------------------------------------
# Backoff
# --------------------------------------------------------------------------

def test_row_is_not_claimed_before_backoff_elapses():
    _record()
    dl.mark_failed("ob-1", "503")  # updated_at = now

    assert dl.sweep_retryable() == []
    assert _row("ob-1")["state"] == "failed"     # untouched, still retryable


def test_backoff_widens_with_each_attempt():
    """attempts=1 must wait longer than the first retry did."""
    _record()
    _fail_and_age("ob-1", dl.RETRY_BACKOFF_SECONDS[0] + 1)
    assert dl.sweep_retryable()                  # first retry fires

    # That retry failed too; the second backoff step is longer, so an age
    # that satisfied step 0 must NOT satisfy step 1.
    _fail_and_age("ob-1", dl.RETRY_BACKOFF_SECONDS[0] + 1)
    assert _row("ob-1")["attempts"] == 1
    assert dl.sweep_retryable() == []

    _fail_and_age("ob-1", dl.RETRY_BACKOFF_SECONDS[1] + 1)
    assert dl.sweep_retryable()


# --------------------------------------------------------------------------
# Bounds: poison rows cannot spin
# --------------------------------------------------------------------------

def test_attempts_cap_abandons_instead_of_retrying_forever():
    _record()
    _fail_and_age("ob-1", 3600)
    _set("ob-1", attempts=dl.MAX_ATTEMPTS)

    assert dl.sweep_retryable() == []
    assert _row("ob-1")["state"] == "abandoned"


def test_stale_cutoff_abandons_an_old_row():
    _record()
    _fail_and_age("ob-1", 3600)
    _set("ob-1", created_at=time.time() - dl.STALE_AFTER_SECONDS - 1)

    assert dl.sweep_retryable() == []
    assert _row("ob-1")["state"] == "abandoned"


def test_attempts_budget_is_shared_with_the_crash_path():
    """Two sweeps, one budget - a row cannot get 3 retries per sweep."""
    _record()
    _fail_and_age("ob-1", 3600)
    _set("ob-1", attempts=dl.MAX_ATTEMPTS - 1)

    assert len(dl.sweep_retryable()) == 1
    assert _row("ob-1")["attempts"] == dl.MAX_ATTEMPTS

    _fail_and_age("ob-1", 3600)
    assert dl.sweep_retryable() == []
    assert _row("ob-1")["state"] == "abandoned"


# --------------------------------------------------------------------------
# Ownership partition
# --------------------------------------------------------------------------

def test_row_owned_by_another_process_is_left_alone():
    _record()
    _fail_and_age("ob-1", 3600)
    _set("ob-1", owner_pid=os.getpid() + 424242)

    assert dl.sweep_retryable() == []
    assert _row("ob-1")["state"] == "failed"


def test_pid_reuse_by_a_different_process_is_not_mistaken_for_us():
    """Same pid number, different process start time - not ours to retry."""
    _record()
    _fail_and_age("ob-1", 3600)
    row = _row("ob-1")
    if row["owner_started_at"] is None:
        pytest.skip("process start time unavailable on this platform")
    _set("ob-1", owner_started_at=int(row["owner_started_at"]) + 99999)

    assert dl.sweep_retryable() == []
    assert _row("ob-1")["state"] == "failed"


def test_reconnect_and_flood_rows_are_left_to_upstream_sweeps():
    """Partition against upstream: ``send_path_degraded`` belongs to
    ``sweep_failed_for_runtime`` (adapter reconnect) and a flood refusal to
    the flood timer; claiming them here would race those paths."""
    _record("ob-degraded")
    dl.mark_failed("ob-degraded", "send_path_degraded")
    _set("ob-degraded", updated_at=time.time() - 3600)
    _record("ob-flood")
    dl.mark_failed("ob-flood", "flood_control:5")
    _set("ob-flood", updated_at=time.time() - 3600)
    _record("ob-raw-flood")
    dl.mark_failed("ob-raw-flood", "Flood control exceeded. Retry in 185 seconds")
    _set("ob-raw-flood", updated_at=time.time() - 3600)
    _record("ob-5xx")
    _fail_and_age("ob-5xx", 3600)

    claimed = dl.sweep_retryable()

    assert [c["obligation_id"] for c in claimed] == ["ob-5xx"]
    assert _row("ob-degraded")["state"] == "failed"
    assert _row("ob-flood")["state"] == "failed"
    assert _row("ob-raw-flood")["state"] == "failed"
    # ...and the runtime sweep still sees its own row, untouched by us.
    runtime = dl.sweep_failed_for_runtime("discord")
    assert {c["obligation_id"] for c in runtime} == {"ob-degraded", "ob-flood", "ob-raw-flood"}


def test_unreadable_start_time_fails_closed(monkeypatch):
    """Like ``sweep_failed_for_runtime``: pid alone cannot prove ownership."""
    _record()
    _fail_and_age("ob-1", 3600)
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), None))

    assert dl.sweep_retryable() == []
    assert _row("ob-1")["state"] == "failed"


def test_delivered_and_pending_rows_are_never_retried():
    _record("ob-pending")
    _record("ob-delivered")
    dl.mark_delivered("ob-delivered")
    _set("ob-pending", updated_at=time.time() - 3600)
    _set("ob-delivered", updated_at=time.time() - 3600)

    assert dl.sweep_retryable() == []


# --------------------------------------------------------------------------
# Platform filter and config gate
# --------------------------------------------------------------------------

def test_absent_platform_does_not_burn_an_attempt():
    _record(platform="discord")
    _fail_and_age("ob-1", 3600)

    assert dl.sweep_retryable(deliverable_platforms={"slack"}) == []
    row = _row("ob-1")
    assert row["attempts"] == 0        # budget preserved for a later pass
    assert row["state"] == "failed"

    assert dl.sweep_retryable(deliverable_platforms={"discord"})


def test_absent_profile_target_does_not_burn_an_attempt():
    """Multiplexed gateways: a connected bot must not spend a disconnected bot's budget."""
    dl.record_obligation(
        obligation_id="ob-alt", session_key="agent:main:discord:channel:C1", platform="discord",
        chat_id="C1", thread_id=None, content="x", adapter_profile="alt",
    )
    _fail_and_age("ob-alt", 3600)

    assert dl.sweep_retryable(
        deliverable_platforms={"discord"}, deliverable_targets={("discord", "default")}
    ) == []
    assert _row("ob-alt")["attempts"] == 0

    claimed = dl.sweep_retryable(
        deliverable_platforms={"discord"}, deliverable_targets={("discord", "alt")}
    )
    assert [c["profile"] for c in claimed] == ["alt"]


def test_retry_sweep_gate_defaults_on_and_honours_false():
    assert dl.retry_sweep_enabled({}) is True
    assert dl.retry_sweep_enabled({"gateway": {}}) is True
    assert dl.retry_sweep_enabled(
        {"gateway": {"delivery_retry_sweep": False}}
    ) is False
    assert dl.retry_sweep_enabled(
        {"gateway": {"delivery_retry_sweep": "off"}}
    ) is False


# --------------------------------------------------------------------------
# Gateway wiring: the watcher's one pass
# --------------------------------------------------------------------------

def _make_runner():
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.DISCORD: MagicMock()}
    runner._profile_adapters = {}
    runner._redeliver_claimed_obligations = AsyncMock(return_value=1)
    return runner


@pytest.mark.asyncio
async def test_retry_pass_feeds_claimed_rows_to_the_shared_redelivery_loop(monkeypatch):
    """One watcher pass: claim through ``sweep_retryable`` scoped to the
    connected adapters, then send through the same loop the crash and
    reconnect paths use (which honours the per-row marker)."""
    from gateway.run_startup import GatewayStartupMixin  # noqa: F401 - mixin methods live here

    _record()
    _fail_and_age("ob-1", 3600)
    runner = _make_runner()

    sent = await runner._retry_failed_obligations()

    assert sent == 1
    claimed = runner._redeliver_claimed_obligations.await_args.args[0]
    assert [c["obligation_id"] for c in claimed] == ["ob-1"]
    assert claimed[0]["marker"] == dl.RETRY_MARKER
    assert _row("ob-1")["state"] == "attempting"


@pytest.mark.asyncio
async def test_retry_pass_is_scoped_to_connected_adapters():
    _record(platform="telegram")
    _fail_and_age("ob-1", 3600)
    runner = _make_runner()  # only discord connected

    assert await runner._retry_failed_obligations() == 0
    runner._redeliver_claimed_obligations.assert_not_awaited()
    assert _row("ob-1")["attempts"] == 0


@pytest.mark.asyncio
async def test_retry_pass_respects_the_config_gate(monkeypatch):
    _record()
    _fail_and_age("ob-1", 3600)
    runner = _make_runner()
    monkeypatch.setattr(dl, "retry_sweep_enabled", lambda config=None: False)

    assert await runner._retry_failed_obligations() == 0
    assert _row("ob-1")["state"] == "failed"


def test_watcher_is_armed_with_the_other_post_reconnect_watchers():
    from gateway.run_startup import GatewayStartupMixin

    assert "_delivery_retry_watcher" in GatewayStartupMixin._POST_RECONNECT_WATCHERS
