"""A recovered reply must render exactly like the live final send.

Regression (DASG, 2026-09-15): a restart landed mid-delivery. The boot sweep
found the ``attempting`` row and redelivered it through
``_redeliver_claimed_obligations``, which called ``adapter.send`` with metadata
carrying only ``thread_id``. The Discord adapter gates the rich closing-block
renderer on ``metadata["notify"]`` (``final_delivery``), so the recovered reply
skipped ``_split_rich_blocks`` entirely: the green RESULT box, the amber
``Feito:`` box and the red handoff box came back as raw markdown, blind-split
by size into "(1/5)" chunks.

Contract asserted here:
- every redelivery declares itself the turn's final response (``notify``)
- ``thread_id`` still travels with it (threads must not regress to the parent)
- a row without a thread still gets ``notify`` (DM / plain channel recovery)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as dl


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


def _runner_with_adapter():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running = True
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=MagicMock(success=True, error=None))
    runner._obligation_adapter = AsyncMock(return_value=adapter)
    runner._arm_flood_timers_for_waiting_rows = AsyncMock(return_value=None)
    return runner, adapter


def _row(**kw):
    row = {
        "obligation_id": "ob-1", "session_key": "agent:main:discord:channel:C1",
        "platform": "discord", "chat_id": "C1", "thread_id": "1532149266271703241",
        "content": "head\n\n------\n\n✅ **Done**\n\n- **TL;DR** — x",
        "needs_marker": False, "profile": "default", "attempts": 1,
    }
    row.update(kw)
    return row


@pytest.mark.asyncio
async def test_redelivery_declares_itself_a_final_delivery():
    """Without ``notify`` the adapter never reaches the rich-block renderer."""
    runner, adapter = _runner_with_adapter()

    assert await runner._redeliver_claimed_obligations([_row()]) == 1

    metadata = adapter.send.await_args.kwargs["metadata"]
    assert metadata.get("notify") is True, (
        "recovered reply sent without notify -> Discord skips _split_rich_blocks "
        "and the colored boxes come back as raw markdown"
    )


@pytest.mark.asyncio
async def test_redelivery_still_targets_the_original_thread():
    runner, adapter = _runner_with_adapter()

    await runner._redeliver_claimed_obligations([_row()])

    metadata = adapter.send.await_args.kwargs["metadata"]
    assert metadata["thread_id"] == "1532149266271703241"
    assert metadata["notify"] is True


@pytest.mark.asyncio
async def test_threadless_row_is_still_a_final_delivery():
    """A DM / plain-channel recovery used to send metadata=None outright."""
    runner, adapter = _runner_with_adapter()

    await runner._redeliver_claimed_obligations([_row(thread_id=None)])

    metadata = adapter.send.await_args.kwargs["metadata"]
    assert metadata is not None and metadata.get("notify") is True
    assert "thread_id" not in metadata
