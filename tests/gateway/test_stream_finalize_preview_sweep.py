"""Every finalize lane must clean up the previews it replaced.

Only ``_send_fallback_final`` and the fresh-final lane deleted stale previews.  A turn that
finalized by plain EDIT (or skipped a redundant finalize) left every extra preview message of
the segment on screen next to the finalized answer - the duplicated raw tail observed on
2026-09-14.  These tests pin the sweep and, just as importantly, the two cases where deleting
would DESTROY delivered content.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

CURSOR = "\u2589"  # real streaming cursor, escaped to keep this file ASCII


def _make_adapter():
    from gateway.platforms.base import BasePlatformAdapter

    cls = type("SweepAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 2000})
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    adapter.delete_message = AsyncMock(return_value=True)
    adapter.send = AsyncMock(
        side_effect=lambda **kw: SimpleNamespace(success=True, message_id="fresh",
                                                 continuation_message_ids=()))
    adapter.edit_message = AsyncMock(
        side_effect=lambda **kw: SimpleNamespace(success=True, message_id=kw["message_id"],
                                                 continuation_message_ids=()))
    return adapter


def _consumer(adapter):
    cfg = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor=CURSOR)
    return GatewayStreamConsumer(adapter, "chat_sweep", cfg)


def _deleted(adapter):
    return {str(call.args[1]) for call in adapter.delete_message.call_args_list}


@pytest.mark.asyncio
async def test_sweep_deletes_the_orphan_preview_of_this_segment():
    adapter = _make_adapter()
    c = _consumer(adapter)
    c._message_id = "final"
    c._segment_preview_message_ids = {"final", "orphan"}
    c._preview_message_ids = {"final", "orphan"}

    await c._sweep_orphan_previews()

    assert _deleted(adapter) == {"orphan"}
    # The delivered message is never touched, and the swept id is forgotten so a second
    # finalize lane cannot re-delete (or re-report) it.
    assert c._segment_preview_message_ids == {"final"}
    assert c._preview_message_ids == {"final"}


@pytest.mark.asyncio
async def test_sweep_is_idempotent():
    adapter = _make_adapter()
    c = _consumer(adapter)
    c._message_id = "final"
    c._segment_preview_message_ids = {"final", "orphan"}
    c._preview_message_ids = {"final", "orphan"}

    await c._sweep_orphan_previews()
    await c._sweep_orphan_previews()

    assert adapter.delete_message.await_count == 1


@pytest.mark.asyncio
async def test_sweep_never_deletes_a_sealed_head_after_a_split():
    """After an overflow split the earlier ids hold delivered chunks, not previews."""
    adapter = _make_adapter()
    c = _consumer(adapter)
    c._message_id = "tail"
    c._turn_split_delivery = True
    c._segment_preview_message_ids = {"head", "tail"}
    c._preview_message_ids = {"head", "tail"}
    c._last_sent_text = "answer without a cursor"

    await c._sweep_orphan_previews()

    assert adapter.delete_message.await_count == 0
    assert c._segment_preview_message_ids == {"head", "tail"}


@pytest.mark.asyncio
async def test_split_with_a_stuck_cursor_is_stripped_not_deleted():
    adapter = _make_adapter()
    c = _consumer(adapter)
    c._message_id = "tail"
    c._turn_split_delivery = True
    c._segment_preview_message_ids = {"head", "tail"}
    c._accumulated = c._stream_ledger = "visible answer"
    c._last_sent_text = "visible answer" + CURSOR

    await c._sweep_orphan_previews()

    assert adapter.delete_message.await_count == 0
    adapter.edit_message.assert_awaited()
    assert CURSOR not in adapter.edit_message.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_sweep_never_deletes_a_finalized_preamble_from_an_earlier_segment():
    """``_preview_message_ids`` spans the whole turn: earlier segments are delivered text."""
    adapter = _make_adapter()
    c = _consumer(adapter)
    c._message_id = "final"
    c._preview_message_ids = {"preamble_of_segment_1", "final"}
    c._segment_preview_message_ids = {"final"}

    await c._sweep_orphan_previews()

    assert adapter.delete_message.await_count == 0
    assert "preamble_of_segment_1" in c._preview_message_ids


@pytest.mark.asyncio
async def test_edit_finalize_lane_sweeps_end_to_end():
    """The real drain loop: a plain edit finalize must leave no extra preview behind."""
    adapter = _make_adapter()
    c = _consumer(adapter)
    task = asyncio.create_task(c.run())
    c.on_delta("a short answer that finalizes through the plain edit lane\n")
    await asyncio.sleep(0.05)
    # An extra preview message of this segment exists on screen (adapter-side split,
    # relay re-post, retry - the sweep must not care how it got there).
    c._segment_preview_message_ids.add("stray")
    c._preview_message_ids.add("stray")
    c.finish()
    await asyncio.wait_for(task, timeout=10)

    assert "stray" in _deleted(adapter)
