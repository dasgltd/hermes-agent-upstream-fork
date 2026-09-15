"""No cursor-bearing preview may survive the turn.

An oversized edit is split by the adapter: the live message keeps chunk 1 and
chunk 2 goes out as a NEW message carrying the streaming cursor. When the turn
finalizes through a path that does not use fresh-final, nobody deletes that
continuation, so the tail of the answer would stay on screen twice - once raw with the cursor
and the "(2/2)" marker, once properly rendered (observed live on 2026-09-14).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

CURSOR = "\u2589"  # the real streaming cursor, escaped to keep this file ASCII
LIMIT = 2000


def _make_adapter():
    from gateway.platforms.base import BasePlatformAdapter

    cls = type("OrphanAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": LIMIT})
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    return adapter


def _wire(adapter):
    """Adapter with real message state: oversized edits split like Discord's."""
    screen = {}
    counter = {"n": 0}

    def _new_id():
        counter["n"] += 1
        return f"m{counter['n']}"

    async def fake_send(**kw):
        mid = _new_id()
        screen[mid] = kw["content"]
        return SimpleNamespace(success=True, message_id=mid,
                               continuation_message_ids=())

    async def fake_edit(**kw):
        mid = kw["message_id"]
        content = kw["content"]
        if len(content) <= LIMIT:
            screen[mid] = content
            return SimpleNamespace(success=True, message_id=mid,
                                   continuation_message_ids=())
        # Overflow split, exactly like _edit_overflow_split: chunk 1 edits the
        # live message, chunk 2..N go out as fresh messages.
        chunks = [content[i:i + LIMIT] for i in range(0, len(content), LIMIT)]
        screen[mid] = f"{chunks[0]} (1/{len(chunks)})"
        cont = []
        for i, chunk in enumerate(chunks[1:], start=2):
            new_id = _new_id()
            screen[new_id] = f"{chunk} ({i}/{len(chunks)})"
            cont.append(new_id)
        return SimpleNamespace(success=True, message_id=cont[-1],
                               continuation_message_ids=tuple(cont))

    async def fake_delete(chat_id, message_id):
        screen.pop(str(message_id), None)
        return True

    adapter.send = AsyncMock(side_effect=fake_send)
    adapter.edit_message = AsyncMock(side_effect=fake_edit)
    adapter.delete_message = AsyncMock(side_effect=fake_delete)
    return screen


async def _run(adapter):
    config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5,
                                  cursor=CURSOR)
    consumer = GatewayStreamConsumer(adapter, "chat_orphan", config)
    task = asyncio.create_task(consumer.run())
    # Grow past the limit in steps so a mid-stream edit overflows and splits.
    for i in range(12):
        consumer.on_delta(f"line {i} of reasoning text that takes up real width\n" * 14)
        await asyncio.sleep(0.03)
    consumer.finish()
    await task


@pytest.mark.asyncio
async def test_no_cursor_preview_survives_the_turn():
    adapter = _make_adapter()
    screen = _wire(adapter)

    await _run(adapter)

    leftovers = {mid: text for mid, text in screen.items() if CURSOR in text}
    assert not leftovers, (
        f"cursor-bearing preview(s) left on screen: {leftovers}")
