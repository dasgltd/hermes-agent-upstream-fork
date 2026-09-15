"""A non-final lane must never seal a payload only the final delivery can render.

Discord re-renders the closing blocks of a reply as colored embeds, which carry
their own (far larger) budget, so ``preferred_final_chunks`` returns such a
payload as ONE atomic chunk even past the text limit. ``adapter.send()`` only
performs that rich split when the final-delivery flag is set; a non-final first
send therefore fell back to the blind size split and shipped the closing blocks
as 2-3 raw markdown messages. The turn-final delivery then re-rendered the SAME
payload as embeds on a new message and the raw copies stayed on screen
underneath - the duplicated tail reported on 2026-09-14 and again on 2026-09-15
(live Discord messages 1549417947041243147 / 1549417948828016661 /
1549417950128119809, byte-identical to ``truncate_message`` output, followed by
1549417952892162071 carrying the three embeds).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

MARK = "\n------\n"
HEAD = "reasoning line\n" * 12
# Purely the closing blocks, comfortably past the adapter's text limit.
BLOCKS = MARK + "closing block body line\n" * 40
BODY = "closing block body line"


def _make_adapter():
    """Adapter whose closing-block payload is one atomic chunk (Discord's embeds)."""
    from gateway.platforms.base import BasePlatformAdapter

    def preferred_final_chunks(self, text, budget):
        # Mirrors the real adapter: a leading separator (with or without the
        # newline a seal strips) means the payload IS the closing blocks.
        if text.lstrip("\n").startswith("------"):
            return [text]  # one atomic chunk: embeds carry their own budget
        if MARK not in text:
            return None
        idx = text.index(MARK)
        return [text[:idx], text[idx:]]

    def preferred_final_split_index(self, text, budget):
        if text.lstrip("\n").startswith("------") or MARK not in text:
            return None
        idx = text.index(MARK)
        return idx + 1 if 0 < idx + 1 <= budget else None

    cls = type(
        "AtomicBlockAdapter",
        (BasePlatformAdapter,),
        {
            "MAX_MESSAGE_LENGTH": 400,
            "preferred_final_chunks": preferred_final_chunks,
            "preferred_final_split_index": preferred_final_split_index,
        },
    )
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    return adapter


def _wire(adapter):
    sends, edits = [], []

    async def fake_send(**kw):
        sends.append((kw.get("content", ""), (kw.get("metadata") or {}).get("notify")))
        return SimpleNamespace(success=True, message_id=f"m{len(sends)}")

    async def fake_edit(**kw):
        edits.append(kw.get("content", ""))
        return SimpleNamespace(success=True, message_id=kw.get("message_id", "m1"))

    adapter.send = AsyncMock(side_effect=fake_send)
    adapter.edit_message = AsyncMock(side_effect=fake_edit)
    return sends, edits


async def _run(adapter):
    """Reproduce the live shape: a streamed head already on screen, then the
    closing blocks arrive and push the buffer past the limit. ``_seal_overflow_heads``
    seals the head and clears the edit target, so the SAME iteration lands the
    leftover blocks on the first-send lane with finalize=False."""
    config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor="")
    consumer = GatewayStreamConsumer(adapter, "chat_atomic", config)
    consumer.on_delta(HEAD)
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.06)
    consumer.on_delta(BLOCKS)
    await asyncio.sleep(0.12)
    consumer.finish()
    await task
    return consumer


@pytest.mark.asyncio
async def test_interim_lane_never_ships_the_atomic_block_payload_raw():
    adapter = _make_adapter()
    sends, _edits = _wire(adapter)

    await _run(adapter)

    leaked = [c for c, notify in sends if not notify and BODY in c]
    assert not leaked, (
        "an interim send sealed the closing blocks as raw text; the final "
        f"delivery re-renders the same payload as embeds: {leaked!r}"
    )


@pytest.mark.asyncio
async def test_the_block_payload_is_delivered_exactly_once():
    adapter = _make_adapter()
    sends, edits = _wire(adapter)

    await _run(adapter)

    # Every send opens a NEW bubble; edits rewrite one that is already on screen.
    # The closing blocks must therefore occupy exactly one bubble's worth of sends.
    shipped = sum(c.count(BODY) for c, _ in sends)
    assert shipped == BLOCKS.count(BODY), (
        f"the closing blocks were shipped as {shipped} lines across new messages, "
        f"expected {BLOCKS.count(BODY)} (a second copy is the duplicated tail)"
    )


@pytest.mark.asyncio
async def test_the_final_delivery_still_carries_the_whole_payload():
    adapter = _make_adapter()
    sends, edits = _wire(adapter)

    await _run(adapter)

    final = [c for c, notify in sends if notify] + edits
    assert final, "the payload was held but never delivered"
    assert any(c.count(BODY) == BLOCKS.count(BODY) for c in final), (
        "the final delivery lost part of the closing blocks"
    )
