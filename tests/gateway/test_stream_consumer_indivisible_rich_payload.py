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


# --- Plain oversized prose: the seal leaves a leftover that STILL overflows ----------
#
# 2026-09-17, Discord thread 1548681041055715509. One turn, one API call, 36642
# chars, and the same answer reached the channel TWICE, interleaved: one sequence
# numbered ``(i/10)``, another ``(i/9)``, with chunk 1 byte-identical between them
# (sha256 of the bodies with the indicator stripped) and the concatenated prefixes
# equal. The normal final send was correctly suppressed, so the duplicate came from
# inside the consumer: ``_seal_overflow_heads`` clears the edit target mid-iteration,
# and the leftover - still far past the limit - landed on the non-final first-send
# lane, where the ADAPTER chunked and capped it. The turn-final lane then published
# the same text with its own denominator.
#
# Contract pinned here: on a non-final lane the consumer owns chunking. The adapter
# must never be handed a payload larger than one message.

# The consumer floors its own budget at 500 chars, so a toy adapter limit below
# that floor makes the consumer legitimately emit chunks larger than the limit -
# a measurement artifact, not a leak. Use the real Discord limit.
PLAIN_LIMIT = 2000
PLAIN_HEAD = "head line\n" * 60
PLAIN_TAIL = "".join(f"tail line {i:03d} filler filler filler filler filler\n" for i in range(240))


def _make_plain_adapter():
    """No rich tail at all: ``preferred_final_chunks`` declines, so an oversized
    payload has no atomic-chunk excuse and must be split by the consumer."""
    from gateway.platforms.base import BasePlatformAdapter

    cls = type(
        "PlainAdapter",
        (BasePlatformAdapter,),
        {
            "MAX_MESSAGE_LENGTH": PLAIN_LIMIT,
            "preferred_final_chunks": lambda self, text, budget: None,
            "preferred_final_split_index": lambda self, text, budget: None,
        },
    )
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    return adapter


async def _run_plain(adapter):
    config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor="")
    consumer = GatewayStreamConsumer(adapter, "chat_plain", config)
    consumer.on_delta(PLAIN_HEAD)
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.06)
    consumer.on_delta(PLAIN_TAIL)
    await asyncio.sleep(0.12)
    consumer.finish()
    # asyncio.wait_for, never a bare await: a hold branch that fails to yield
    # spins hot and would hang the suite instead of failing it.
    await asyncio.wait_for(task, timeout=10)
    return consumer


@pytest.mark.asyncio
async def test_non_final_lane_never_hands_the_adapter_an_oversized_payload():
    adapter = _make_plain_adapter()
    sends, _edits = _wire(adapter)

    await _run_plain(adapter)

    oversized = [
        len(c) for c, notify in sends
        if not notify and len(c) > adapter.MAX_MESSAGE_LENGTH
    ]
    assert not oversized, (
        "a non-final send handed the adapter a payload it has to chunk and cap "
        f"itself (lengths {oversized}); the turn-final lane then republishes the "
        "same text with a different denominator"
    )


@pytest.mark.asyncio
async def test_no_tail_line_is_published_twice_across_new_messages():
    adapter = _make_plain_adapter()
    sends, _edits = _wire(adapter)

    await _run_plain(adapter)

    joined = "".join(c for c, _ in sends)
    repeated = [
        f"tail line {i:03d}" for i in range(240)
        if joined.count(f"tail line {i:03d}") > 1
    ]
    assert not repeated, (
        f"{len(repeated)} tail line(s) reached the channel more than once as NEW "
        f"messages (first: {repeated[:3]}) - that is the interleaved duplicate"
    )
