"""The consumer cuts an oversized reply where the adapter asks it to.

Discord renders the closing blocks of a reply as colored embeds. Splitting a
reply at the last line break that fits would cut a block in half: the first
part becomes an embed, the rest lands as plain text in the next message
(observed live on 2026-09-13). The adapter exposes the block boundary through
``preferred_final_split_index`` and the consumer must honour it.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

BOUNDARY_MARK = "\n------\n"


def _make_adapter(preferred=None):
    from gateway.platforms.base import BasePlatformAdapter

    body = {"MAX_MESSAGE_LENGTH": 700}
    if preferred is not None:
        body["preferred_final_split_index"] = preferred
    cls = type("BoundaryAdapter", (BasePlatformAdapter,), body)
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    return adapter


def _wire(adapter):
    sent, edited = [], []

    async def fake_send(**kw):
        sent.append(kw["content"])
        return SimpleNamespace(success=True, message_id=f"m{len(sent)}")

    async def fake_edit(**kw):
        edited.append(kw["content"])
        return SimpleNamespace(success=True, message_id="m1")

    adapter.send = AsyncMock(side_effect=fake_send)
    adapter.edit_message = AsyncMock(side_effect=fake_edit)
    return sent, edited


def _reply():
    # Head alone fits; head + closing block does not, so the consumer splits.
    head = "reasoning line\n" * 30
    return head + BOUNDARY_MARK + "closing block body\n" * 10


async def _run(adapter):
    config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor="")
    consumer = GatewayStreamConsumer(adapter, "chat_boundary", config)
    consumer.on_delta(_reply())
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.08)
    consumer.finish()
    await task


@pytest.mark.asyncio
async def test_split_lands_on_the_adapter_boundary():
    adapter = _make_adapter(
        preferred=lambda self, text, budget: (
            text.index(BOUNDARY_MARK) + 1
            if BOUNDARY_MARK in text and text.index(BOUNDARY_MARK) + 1 <= budget
            else None
        )
    )
    sent, edited = _wire(adapter)

    await _run(adapter)

    delivered = edited + sent
    assert delivered, "nothing was delivered"
    # No message carries a piece of the head AND a piece of the closing block.
    for text in delivered:
        assert not ("reasoning line" in text and "closing block body" in text)
    assert any(text.lstrip().startswith("------") for text in delivered)


@pytest.mark.asyncio
async def test_no_hook_keeps_the_plain_line_break_split():
    adapter = _make_adapter()
    sent, edited = _wire(adapter)

    await _run(adapter)

    assert edited or sent


@pytest.mark.asyncio
async def test_a_raising_hook_falls_back_instead_of_breaking_delivery():
    def boom(self, text, budget):
        raise RuntimeError("hook exploded")

    adapter = _make_adapter(preferred=boom)
    sent, edited = _wire(adapter)

    await _run(adapter)

    assert edited or sent


@pytest.mark.asyncio
async def test_out_of_range_index_is_ignored():
    adapter = _make_adapter(preferred=lambda self, text, budget: 10 ** 6)
    sent, edited = _wire(adapter)

    await _run(adapter)

    assert edited or sent


# ``preferred_final_split_index`` can only move ONE cut and only inside the
# budget, so a reasoning head longer than a single message left the consumer
# blind-splitting the closing block. ``preferred_final_chunks`` hands over the
# whole chunk list instead, and the structured tail stays in one message.
def _long_reply():
    head = "reasoning line that runs long\n" * 60  # several messages on its own
    return head + BOUNDARY_MARK + "closing block body\n" * 60


def _chunking_adapter():
    def preferred_final_chunks(self, text, budget):
        # Mirrors the Discord adapter: a payload that IS the closing block is
        # atomic (it becomes embeds, which carry their own budget).
        if text.lstrip().startswith(BOUNDARY_MARK.strip()):
            return [text]
        if BOUNDARY_MARK not in text:
            return None
        cut = text.index(BOUNDARY_MARK) + 1
        head, tail = text[:cut], text[cut:]
        chunks = [head[i:i + budget] for i in range(0, len(head), budget)]
        return [c for c in chunks if c.strip()] + [tail]

    cls = type("ChunkingAdapter", (_base(),), {
        "MAX_MESSAGE_LENGTH": 700,
        "preferred_final_chunks": preferred_final_chunks,
    })
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    return adapter


def _base():
    from gateway.platforms.base import BasePlatformAdapter

    return BasePlatformAdapter


async def _run_text(adapter, text):
    config = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5, cursor="")
    consumer = GatewayStreamConsumer(adapter, "chat_boundary", config)
    consumer.on_delta(text)
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.08)
    consumer.finish()
    await task


@pytest.mark.asyncio
async def test_long_head_never_cuts_the_closing_block():
    adapter = _chunking_adapter()
    sent, edited = _wire(adapter)
    text = _long_reply()

    await _run_text(adapter, text)

    delivered = edited + sent
    assert delivered, "nothing was delivered"
    for message in delivered:
        assert not ("reasoning line" in message and "closing block body" in message)
    # The block arrived whole, in a single message, past the 700-char limit.
    block_messages = [m for m in delivered if "closing block body" in m]
    assert block_messages
    assert max(len(m) for m in block_messages) > adapter.MAX_MESSAGE_LENGTH


@pytest.mark.asyncio
async def test_a_raising_chunk_hook_falls_back_instead_of_breaking_delivery():
    def boom(self, text, budget):
        raise RuntimeError("chunk hook exploded")

    cls = type("BoomAdapter", (_base(),), {
        "MAX_MESSAGE_LENGTH": 700, "preferred_final_chunks": boom})
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    sent, edited = _wire(adapter)

    await _run_text(adapter, _long_reply())

    assert edited or sent


@pytest.mark.asyncio
async def test_a_mangled_chunk_list_is_rejected():
    """The last chunk must be a suffix of the text or the list is dropped."""
    cls = type("MangleAdapter", (_base(),), {
        "MAX_MESSAGE_LENGTH": 700,
        "preferred_final_chunks": lambda self, text, budget: ["totally", "different"]})
    cls.__abstractmethods__ = frozenset()
    adapter = cls.__new__(cls)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    sent, edited = _wire(adapter)

    await _run_text(adapter, _long_reply())

    delivered = edited + sent
    assert delivered
    assert all("totally" not in m for m in delivered)


@pytest.mark.asyncio
async def test_an_all_block_reply_is_held_whole_instead_of_spinning():
    """A buffer the adapter keeps whole makes no progress in the split path.

    Before the guard, ``run()`` re-entered ``_split_first_send`` without ever
    yielding: a hot loop that never drained the done tick, so the turn hung.
    """
    adapter = _chunking_adapter()
    sent, edited = _wire(adapter)
    text = BOUNDARY_MARK.lstrip("\n") + "closing block body\n" * 60

    await asyncio.wait_for(_run_text(adapter, text), timeout=5)

    delivered = edited + sent
    assert delivered, "nothing was delivered"
    assert any("closing block body" in m for m in delivered)
    # Delivered in one piece, not sliced to the 700-char text limit.
    assert max(len(m) for m in delivered) > adapter.MAX_MESSAGE_LENGTH
