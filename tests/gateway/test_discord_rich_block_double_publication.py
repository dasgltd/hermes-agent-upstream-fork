"""One turn must not put the same closing blocks on screen twice.

Two independent lanes publish closing blocks on Discord: the consumer's split
lane sends them as an embed-only message (``_send_embed_only_message``) and the
turn-final edit republishes them as the rich tail of ``edit_message`` via
``_send_rich_blocks``. Neither can see the other.

Measured live on 2026-09-17 in thread 1532149266271703241: messages
1550226868643438683 (19:28:07.307) and 1550226871101169756 (19:28:07.893)
carried byte-identical embeds 586ms apart, with the head edit landing between
them at 19:28:07.539. 586ms also rules out a discord.py transport retry, whose
first retry sleeps at least 1 second.
"""

import asyncio
from types import SimpleNamespace

import pytest

from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.rich_blocks import RichBlock


class _FakeMsg:
    def __init__(self, mid):
        self.id = mid


class _FakeChannel:
    def __init__(self, cid="chan1"):
        self.id = cid
        self.embed_sends = []
        self._next = 0

    async def send(self, content=None, embed=None, embeds=None, reference=None, **kw):
        if embed is not None or embeds:
            self.embed_sends.append(embed if embed is not None else embeds)
        self._next += 1
        return _FakeMsg(f"m{self._next}")


def _adapter(channel):
    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter.platform = SimpleNamespace(value="discord")
    adapter._rich_result_embeds = True

    async def _resolve(_chat_id):
        return channel

    adapter._resolve_channel = _resolve
    adapter._build_rich_embed = lambda block: SimpleNamespace(title=block.title)
    return adapter


def _blocks():
    return [RichBlock(title="RESULT", description="body", color=5763719)]


def test_second_lane_is_suppressed_when_embed_only_published_first():
    """Live shape: embed-only send lands first, the final edit's tail must not repeat it."""
    channel = _FakeChannel()
    adapter = _adapter(channel)

    async def run():
        first = await adapter._send_embed_only_message(channel, _blocks())
        await adapter._send_rich_blocks("chan1", _blocks(), {})
        return first

    first = asyncio.run(run())
    assert first.success and first.message_id == "m1"
    assert len(channel.embed_sends) == 1, "the closing blocks reached the channel twice"


def test_second_lane_is_suppressed_in_the_reverse_order():
    """Rich tail first: the embed-only lane must hand back the id the other lane got."""
    channel = _FakeChannel()
    adapter = _adapter(channel)

    async def run():
        await adapter._send_rich_blocks("chan1", _blocks(), {})
        return await adapter._send_embed_only_message(channel, _blocks())

    result = asyncio.run(run())
    assert len(channel.embed_sends) == 1, "the closing blocks reached the channel twice"
    # A real edit target, never None: None would push the consumer into fallback mode.
    assert result.success and result.message_id == "m1"
    assert result.raw_response.get("deduplicated") is True


def test_a_different_payload_is_not_suppressed():
    """The guard keys on the payload, so a genuinely different turn still publishes."""
    channel = _FakeChannel()
    adapter = _adapter(channel)
    other = [RichBlock(title="OTHER RESULT", description="body", color=5763719)]

    async def run():
        await adapter._send_rich_blocks("chan1", _blocks(), {})
        await adapter._send_rich_blocks("chan1", other, {})

    asyncio.run(run())
    assert len(channel.embed_sends) == 2


def test_the_same_payload_in_another_chat_is_not_suppressed():
    """The claim is per destination: the same boxes may legitimately go to two chats."""
    chan_a, chan_b = _FakeChannel("a"), _FakeChannel("b")
    adapter = _adapter(chan_a)

    async def run():
        await adapter._send_rich_blocks("a", _blocks(), {})

        async def _resolve_b(_chat_id):
            return chan_b

        adapter._resolve_channel = _resolve_b
        await adapter._send_rich_blocks("b", _blocks(), {})

    asyncio.run(run())
    assert len(chan_a.embed_sends) == 1
    assert len(chan_b.embed_sends) == 1


def test_the_claim_expires_so_a_later_turn_can_repeat_itself():
    """A guard with no horizon would silently swallow an identical reply hours later."""
    channel = _FakeChannel()
    adapter = _adapter(channel)

    async def run():
        await adapter._send_rich_blocks("chan1", _blocks(), {})
        for key, entry in list(adapter._rich_block_claims.items()):
            adapter._rich_block_claims[key] = (
                entry[0] - adapter._RICH_BLOCK_CLAIM_TTL - 1.0, entry[1], entry[2])
        await adapter._send_rich_blocks("chan1", _blocks(), {})

    asyncio.run(run())
    assert len(channel.embed_sends) == 2


def test_the_claim_table_is_bounded():
    """Unbounded per-chat state is a slow leak on a 24/7 gateway."""
    channel = _FakeChannel()
    adapter = _adapter(channel)

    async def run():
        for i in range(adapter._RICH_BLOCK_CLAIM_MAX * 2):
            await adapter._send_rich_blocks(
                "chan1", [RichBlock(title=f"T{i}", description="b", color=1)], {})

    asyncio.run(run())
    assert len(adapter._rich_block_claims) <= adapter._RICH_BLOCK_CLAIM_MAX
