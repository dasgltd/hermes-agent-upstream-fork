"""Adapter-level tests for the colored closing blocks on Discord.

The parser itself is covered by tests/plugins/platforms/test_discord_rich_blocks.py.
These drive the real ``edit_message``/``send`` paths and assert the wire shape:
the streamed message keeps only the reasoning head, and the RESULT/handoff
blocks arrive as separate messages carrying real ``discord.Embed`` objects.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig

# tests/gateway/conftest.py installs the discord stub (or defers to the real
# library) at collection time - importing it here would run it a second time
# and change what later test modules see.
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402
from plugins.platforms.discord.rich_blocks import (  # noqa: E402
    HANDOFF_COLOR,
    RECAP_COLOR,
    RESULT_COLOR,
)

HANDOFF = "\U0001F449"
TLDR = "\U0001F3AF"

REPLY = "\n".join(
    [
        "**Thinking...**",
        "> checked the adapter",
        "",
        "⏳ Worked for 43s",
        "",
        "------",
        "",
        "✅ **the box works**",
        "",
        "- **TL;DR** — the colored box is an embed.",
        "",
        "**Feito:**",
        "- i. wired the adapter",
        "",
        "\u26a0\ufe0f **one caveat:** it needs its own box",
        "",
        "------",
        "",
        HANDOFF + " **SUA VEZ (nada me bloqueia)**:",
        "1. ship it? → **a)** yes · **b)** no",
    ]
)


def _field(embed, index):
    """Normalise a field across the real Embed and the suite's discord stub."""
    item = embed.fields[index]
    if isinstance(item, dict):
        return item["name"], item["value"]
    return item.name, item.value


def _color(embed):
    return getattr(embed.color, "value", embed.color)


def _make_adapter(**extra):
    # suppress_link_previews off keeps the edit on the plain ``msg.edit`` path;
    # the low-level flags route needs a real discord.http.
    extra.setdefault("suppress_link_previews", False)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***", extra=extra))
    adapter._record_discord_response = MagicMock()
    return adapter


def _wire(adapter):
    """Wire a client whose channel records sends and whose message records edits."""
    sends = []
    edits = []

    async def fake_send(**kwargs):
        sends.append(kwargs)
        return SimpleNamespace(id=9000 + len(sends))

    async def fake_edit(channel_id, message_id, *, params):
        edits.append(params)
        return {}

    async def fake_msg_edit(**kwargs):
        edits.append(kwargs)
        return None

    msg = SimpleNamespace(
        id=42,
        channel=SimpleNamespace(id=555),
        edit=AsyncMock(side_effect=fake_msg_edit),
        _state=SimpleNamespace(http=SimpleNamespace(edit_message=fake_edit)),
    )
    channel = SimpleNamespace(
        id=555,
        get_partial_message=MagicMock(return_value=msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    return channel, msg, sends, edits


def _edited_content(edits):
    return (edits[-1] or {}).get("content", "")


@pytest.mark.asyncio
async def test_final_edit_keeps_the_head_and_ships_three_embeds():
    adapter = _make_adapter()
    channel, _msg, sends, edits = _wire(adapter)

    result = await adapter.edit_message("555", "42", REPLY, finalize=True)

    assert result.success
    head = _edited_content(edits)
    assert "Worked for 43s" in head
    assert "the box works" not in head and HANDOFF not in head

    embeds = [call["embed"] for call in sends]
    assert len(embeds) == 3
    assert [_color(e) for e in embeds] == [RESULT_COLOR, RECAP_COLOR, HANDOFF_COLOR]
    assert embeds[0].title == "✅ THE BOX WORKS"
    assert _field(embeds[0], 0)[0] == "TL;DR"
    assert "colored box is an embed" in _field(embeds[0], 0)[1]
    # The recap reads inside the green box; the caveat is the one that leaves.
    assert any(_field(embeds[0], i)[0] == "Feito" for i in range(len(embeds[0].fields)))
    assert embeds[1].title.startswith("\u26a0")
    assert embeds[1].title == "\u26a0\ufe0f ONE CAVEAT"
    assert any("wired the adapter" in _field(embeds[0], i)[1]
               for i in range(len(embeds[0].fields)))
    assert "needs its own box" in _field(embeds[1], 0)[1]
    # The handoff status moved out of the title into its own field. The label is
    # STATUS, not TL;DR: the red box states whether I am blocked, it is not a summary.
    assert embeds[2].title == HANDOFF + " SUA VEZ"
    assert _field(embeds[2], 0) == ("STATUS", "Nada me bloqueia")
    assert "1. ship it?" in _field(embeds[2], 1)[1]
    # Rich embeds must not be sent with SUPPRESS_EMBEDS - that flag hides them.
    assert all("suppress_embeds" not in call for call in sends)


@pytest.mark.asyncio
async def test_recovery_record_keeps_the_whole_reply_not_just_the_head():
    adapter = _make_adapter()
    _channel, _msg, _sends, _edits = _wire(adapter)

    await adapter.edit_message("555", "42", REPLY, finalize=True)

    recorded = adapter._record_discord_response.call_args.kwargs["content"]
    assert HANDOFF in recorded and "the box works" in recorded


@pytest.mark.asyncio
async def test_mid_stream_edit_is_never_split():
    adapter = _make_adapter()
    _channel, _msg, sends, edits = _wire(adapter)

    await adapter.edit_message("555", "42", REPLY, finalize=False)

    assert sends == []
    assert HANDOFF in _edited_content(edits)


@pytest.mark.asyncio
async def test_feature_can_be_turned_off_per_profile():
    adapter = _make_adapter(rich_result_embeds=False)
    _channel, _msg, sends, edits = _wire(adapter)

    await adapter.edit_message("555", "42", REPLY, finalize=True)

    assert sends == []
    assert HANDOFF in _edited_content(edits)


@pytest.mark.asyncio
async def test_plain_reply_is_delivered_unchanged():
    adapter = _make_adapter()
    _channel, _msg, sends, edits = _wire(adapter)

    await adapter.edit_message("555", "42", "just a short answer", finalize=True)

    assert sends == []
    assert _edited_content(edits) == "just a short answer"


@pytest.mark.asyncio
async def test_embed_failure_never_fails_the_turn():
    adapter = _make_adapter()
    channel, _msg, _sends, edits = _wire(adapter)
    channel.send = AsyncMock(side_effect=RuntimeError("discord is down"))

    result = await adapter.edit_message("555", "42", REPLY, finalize=True)

    assert result.success  # the reply itself landed; a missing box is cosmetic
    assert "Worked for 43s" in _edited_content(edits)


def test_preferred_split_index_points_at_the_first_rule():
    """An oversized reply is cut where the boxes start, not mid-block."""
    adapter = _make_adapter()
    index = adapter.preferred_final_split_index(REPLY, 2000)
    assert index is not None
    assert REPLY[index:].startswith("------")
    assert REPLY[:index].rstrip().endswith("Worked for 43s")


def test_preferred_split_index_declines_when_the_boxes_start_past_the_budget():
    adapter = _make_adapter()
    assert adapter.preferred_final_split_index(REPLY, 10) is None


def test_preferred_split_index_is_none_when_the_feature_is_off():
    adapter = _make_adapter(rich_result_embeds=False)
    assert adapter.preferred_final_split_index(REPLY, 2000) is None


def test_preferred_split_index_is_none_for_a_plain_reply():
    adapter = _make_adapter()
    assert adapter.preferred_final_split_index("no closing block here", 2000) is None


@pytest.mark.asyncio
async def test_boundary_tail_is_delivered_as_one_embed_only_message():
    """The split tail is pure closing blocks: embeds, no empty text bubble."""
    adapter = _make_adapter()
    channel, _msg, sends, _edits = _wire(adapter)
    index = adapter.preferred_final_split_index(REPLY, 2000)

    result = await adapter.send(
        "555", REPLY[index:], metadata={"notify": True},
    )

    assert result.success and result.message_id
    assert len(sends) == 1
    assert not (sends[0].get("content") or "")
    embeds = sends[0]["embeds"]
    assert [_color(e) for e in embeds] == [RESULT_COLOR, RECAP_COLOR, HANDOFF_COLOR]
    assert _field(embeds[0], 0)[0] == "TL;DR"
    # The ledger still replays the whole payload, not the empty text.
    assert adapter._record_discord_response.call_args.kwargs["content"] == REPLY[index:]


# The consumer cuts an oversized reply on the block boundary (see
# tests/gateway/test_stream_consumer_block_boundary.py), so the last streamed
# message can hold nothing but the closing blocks. Editing it must turn the
# text INTO the boxes - the shape that shipped raw text on every split turn.
TAIL_ONLY = "\n".join(
    [
        "------",
        "",
        "✅ **the box works**",
        "",
        "- **TL;DR** — the colored box is an embed.",
        "",
        "**Feito:**",
        "- i. wired the adapter",
        "",
        "\u26a0\ufe0f **one caveat:** it needs its own box",
    ]
)


@pytest.mark.asyncio
async def test_block_only_tail_becomes_embeds_on_the_same_message():
    adapter = _make_adapter()
    _channel, _msg, sends, edits = _wire(adapter)

    result = await adapter.edit_message("555", "42", TAIL_ONLY, finalize=True)

    assert result.success
    # No trailing extra message: the tail message itself carries the boxes.
    assert sends == []
    payload = edits[-1]
    assert payload.get("content") == ""
    embeds = payload["embeds"]
    assert [_color(e) for e in embeds] == [RESULT_COLOR, RECAP_COLOR]
    assert embeds[0].title == "✅ THE BOX WORKS"
    assert _field(embeds[0], 0)[0] == "TL;DR"
    assert embeds[1].title == "\u26a0\ufe0f ONE CAVEAT"
    # The raw markers must not survive anywhere in the delivered text.
    assert "------" not in (payload.get("content") or "")


@pytest.mark.asyncio
async def test_block_only_tail_records_delivery_as_final():
    adapter = _make_adapter()
    _channel, _msg, _sends, _edits = _wire(adapter)

    await adapter.edit_message(
        "555", "42", TAIL_ONLY, finalize=True,
        metadata={"reply_to_message_id": "77"},
    )

    kwargs = adapter._record_discord_response.call_args.kwargs
    assert kwargs["final"] is True
    assert kwargs["reply_to"] == "77"
    assert "the box works" in kwargs["content"]


@pytest.mark.asyncio
async def test_block_only_tail_falls_back_to_text_when_the_embed_edit_fails():
    adapter = _make_adapter()
    _channel, msg, _sends, edits = _wire(adapter)

    calls = {"n": 0}
    original = msg.edit

    async def flaky(**kwargs):
        calls["n"] += 1
        if "embeds" in kwargs:
            raise RuntimeError("embed edit rejected")
        return await original(**kwargs)

    msg.edit = flaky

    result = await adapter.edit_message("555", "42", TAIL_ONLY, finalize=True)

    # Styling must never cost the user the answer.
    assert result.success
    assert "the box works" in _edited_content(edits)


@pytest.mark.asyncio
async def test_block_only_tail_stays_plain_text_when_the_feature_is_off():
    adapter = _make_adapter(rich_result_embeds=False)
    _channel, _msg, sends, edits = _wire(adapter)

    await adapter.edit_message("555", "42", TAIL_ONLY, finalize=True)

    assert sends == []
    assert "the box works" in _edited_content(edits)


# A reasoning head longer than one Discord message pushes the boxes past the
# budget. ``preferred_final_split_index`` declines there, and the blind split
# used to cut a box in half (observed live on 2026-09-14): the severed half
# landed as raw text right next to the re-rendered whole box.
LONG_HEAD_REPLY = ("**Thinking...**\n> " + "reasoning that runs long. " * 200
                   + "\n\n⏳ Worked for 4m 20s\n\n" + REPLY.split("------", 1)[1].join(
                       ["------", ""]))


@pytest.mark.asyncio
async def test_chunking_never_cuts_a_box_when_the_head_overflows():
    adapter = _make_adapter()
    chunks = adapter.preferred_final_chunks(LONG_HEAD_REPLY, 2000)

    assert chunks and len(chunks) > 2  # the head alone needs several messages
    assert LONG_HEAD_REPLY.endswith(chunks[-1])
    # The closing blocks travel whole, in one chunk, past the text limit.
    assert chunks[-1].lstrip().startswith("------")
    assert len(chunks[-1]) > 0
    for chunk in chunks[:-1]:
        assert "the box works" not in chunk and HANDOFF not in chunk
    assert "reasoning that runs long" not in chunks[-1]


def test_chunking_declines_a_plain_reply():
    adapter = _make_adapter()
    assert adapter.preferred_final_chunks("no closing block here", 2000) is None


def test_chunking_is_none_when_the_feature_is_off():
    adapter = _make_adapter(rich_result_embeds=False)
    assert adapter.preferred_final_chunks(LONG_HEAD_REPLY, 2000) is None


def test_a_block_only_payload_is_one_atomic_chunk():
    """The head was sealed earlier: what is left must not be split again."""
    adapter = _make_adapter()
    assert adapter.preferred_final_chunks(TAIL_ONLY, 10) == [TAIL_ONLY]


@pytest.mark.asyncio
async def test_oversized_block_only_payload_still_ships_as_embeds():
    adapter = _make_adapter()
    _channel, _msg, sends, _edits = _wire(adapter)
    padded = TAIL_ONLY.replace(
        "- i. wired the adapter",
        "\n".join("- i%d. wired the adapter with padding text" % i for i in range(60)),
    )
    assert len(padded) > 2000

    result = await adapter.send("555", padded, metadata={"notify": True})

    assert result.success
    assert len(sends) == 1  # one message, not a length split
    assert not (sends[0].get("content") or "")
    assert [_color(e) for e in sends[0]["embeds"]] == [RESULT_COLOR, RECAP_COLOR]
