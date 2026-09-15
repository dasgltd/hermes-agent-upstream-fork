"""Regression coverage: proxy tool-trail frames must die at the content boundary.

The local claude-proxy emits rendered tool progress inside a private frame
(``\\x1eHERMES_TOOL_TRAIL:[...]\\x1f``). The gateway stream consumer renders that
frame for display, but the SAME bytes also arrive as ordinary assistant
``content``. Nothing stripped them there, so the frame reached
``final_response`` and conversation history. That mismatch (clean stream ledger
vs contaminated final response) made the delivery reconciler treat a fully
delivered answer as stale and re-send it, duplicating the reply and exposing
the raw sentinel to the user.

ADR: claude-proxy docs/decisoes/0004-tool-trail-frames-must-be-stripped-at-the-content-boundary.md
"""

from types import SimpleNamespace
import pytest

from agent.message_sanitization import strip_tool_trail_frames

PREFIX = "\x1eHERMES_TOOL_TRAIL:"
SUFFIX = "\x1f"


def _frame(payload='["terminal npm test"]'):
    return PREFIX + payload + SUFFIX


def test_frame_embedded_between_prose_is_removed():
    text = "Answer part one." + _frame() + " Answer part two."
    assert strip_tool_trail_frames(text) == "Answer part one. Answer part two."


def test_frame_at_start_is_removed():
    assert strip_tool_trail_frames(_frame() + "Only prose.") == "Only prose."


def test_multiple_frames_are_all_removed():
    text = "a" + _frame('["one"]') + "b" + _frame('["two"]') + "c"
    assert strip_tool_trail_frames(text) == "abc"


def test_text_without_frames_is_untouched():
    text = "Plain answer mentioning HERMES_TOOL_TRAIL by name."
    assert strip_tool_trail_frames(text) == text


def test_unterminated_frame_is_removed_to_end():
    """A truncated stream can leave an opened frame with no closing sentinel.

    Leaving the tail in place would ship raw sentinel bytes to the user, so the
    opened frame is dropped through end-of-text.
    """
    assert strip_tool_trail_frames("prose" + PREFIX + '["x"') == "prose"


def test_non_string_input_passes_through():
    assert strip_tool_trail_frames(None) is None


class _FakeAgent:
    verbose_logging = False
    reasoning_callback = None
    stream_delta_callback = None
    _stream_callback = None

    def _extract_reasoning(self, msg):
        return None

    def _strip_think_blocks(self, content):
        return content

    def _needs_thinking_reasoning_pad(self):
        return False


class _Msg:
    content = "Done." + _frame('["\\ud83d\\udcbb terminal ls"]') + " Result above."
    tool_calls = None
    reasoning_content = None
    reasoning_details = None


def test_build_assistant_message_strips_frame_from_history():
    """The stored assistant message must never carry the frame into history."""
    from agent.chat_completion_helpers import build_assistant_message

    built = build_assistant_message(_FakeAgent(), _Msg(), "stop")
    content = built.get("content") or ""
    assert PREFIX not in content
    assert SUFFIX not in content
    assert "Done." in content and "Result above." in content


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_81641's fixture)."""
    from unittest.mock import MagicMock, patch
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    """Minimal chat-completion stand-in, inlined on purpose.

    This used to import ``_mock_response`` from an upstream test module; upstream
    deleted ``tests/run_agent/`` in the Sep 2026 reorganization and the carry broke.
    Same shape every upstream test module declares locally.
    """
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_whole_turn_final_response_and_history_are_frame_free(loop_agent):
    """ADR 0004 contract: final_response for a turn carrying a tool trail has no
    sentinel bytes, and neither does the stored assistant row. Drives the real
    loop (turn_final_response.finish_text_response + build_assistant_message)."""
    from unittest.mock import patch

    answer = "Done." + _frame('["terminal ls"]') + " Result above."
    loop_agent.client.chat.completions.create.side_effect = [
        _mock_response(content=answer, finish_reason="stop"),
    ]
    with (
        patch.object(loop_agent, "_flush_messages_to_session_db", return_value=True),
        patch.object(loop_agent, "_persist_session"),
        patch.object(loop_agent, "_save_trajectory"),
        patch.object(loop_agent, "_cleanup_task_resources"),
    ):
        result = loop_agent.run_conversation("what did you do?")

    final = result["final_response"]
    assert PREFIX not in final and SUFFIX not in final
    assert final == "Done. Result above."
    stored = [
        m for m in result["messages"]
        if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    assert stored, "assistant row must be in history"
    for m in stored:
        content = m.get("content") or ""
        assert "\x1e" not in content and "\x1f" not in content
