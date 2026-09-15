"""Rich closing-block extraction for Discord replies.

A finished assistant reply follows a fixed shape: reasoning checkpoints and a
duration line, then a horizontal rule, then a RESULT/DECISION block, then
another rule, then the handoff block. Discord renders plain text for all of
it, which makes the two blocks that actually carry decisions blend into the
wall of reasoning above them.

This module splits that reply into the plain-text head plus structured blocks
the adapter turns into colored embeds (green for the result, red for the
handoff). Everything here is pure - no discord.py import - so the parser stays
unit-testable and the adapter owns the only platform-specific step.

Non-ASCII markers are written as escapes so the source file stays pure ASCII.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Markers used by the reply format. Escapes keep this file ASCII-only.
HANDOFF_MARKER = "\U0001F449"  # pointing hand, opens the handoff block
TLDR_MARKER = "\U0001F3AF"  # dart, legacy optional prefix on the TL;DR bullet
WARNING_MARKER = "\u26A0"  # warning sign, starts a standalone caveat

# Discord embed accent colors (matches the sample approved by the user).
RESULT_COLOR = 0x57F287
RECAP_COLOR = 0xFEE75C
HANDOFF_COLOR = 0xED4245

# Warning sign with the emoji variation selector, used as the caveat box title.
WARNING_SIGN = "\u26A0\uFE0F"
# Upper bound on caveat boxes; extra caveats are folded into the last one so a
# reply can never exceed Discord's 10-embeds-per-message ceiling.
MAX_WARNING_BLOCKS = 6

# Discord embed hard limits.
MAX_TITLE = 256
MAX_FIELD_NAME = 256
MAX_FIELD_VALUE = 1024
MAX_DESCRIPTION = 4096
MAX_FIELDS = 25

# Zero-width space: a field name Discord accepts but renders as blank, so an
# unlabeled chunk can still live in the ordered field list.
INVISIBLE_NAME = "\u200b"

# Field name for the handoff aside. The green box calls its summary TL;DR;
# the red box carries a blocked/not-blocked status, which is a different
# thing and must not borrow that label.
HANDOFF_ASIDE_NAME = "STATUS"

_SEPARATOR_RE = re.compile(r"^\s*-{3,}\s*$")
# First line of an extractable block: an optional leading emoji, then a bold
# run, optionally followed by a colon.
_BLOCK_TITLE_RE = re.compile(r"^\s*(?:\S{1,4}\s+)?\*\*.+\*\*\s*:?\s*$")
_SUBHEADER_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*:?\s*$")
_TLDR_RE = re.compile(
    r"^\s*[-*]\s*(?:" + TLDR_MARKER + r"\s*)?\*\*TL;DR\*\*\s*(?:[-\u2013\u2014:]\s*)?(.*)$"
)
# A line that opens a new block, so a TL;DR wrap-around stops before it.
_BLOCK_START_RE = re.compile(r"^\s*(?:[-*]\s|\d+[.)]\s|\*\*|>|#)")
_MEDIA_RE = re.compile(r"^\s*MEDIA:\S+\s*$")
# A list item, captured so a caveat written as a bullet is still seen as one.
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")
# Backticks inside a title: embed titles render no markdown, so they are dropped.
_CODE_TICK_RE = re.compile(r"`+")
# An inline-code span, captured so a title can keep that token's own case.
_CODE_SPAN_RE = re.compile(r"(`+[^`]+`+)")
# A first word that is a technical token (version, path, config key, handle):
# shouting its initial would break copy/paste, so it is never capitalized.
_TECHNICAL_TOKEN_RE = re.compile(r"^\S*[0-9./_@:\\-]\S*$")
# Trailing parenthetical on a block title, lifted into a TL;DR-style field.
_TITLE_PARENTHETICAL_RE = re.compile(r"^(.*?)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)$")


@dataclass
class EmbedField:
    """One name/value pair inside an embed."""

    name: str
    value: str


@dataclass
class RichBlock:
    """Platform-neutral description of a single colored embed."""

    title: str
    color: int
    description: Optional[str] = None
    fields: List[EmbedField] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.description or "").strip() and not self.fields


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _strip_bold(text: str) -> str:
    return text.replace("**", "").strip()


def _capitalize_first(text: str) -> str:
    """Upper-case the first letter of a prose value.

    A box body is lifted out of its bullet ("- **TL;DR** - the box is..."), so
    the sentence that reaches the embed often starts lower case even though the
    reply reads correctly as markdown. Capitalizing it here is what makes every
    box open like a sentence.

    Only a plain lower-case letter is touched, and never when the first word is
    a technical token (``n8n``, ``config.yaml``, ``claude-proxy``) or carries
    markup: those are case sensitive and shouting them breaks copy/paste.
    """
    if not text:
        return text
    stripped = text.lstrip()
    if not stripped:
        return text
    first = stripped[0]
    if not (first.isalpha() and first.islower()):
        return text
    word = stripped.split(None, 1)[0]
    if "`" in word or "*" in word or _TECHNICAL_TOKEN_RE.match(word):
        return text
    pad = len(text) - len(stripped)
    return text[:pad] + first.upper() + stripped[1:]


def _split_leading_bold(text: str) -> Tuple[Optional[str], str]:
    """Split ``**lead** rest`` into its bold lead and the rest of the line."""
    if not text.startswith("**"):
        return None, text
    close = text.find("**", 2)
    if close <= 2:
        return None, text
    lead = text[2:close].strip()
    if not lead:
        return None, text
    return lead, text[close + 2 :].lstrip()


def _block_title(line: str) -> str:
    """Title line in upper case, without bold, code ticks or trailing colon.

    Discord does not render markdown inside an embed title, so a backtick
    would show up literally; strip the ticks and keep the token itself. Upper
    case is the requested look for the box header - str.upper keeps accented
    letters intact, so a Portuguese title stays correct.

    A token that was written as inline code keeps its ORIGINAL case: a file
    name, a command or a config key is case sensitive, and shouting it breaks
    copy/paste (``config.yaml`` must not become ``CONFIG.YAML``).
    """
    text = _strip_bold(line).rstrip(":").rstrip()
    parts = []
    for piece in _CODE_SPAN_RE.split(text):
        if piece and piece[0] == "`":
            parts.append(_CODE_TICK_RE.sub("", piece))
        else:
            parts.append(piece.upper())
    return _clip("".join(parts), MAX_TITLE)


def _split_title_parenthetical(line: str) -> Tuple[str, Optional[str]]:
    """Split ``TITLE (aside)`` into the title line and the aside.

    The handoff header carries its status in parentheses ("nada me bloqueia",
    "estou parado nestes 2 pontos"). Inside a box that reads as noise glued to
    the header, so the aside is lifted out and rendered the same way the TL;DR
    is rendered in the result box: a named field above the body.
    """
    text = _strip_bold(line).rstrip(":").rstrip()
    match = _TITLE_PARENTHETICAL_RE.match(text)
    if not match:
        return line, None
    head, aside = match.group(1).strip(), match.group(2).strip()
    if not head or not aside:
        return line, None
    return head, aside


def _drop_media_lines(lines: List[str]) -> List[str]:
    """Remove bare MEDIA: directives - they are delivered as attachments."""
    return [line for line in lines if not _MEDIA_RE.match(line)]


def _split_segments(lines: List[str]) -> List[Tuple[int, int, int]]:
    """Split lines on horizontal rules.

    Returns (separator_index, body_start, body_end) triples in order. The
    separator index is -1 for the leading segment, which has no rule above it.
    """
    segments: List[Tuple[int, int, int]] = []
    separator = -1
    start = 0
    for index, line in enumerate(lines):
        if _SEPARATOR_RE.match(line):
            segments.append((separator, start, index))
            separator = index
            start = index + 1
    segments.append((separator, start, len(lines)))
    return segments


def _first_content_index(lines: List[str], start: int, end: int) -> Optional[int]:
    for index in range(start, end):
        if lines[index].strip():
            return index
    return None


def _chunk_value(value: str, limit: int = MAX_FIELD_VALUE) -> List[str]:
    """Split a field value on line boundaries so no chunk exceeds the limit."""
    chunks: List[str] = []
    current = ""
    for line in value.split("\n"):
        candidate = line if not current else current + "\n" + line
        if len(candidate) > limit:
            if current:
                chunks.append(current)
                current = line[:limit]
            else:
                chunks.append(line[:limit])
                current = ""
        else:
            current = candidate
    if current.strip():
        chunks.append(current)
    return chunks


def _add_field(block: RichBlock, name: str, value: str) -> None:
    value = (value or "").strip("\n").rstrip()
    if not value.strip():
        return
    for position, chunk in enumerate(_chunk_value(value)):
        if len(block.fields) >= MAX_FIELDS:
            return
        block.fields.append(
            EmbedField(
                name=_clip(name if position == 0 else INVISIBLE_NAME, MAX_FIELD_NAME),
                value=_capitalize_first(chunk) if position == 0 else chunk,
            )
        )


def _extract_tldr(lines: List[str]) -> Tuple[Optional[str], List[str]]:
    """Pull the TL;DR bullet (and its wrapped continuation) out of a chunk."""
    for index, line in enumerate(lines):
        match = _TLDR_RE.match(line)
        if not match:
            continue
        value = [match.group(1).strip()]
        cursor = index + 1
        while cursor < len(lines):
            nxt = lines[cursor]
            if not nxt.strip() or _BLOCK_START_RE.match(nxt):
                break
            value.append(nxt.strip())
            cursor += 1
        remainder = lines[:index] + lines[cursor:]
        return "\n".join(value).strip(), remainder
    return None, lines


def _warning_head(line: str) -> Optional[str]:
    """Return the caveat text when ``line`` opens a standalone warning.

    A caveat is written either as its own paragraph ("WARN **lead** ...") or as
    a bullet in the leading chunk ("- WARN **lead** ..."). The bullet form used
    to hide the marker behind the dash, so the caveat stayed filed in the green
    box instead of getting the amber box of its own it is supposed to get.
    """
    stripped = line.lstrip()
    if stripped.startswith(WARNING_MARKER):
        return stripped
    match = _BULLET_RE.match(line)
    if match and match.group(1).lstrip().startswith(WARNING_MARKER):
        return match.group(1).lstrip()
    return None


def _is_bullet_warning(line: str) -> bool:
    """True when the caveat is a list item rather than its own paragraph."""
    match = _BULLET_RE.match(line)
    return bool(match and match.group(1).lstrip().startswith(WARNING_MARKER))


def _opens_bullet_warning(chunk: List[str]) -> bool:
    """True when this buffer already started as a run of caveat bullets."""
    for line in chunk:
        if line.strip():
            return _is_bullet_warning(line)
    return False


def _sections(lines: List[str]) -> List[Tuple[Optional[str], List[str]]]:
    """Group body lines into (label, lines) sections.

    A bold-only line opens a labeled section; a standalone warning line opens
    an unlabeled one so caveats do not get filed under the previous heading.
    """
    sections: List[Tuple[Optional[str], List[str]]] = []
    label: Optional[str] = None
    buffer: List[str] = []
    for line in lines:
        header = _SUBHEADER_RE.match(line)
        if header:
            sections.append((label, buffer))
            label = _strip_bold(header.group(1)).rstrip(":")
            buffer = []
            continue
        if _warning_head(line) is not None and any(x.strip() for x in buffer):
            # A bullet caveat only breaks out of an unlabeled chunk: inside a
            # labeled one ("Feito:") it is an item of that list, and ripping it
            # out would leave the list with a hole.
            if _is_bullet_warning(line):
                if label is not None:
                    buffer.append(line)
                    continue
                if _opens_bullet_warning(buffer):
                    buffer.append(line)  # consecutive caveats share one box
                    continue
            sections.append((label, buffer))
            label = None
            buffer = [line]
            continue
        buffer.append(line)
    sections.append((label, buffer))
    return sections


def _is_warning_chunk(chunk: List[str]) -> bool:
    """True when this unlabeled section is a standalone caveat paragraph."""
    for line in chunk:
        if line.strip():
            return _warning_head(line) is not None
    return False


def _build_warning_block(chunk: List[str]) -> Optional[RichBlock]:
    """Turn a caveat paragraph into its own amber box.

    A caveat is not part of the result it follows - it is the thing the reader
    has to notice before acting - so it gets a box of its own instead of being
    filed as one more field at the bottom of the green one. The bold lead of
    the paragraph becomes the box title; everything after it is the body.
    """
    heads = [index for index, line in enumerate(chunk) if _warning_head(line) is not None]
    if not heads:
        return None
    first = heads[0]
    if len(heads) > 1:
        # Several caveats in a row share one box and keep their own bold leads:
        # promoting the first one's lead would let it speak for all the others.
        block = RichBlock(title=WARNING_SIGN, color=RECAP_COLOR)
        _add_field(block, INVISIBLE_NAME, "\n".join(chunk[first:]).strip())
        return block
    head = (_warning_head(chunk[first]) or "")[len(WARNING_MARKER) :].lstrip("\uFE0F").lstrip()
    lead, remainder = _split_leading_bold(head)
    if lead is None:
        title = WARNING_SIGN
        body_lines = [head] + list(chunk[first + 1 :])
    else:
        title = _clip(WARNING_SIGN + " " + _block_title(lead).rstrip("."), MAX_TITLE)
        body_lines = ([remainder] if remainder.strip() else []) + list(chunk[first + 1 :])
    block = RichBlock(title=title, color=RECAP_COLOR)
    _add_field(block, INVISIBLE_NAME, "\n".join(body_lines))
    return block


def _build_result_blocks(lines: List[str], title_index: int, end: int) -> List[RichBlock]:
    """Build the green result box plus one amber box per standalone caveat.

    The completed-steps recap ("Feito:") stays inside the green box: it is the
    evidence behind the TL;DR and reads as part of the same statement. What
    moves out is every ``WARNING`` paragraph, which is a separate signal and
    used to disappear at the bottom of a long green box.
    """
    block = RichBlock(
        title=_block_title(lines[title_index]),
        color=RESULT_COLOR,
    )
    body = _drop_media_lines(lines[title_index + 1 : end])
    sections = _sections(body)

    # The TL;DR only ever lives in the leading unlabeled chunk; lifting it into
    # the first field is what gives it the standalone look inside the box.
    if sections and sections[0][0] is None:
        tldr, remainder = _extract_tldr(sections[0][1])
        sections[0] = (None, remainder)
    else:
        tldr = None
    if tldr:
        _add_field(block, "TL;DR", tldr)

    warnings: List[RichBlock] = []
    for label, chunk in sections:
        if label is None and _is_warning_chunk(chunk):
            warning = _build_warning_block(chunk)
            if warning is not None and not warning.is_empty():
                warnings.append(warning)
                continue
        _add_field(block, label or INVISIBLE_NAME, "\n".join(chunk))

    # Discord accepts 10 embeds per message; fold any excess caveat into the
    # last box that fits rather than dropping it.
    if len(warnings) > MAX_WARNING_BLOCKS:
        overflow = warnings[MAX_WARNING_BLOCKS - 1 :]
        warnings = warnings[: MAX_WARNING_BLOCKS - 1]
        merged = overflow[0]
        for extra in overflow[1:]:
            for item in extra.fields:
                _add_field(merged, item.name, item.value)
        warnings.append(merged)

    return [block] + warnings


def _build_handoff_block(lines: List[str], title_index: int, end: int) -> RichBlock:
    title_line, aside = _split_title_parenthetical(lines[title_index])
    body = "\n".join(_drop_media_lines(lines[title_index + 1 : end])).strip()
    block = RichBlock(title=_block_title(title_line), color=HANDOFF_COLOR)
    if not aside:
        block.description = _clip(_capitalize_first(body), MAX_DESCRIPTION) or None
        return block
    # With an aside the box switches to fields: a description always renders
    # above every field, which would put the body before its own subtitle. The
    # label is STATUS, not TL;DR: the parenthetical says whether the turn is
    # blocked, it is not a summary of the handoff below it.
    _add_field(block, HANDOFF_ASIDE_NAME, aside)
    _add_field(block, INVISIBLE_NAME, body)
    return block


def _locate(text: str):
    """Find the closing blocks in ``text``.

    Returns ``(lines, segments, result_index, handoff_index)`` or ``None``
    when the reply does not match the expected closing shape.
    """
    if not text or "**" not in text:
        return None

    lines = text.split("\n")
    segments = _split_segments(lines)
    if len(segments) < 2:
        return None

    # Index the last segment that actually has content; anything after the
    # handoff would have to be reordered, which we refuse to do.
    last_content = None
    for index in range(len(segments) - 1, -1, -1):
        separator, start, end = segments[index]
        if _first_content_index(lines, start, end) is not None:
            last_content = index
            break
    if last_content is None or last_content == 0:
        return None

    def head_index(index: int) -> Optional[int]:
        separator, start, end = segments[index]
        if separator < 0:
            return None
        return _first_content_index(lines, start, end)

    handoff_index = None
    title_line = head_index(last_content)
    if title_line is not None and lines[title_line].lstrip().startswith(HANDOFF_MARKER):
        handoff_index = last_content

    # The result block is the last titled segment above the handoff.
    result_index = None
    search_from = (handoff_index if handoff_index is not None else last_content + 1) - 1
    for index in range(search_from, 0, -1):
        separator, start, end = segments[index]
        first = _first_content_index(lines, start, end)
        if first is None:
            continue
        if not lines[first].lstrip().startswith(HANDOFF_MARKER) and _BLOCK_TITLE_RE.match(
            lines[first]
        ):
            result_index = index
        break  # only the segment directly above the handoff is eligible

    if result_index is None and handoff_index is None:
        return None
    return lines, segments, result_index, handoff_index


def closing_block_offset(text: str) -> Optional[int]:
    """Character offset where the closing blocks start, or ``None``.

    ``text[:offset]`` is the plain reasoning head and ``text[offset:]`` opens
    with the horizontal rule of the first colored block. The stream consumer
    uses this to split an oversized reply ON that boundary instead of at an
    arbitrary line, so a block is never cut in half between two messages.
    """
    located = _locate(text)
    if located is None:
        return None
    lines, segments, result_index, handoff_index = located
    first_extracted = result_index if result_index is not None else handoff_index
    separator_line = segments[first_extracted][0]
    if separator_line <= 0:
        return None
    offset = sum(len(line) + 1 for line in lines[:separator_line])
    if offset <= 0 or offset >= len(text):
        return None
    return offset


def split_rich_blocks(
    text: str, *, allow_empty_head: bool = False
) -> Tuple[str, List[RichBlock]]:
    """Split a final reply into its plain head and its colored closing blocks.

    Returns ``(text, [])`` unchanged whenever the reply does not match the
    expected shape, so an unusual reply is always delivered the old way rather
    than mangled.

    ``allow_empty_head`` is set by a caller that can deliver a message made of
    embeds alone (a fresh ``send``). The streamed-edit path leaves it off: an
    edit cannot empty a message's text.
    """
    located = _locate(text)
    if located is None:
        return text, []
    lines, segments, result_index, handoff_index = located

    def assemble(result_i: Optional[int], handoff_i: Optional[int]):
        first_extracted = result_i if result_i is not None else handoff_i
        head = "\n".join(lines[: segments[first_extracted][0]]).rstrip()
        blocks: List[RichBlock] = []
        if result_i is not None:
            _sep, start, end = segments[result_i]
            first = _first_content_index(lines, start, end)
            blocks.extend(_build_result_blocks(lines, first, end))
        if handoff_i is not None:
            _sep, start, end = segments[handoff_i]
            first = _first_content_index(lines, start, end)
            blocks.append(_build_handoff_block(lines, first, end))
        return head, [b for b in blocks if not b.is_empty()]

    head, blocks = assemble(result_index, handoff_index)
    if not blocks:
        return text, []
    if head.strip():
        return head, blocks
    if allow_empty_head:
        # The whole payload is closing blocks (the consumer split the reply on
        # the block boundary). Deliver them as embeds with no text at all.
        return "", blocks
    if result_index is not None and handoff_index is not None:
        # Nothing would be left in the streamed message. Keep the result block
        # as plain text there rather than emptying it.
        head, blocks = assemble(None, handoff_index)
        if head.strip() and blocks:
            return head, blocks
    return text, []
