"""Tests for the Discord rich closing-block parser.

The parser is pure, so these run without discord.py installed.
"""

from plugins.platforms.discord.rich_blocks import (
    HANDOFF_COLOR,
    MAX_FIELD_VALUE,
    RECAP_COLOR,
    RESULT_COLOR,
    closing_block_offset,
    split_rich_blocks,
)

HANDOFF = "\U0001F449"
TLDR = "\U0001F3AF"
CHECK = "✅"
WARN = "⚠️"

SAMPLE = "\n".join(
    [
        "**Thinking...**",
        "> looked at the adapter",
        "",
        "⏳ Worked for 43s",
        "",
        "------",
        "",
        CHECK + " **it works**",
        "",
        "- **TL;DR** — the colored box is an embed,",
        "  not markdown.",
        "- second bullet with `inline code`",
        "",
        "**Feito:**",
        "- i. wrote the parser",
        "- ii. ran the tests",
        "",
        WARN + " one caveat worth reading",
        "",
        "------",
        "",
        HANDOFF + " **SUA VEZ (nada me bloqueia)**:",
        "1. ship it? → **a)** yes · **b)** no",
    ]
)


def _names(block):
    return [f.name for f in block.fields]


def test_sample_splits_into_head_and_three_blocks():
    head, blocks = split_rich_blocks(SAMPLE)
    assert "Worked for 43s" in head
    assert "it works" not in head and HANDOFF not in head
    assert not head.rstrip().endswith("-")  # the rule itself is consumed
    # green result (recap included), amber caveat, red handoff.
    assert [b.color for b in blocks] == [RESULT_COLOR, RECAP_COLOR, HANDOFF_COLOR]
    assert blocks[1].title.startswith(WARN)


def test_result_block_puts_tldr_in_the_first_field():
    _head, blocks = split_rich_blocks(SAMPLE)
    result = blocks[0]
    assert result.title == CHECK + " IT WORKS"
    assert result.fields[0].name == "TL;DR"
    # The wrapped continuation line is folded into the same field.
    assert "not markdown." in result.fields[0].value
    assert "TL;DR" not in result.fields[1].value


def test_recap_stays_inside_the_green_box():
    """The recap is the evidence behind the TL;DR, so it reads in the same box."""
    _head, blocks = split_rich_blocks(SAMPLE)
    assert "Feito" in _names(blocks[0])
    recap = next(f for f in blocks[0].fields if f.name == "Feito")
    assert "i. wrote the parser" in recap.value
    assert "caveat" not in recap.value  # the caveat is boxed on its own


def test_caveat_becomes_its_own_amber_box():
    _head, blocks = split_rich_blocks(SAMPLE)
    assert all("caveat" not in f.value for f in blocks[0].fields)
    caveat = blocks[1]
    assert caveat.color == RECAP_COLOR
    assert caveat.title.startswith(WARN)
    assert "caveat" in caveat.fields[0].value


def test_caveat_bold_lead_becomes_the_box_title():
    text = SAMPLE.replace(
        WARN + " one caveat worth reading",
        WARN + " **nao e um problema:** so um aviso solto",
    )
    _head, blocks = split_rich_blocks(text)
    caveat = blocks[1]
    assert caveat.title == WARN + " NAO E UM PROBLEMA"
    assert caveat.fields[0].value == "So um aviso solto"


def test_several_caveats_each_get_a_box():
    text = SAMPLE.replace(
        WARN + " one caveat worth reading",
        WARN + " first caveat\n\n" + WARN + " second caveat",
    )
    _head, blocks = split_rich_blocks(text)
    assert [b.color for b in blocks] == [
        RESULT_COLOR, RECAP_COLOR, RECAP_COLOR, HANDOFF_COLOR,
    ]


def test_reply_without_a_recap_still_yields_its_boxes():
    text = SAMPLE.replace("**Feito:**\n", "").replace(
        "- i. wrote the parser\n- ii. ran the tests\n", ""
    )
    _head, blocks = split_rich_blocks(text)
    assert [b.color for b in blocks] == [RESULT_COLOR, RECAP_COLOR, HANDOFF_COLOR]


def test_reply_without_a_caveat_yields_two_boxes():
    text = SAMPLE.replace(WARN + " one caveat worth reading\n", "")
    _head, blocks = split_rich_blocks(text)
    assert [b.color for b in blocks] == [RESULT_COLOR, HANDOFF_COLOR]


def test_field_values_open_with_a_capital_letter():
    text = SAMPLE.replace(
        "- **TL;DR** - the colored box is an embed,",
        "- **TL;DR** - the colored box is an embed,",
    )
    _head, blocks = split_rich_blocks(text)
    assert blocks[0].fields[0].value[0] == "t".upper()


def test_a_technical_first_token_is_never_capitalized():
    text = "\n".join(
        [
            "head",
            "",
            "------",
            "",
            CHECK + " **x**",
            "",
            "- **TL;DR** - n8n stays lower case",
        ]
    )
    _head, blocks = split_rich_blocks(text)
    assert blocks[0].fields[0].value.startswith("n8n")


def test_handoff_title_aside_becomes_a_status_field():
    _head, blocks = split_rich_blocks(SAMPLE)
    handoff = blocks[-1]
    assert handoff.color == HANDOFF_COLOR
    assert handoff.title == HANDOFF + " SUA VEZ"
    assert handoff.description is None
    # The red box states whether the turn is blocked; that is a status, and
    # labeling it TL;DR read as a summary the handoff never had.
    assert handoff.fields[0].name == "STATUS"
    assert handoff.fields[0].value == "Nada me bloqueia"
    assert all(item.name != "TL;DR" for item in handoff.fields)
    assert "1. ship it?" in handoff.fields[1].value
    assert "**a)**" in handoff.fields[1].value


def test_handoff_without_an_aside_keeps_the_description():
    text = SAMPLE.replace(" (nada me bloqueia)", "")
    _head, blocks = split_rich_blocks(text)
    handoff = blocks[-1]
    assert handoff.title == HANDOFF + " SUA VEZ"
    assert "1. ship it?" in handoff.description
    assert not handoff.fields


def test_handoff_only_reply_still_gets_a_red_box():
    text = "some prose\n\n------\n\n" + HANDOFF + " **SUA VEZ**:\n1. go? → **a)** ok"
    head, blocks = split_rich_blocks(text)
    assert head.strip() == "some prose"
    assert [b.color for b in blocks] == [HANDOFF_COLOR]


def test_plain_reply_is_returned_untouched():
    for text in ("", "just a sentence", "a\n\nb", "**bold** but no rule"):
        assert split_rich_blocks(text) == (text, [])


def test_reply_that_would_leave_an_empty_head_is_not_gutted():
    # Only a rule then a result block: extracting it would empty the message.
    text = "------\n\n" + CHECK + " **x**\n\n- **TL;DR** — y"
    assert split_rich_blocks(text) == (text, [])


def test_media_directives_never_land_inside_a_box():
    text = SAMPLE.replace(
        WARN + " one caveat worth reading",
        "MEDIA:/home/dasg/report.md",
    )
    _head, blocks = split_rich_blocks(text)
    assert all("MEDIA:" not in f.value for f in blocks[0].fields)


def test_oversized_section_is_chunked_across_fields():
    long_body = "\n".join("- bullet %d with some padding text" % i for i in range(120))
    text = "\n".join(
        [
            "head",
            "",
            "------",
            "",
            CHECK + " **big**",
            "",
            "**Feito:**",
            long_body,
        ]
    )
    _head, blocks = split_rich_blocks(text)
    values = [f.value for f in blocks[0].fields]
    assert len(values) > 1
    assert all(len(v) <= MAX_FIELD_VALUE for v in values)
    assert blocks[0].fields[1].name == "​"  # continuation is unlabeled


def test_text_after_the_handoff_is_not_reordered():
    text = SAMPLE + "\n\n------\n\nsomething trailing"
    head, blocks = split_rich_blocks(text)
    # The trailing segment is not a handoff, so nothing is extracted from it.
    assert "something trailing" in head or not blocks


def test_legacy_result_prefix_and_dart_still_parse():
    """Old replies (RESULT: title, dart-prefixed TL;DR) must keep rendering."""
    text = (
        "head line\n\n"
        "------\n\n"
        + CHECK
        + " **RESULT: legacy title**\n\n"
        "- "
        + TLDR
        + " **TL;DR** - legacy bullet\n"
    )
    head, blocks = split_rich_blocks(text)
    assert head.strip() == "head line"
    assert len(blocks) == 1
    assert blocks[0].title == CHECK + " RESULT: LEGACY TITLE"
    assert blocks[0].fields[0].name == "TL;DR"
    assert blocks[0].fields[0].value == "Legacy bullet"


def test_title_is_upper_cased_but_keeps_code_tokens_lowercase():
    """Ticks are dropped, prose is CAPS, and the code token keeps its case."""
    text = (
        "head line\n\n"
        "------\n\n"
        + CHECK
        + " **`docling` desligado e `config.yaml` enxuto**\n\n"
        "- **TL;DR** - feito\n"
    )
    _head, blocks = split_rich_blocks(text)
    assert blocks[0].title == CHECK + " docling DESLIGADO E config.yaml ENXUTO"
    assert "`" not in blocks[0].title


def test_title_upper_case_keeps_accents():
    text = (
        "head line\n\n"
        "------\n\n"
        + CHECK
        + " **nao esta pronto**\n\n"
        "- **TL;DR** - ok\n"
    ).replace("nao esta", "n\u00e3o est\u00e1")
    _head, blocks = split_rich_blocks(text)
    assert blocks[0].title == CHECK + " N\u00c3O EST\u00c1 PRONTO"


def test_closing_block_offset_marks_the_rule_that_opens_the_first_box():
    offset = closing_block_offset(SAMPLE)
    assert offset is not None
    assert SAMPLE[:offset].rstrip().endswith("Worked for 43s")
    assert SAMPLE[offset:].startswith("------")


def test_closing_block_offset_is_none_without_a_closing_block():
    assert closing_block_offset("just a sentence") is None
    assert closing_block_offset("------\n\n" + CHECK + " **x**\n\n- y") is None


def test_split_on_the_offset_still_yields_both_boxes_from_the_tail():
    """An oversized reply cut on the offset keeps every box in the tail."""
    offset = closing_block_offset(SAMPLE)
    tail = SAMPLE[offset:]
    head, blocks = split_rich_blocks(tail, allow_empty_head=True)
    assert head == ""
    assert [b.color for b in blocks] == [RESULT_COLOR, RECAP_COLOR, HANDOFF_COLOR]
    assert blocks[0].fields[0].name == "TL;DR"


def test_edit_path_never_empties_the_message_it_edits():
    """Without allow_empty_head the result stays as text so the edit has body."""
    offset = closing_block_offset(SAMPLE)
    tail = SAMPLE[offset:]
    head, blocks = split_rich_blocks(tail)
    assert head.strip()
    assert [b.color for b in blocks] == [HANDOFF_COLOR]


def test_caveat_written_as_a_bullet_still_leaves_the_green_box():
    """A caveat is a caveat whether or not a dash sits in front of it."""
    text = (
        "**Thinking...**\n> olhei o renderizador\n\n"
        "------\n\n"
        + CHECK
        + " **release conferido**\n\n"
        "- **TL;DR** - tudo pronto\n"
        "- " + WARN + " **A API nao devolve cabecalho de seguranca** (o site devolve).\n"
    )
    _head, blocks = split_rich_blocks(text)
    assert [b.color for b in blocks] == [RESULT_COLOR, RECAP_COLOR]
    green, amber = blocks
    assert "cabecalho de seguranca" not in "".join(f.value for f in green.fields)
    # The bold lead is promoted to the title, exactly as for a paragraph caveat.
    assert amber.title == WARN + " A API NAO DEVOLVE CABECALHO DE SEGURANCA"
    assert amber.fields[0].value == "(o site devolve)."


def test_a_run_of_caveat_bullets_shares_one_amber_box():
    """Three caveats in a row are one signal, not three boxes each claiming a title."""
    text = (
        "**Thinking...**\n> olhei o renderizador\n\n"
        "------\n\n"
        + CHECK
        + " **release conferido**\n\n"
        "- **TL;DR** - tudo pronto\n"
        "- " + WARN + " **Primeiro aviso** corpo um.\n"
        "- " + WARN + " **Segundo aviso** corpo dois.\n"
        "- " + WARN + " **Terceiro aviso** corpo tres.\n"
    )
    _head, blocks = split_rich_blocks(text)
    assert [b.color for b in blocks] == [RESULT_COLOR, RECAP_COLOR]
    amber = blocks[1]
    assert amber.title == WARN
    body = amber.fields[0].value
    for lead in ("Primeiro aviso", "Segundo aviso", "Terceiro aviso"):
        assert lead in body


def test_a_caveat_bullet_inside_a_recap_stays_in_the_recap():
    """Ripping an item out of a numbered recap would leave the list with a hole."""
    text = (
        "**Thinking...**\n> olhei o renderizador\n\n"
        "------\n\n"
        + CHECK
        + " **release conferido**\n\n"
        "- **TL;DR** - tudo pronto\n\n"
        "**Feito:**\n"
        "- i. primeiro passo\n"
        "- " + WARN + " ii. passo que ficou pela metade\n"
        "- iii. terceiro passo\n"
    )
    _head, blocks = split_rich_blocks(text)
    assert [b.color for b in blocks] == [RESULT_COLOR]
    recap = "\n".join(f.value for f in blocks[0].fields)
    assert "primeiro passo" in recap
    assert "pela metade" in recap
    assert "terceiro passo" in recap
