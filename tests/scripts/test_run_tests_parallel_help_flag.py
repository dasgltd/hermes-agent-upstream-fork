"""``--help`` must print usage, never launch an accidental full-suite sweep.

The runner routes any unknown ``-``-token to the per-file pytest so that bare
``-q``/``-x``/``--tb=long`` "just work". ``-h``/``--help`` fell into that same
bucket: argparse never saw it, discovery ran, and the runner spawned one
``pytest --help`` per test file -- 4218 files on this repo, each exiting 0
without running a test. Asking for usage started a multi-hour sweep that
reported "no tests ran". Usage requests must short-circuit before discovery.

An explicit ``--`` separator still means verbatim passthrough, so a ``--help``
AFTER it belongs to pytest and must NOT be intercepted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = REPO_ROOT / "scripts" / "run_tests_parallel.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_parallel_help", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_main(monkeypatch, argv: list[str]):
    """Call ``main()`` with ``argv``, blowing up if discovery is reached."""
    mod = _load_runner()
    monkeypatch.setattr(sys, "argv", ["run_tests_parallel.py", *argv])

    def _explode(*_a, **_kw):
        raise AssertionError("discovery reached: the run was about to start")

    monkeypatch.setattr(mod, "_discover_files", _explode)
    return mod.main()


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_prints_usage_and_never_discovers(monkeypatch, capsys, flag: str) -> None:
    rc = _run_main(monkeypatch, [flag])
    assert rc == 0
    out = capsys.readouterr().out
    assert "usage: run_tests_parallel" in out
    assert "--paths" in out


def test_help_wins_even_next_to_other_flags(monkeypatch, capsys) -> None:
    # A usage request mixed with real options is still a usage request.
    rc = _run_main(monkeypatch, ["-j", "4", "--paths", "tests/gateway", "--help"])
    assert rc == 0
    assert "usage: run_tests_parallel" in capsys.readouterr().out


def test_help_after_double_dash_stays_pytest_passthrough(monkeypatch) -> None:
    # After ``--`` the developer asked for verbatim passthrough: do not
    # intercept, so the call must proceed and reach discovery.
    with pytest.raises(AssertionError, match="discovery reached"):
        _run_main(monkeypatch, ["--paths", "tests/gateway", "--", "--help"])


def test_ordinary_invocation_still_reaches_discovery(monkeypatch) -> None:
    # Guard against the fix swallowing normal runs.
    with pytest.raises(AssertionError, match="discovery reached"):
        _run_main(monkeypatch, ["-q", "--paths", "tests/gateway"])
