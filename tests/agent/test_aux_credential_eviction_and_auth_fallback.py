"""Two defects that turned one expired Vertex bearer into a multi-hour compression outage.

1. ``_evict_cached_clients`` read the PROFILE HOME (``key[0]``) instead of the provider
   (``key[1]``), so provider-scoped eviction never matched anything. Every OAuth-backed
   provider kept its stale client after a successful credential refresh, and the
   same-provider retry replayed the expired bearer. A ~1h Vertex token 401s with
   ``ACCESS_TOKEN_TYPE_UNSUPPORTED`` for the rest of the process lifetime.

2. ``_ladder_provider_fallback`` refused to walk the fallback chain on an auth error when
   the provider was named explicitly. The compression task has two configured spares and
   telemetry still reported ``fallback_used: false`` on every attempt.

Both tests were validated in reverse: with the fixes reverted they fail.
"""

import pytest

import agent.auxiliary_client as ac


class _FakeClient:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _real_key(provider: str, model: str = "m") -> tuple:
    """A cache key built by production code, not a hand-made tuple."""
    return ac._client_cache_key(
        provider, async_mode=False, base_url=None, api_key="", api_mode=None,
        main_runtime=None, is_vision=False, task="compression", model=model,
    )


@pytest.fixture(autouse=True)
def _clean_cache():
    ac._client_cache.clear()
    yield
    ac._client_cache.clear()


def test_provider_index_constant_matches_a_real_cache_key():
    """Pins the contract the eviction depends on: moving the provider breaks this first."""
    key = _real_key("vertex")
    assert key[ac._CACHE_KEY_PROVIDER_INDEX] == "vertex"
    assert key[0] != "vertex", "key[0] is the profile home, never the provider"


def test_eviction_actually_drops_the_stale_provider_client():
    key = _real_key("vertex")
    client = _FakeClient()
    ac._client_cache[key] = (client, "m", None)

    ac._evict_cached_clients("vertex")

    assert key not in ac._client_cache, "stale Vertex client survived the eviction"
    assert client.closed is True


def test_eviction_leaves_other_providers_alone():
    vertex_key, other_key = _real_key("vertex"), _real_key("anthropic")
    ac._client_cache[vertex_key] = (_FakeClient(), "m", None)
    ac._client_cache[other_key] = (_FakeClient(), "m", None)

    ac._evict_cached_clients("vertex")

    assert vertex_key not in ac._client_cache
    assert other_key in ac._client_cache


def test_configured_chain_detection(monkeypatch):
    monkeypatch.setattr(
        ac, "_get_auxiliary_task_config",
        lambda task: {"fallback_chain": [{"provider": "anthropic", "model": "x"}]},
    )
    assert ac._task_has_configured_fallback_chain("compression") is True

    monkeypatch.setattr(ac, "_get_auxiliary_task_config", lambda task: {"fallback_chain": []})
    assert ac._task_has_configured_fallback_chain("compression") is False

    monkeypatch.setattr(ac, "_get_auxiliary_task_config", lambda task: {})
    assert ac._task_has_configured_fallback_chain("compression") is False

    def _boom(task):
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(ac, "_get_auxiliary_task_config", _boom)
    assert ac._task_has_configured_fallback_chain("compression") is False


def _auth_error() -> Exception:
    exc = RuntimeError("Error code: 401 - ACCESS_TOKEN_TYPE_UNSUPPORTED")
    exc.status_code = 401
    return exc


def _route(provider: str) -> "ac._LadderRoute":
    return ac._LadderRoute(
        client=_FakeClient(), task="compression", tag="", async_mode=False,
        base_info="https://aiplatform.googleapis.com/v1beta1/x", resolved_provider=provider,
        resolved_model="google/gemini-3.1-pro-preview", resolved_base_url=None,
        resolved_api_key="", resolved_api_mode=None,
        final_model="google/gemini-3.1-pro-preview", main_runtime={}, route_info=None,
    )


def _drive_until_first_step(gen):
    """Advance the ladder generator to its first requested step, or None if it gives up."""
    try:
        return next(gen)
    except StopIteration:
        return None


def test_auth_error_on_an_explicit_provider_reaches_the_configured_chain(monkeypatch):
    spare = _FakeClient()
    monkeypatch.setattr(ac, "_task_has_configured_fallback_chain", lambda task: True)
    monkeypatch.setattr(
        ac, "_try_configured_fallback_chain",
        lambda *a, **kw: (spare, "claude-haiku-4-5", "fallback_chain[0](anthropic)"),
    )
    monkeypatch.setattr(ac, "_try_main_agent_model_fallback", lambda *a, **kw: (None, None, ""))

    step = _drive_until_first_step(
        ac._ladder_provider_fallback(_auth_error(), _route("vertex"))
    )

    assert step is not None, "an explicit provider's 401 never reached the configured chain"
    assert step.kind == "fallback"
    assert step.args[0] is spare


def test_auth_error_without_a_configured_chain_still_stays_on_the_provider(monkeypatch):
    """The widening is scoped: no configured chain means the old, stricter behaviour holds."""
    monkeypatch.setattr(ac, "_task_has_configured_fallback_chain", lambda task: False)
    monkeypatch.setattr(
        ac, "_try_configured_fallback_chain", lambda *a, **kw: (_FakeClient(), "m", "x"))
    monkeypatch.setattr(
        ac, "_try_main_agent_model_fallback", lambda *a, **kw: (_FakeClient(), "m", "y"))

    step = _drive_until_first_step(
        ac._ladder_provider_fallback(_auth_error(), _route("vertex"))
    )

    assert step is None, "a 401 escaped to fallback with no chain configured for the task"


def test_explicit_provider_auth_error_never_spills_onto_the_main_agent_model(monkeypatch):
    """The task's own chain may be walked; the main-agent-model net must not be."""
    main_net = _FakeClient()
    monkeypatch.setattr(ac, "_task_has_configured_fallback_chain", lambda task: True)
    monkeypatch.setattr(ac, "_try_configured_fallback_chain", lambda *a, **kw: (None, None, ""))
    monkeypatch.setattr(
        ac, "_try_main_agent_model_fallback", lambda *a, **kw: (main_net, "m", "main-agent(x)"))

    step = _drive_until_first_step(
        ac._ladder_provider_fallback(_auth_error(), _route("vertex"))
    )

    assert step is None, "a 401 on an explicit provider leaked onto the main agent model"
