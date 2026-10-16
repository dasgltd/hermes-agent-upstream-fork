"""Async launch-profile entry prepares file-backed policy off the owner loop."""
import threading

import pytest

from agent.secret_scope import current_secret_scope, get_secret, set_multiplex_active
from hermes_constants import get_hermes_home_override
from tools.terminal_scope import get_terminal_scope, terminal_env
from tui_gateway import launch_profile_policy as policy


@pytest.mark.asyncio
async def test_async_launch_scope_reads_off_loop_and_restores_context(tmp_path, monkeypatch):
    home = tmp_path / 'launch'
    home.mkdir()
    (home / '.env').write_text('FIXTURE_SCOPE_VALUE=from-file\n')
    (home / 'config.yaml').write_text('terminal:\n  env: local\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(policy, '_snapshot', None)
    before = (get_hermes_home_override(), current_secret_scope(), get_terminal_scope())
    loop_thread = threading.get_ident()
    reads = []
    real = policy.launch_secret_scope

    def load(path):
        reads.append(threading.get_ident())
        return real(path)

    monkeypatch.setattr(policy, 'launch_secret_scope', load)
    set_multiplex_active(True)
    try:
        with pytest.raises(ValueError, match='fixture exit'):
            async with policy.async_launch_profile_scope_if_multiplexed():
                assert get_hermes_home_override() == str(home)
                assert get_secret('FIXTURE_SCOPE_VALUE') == 'from-file'
                assert terminal_env('TERMINAL_ENV') == 'local'
                raise ValueError('fixture exit')
        assert (get_hermes_home_override(), current_secret_scope(), get_terminal_scope()) == before
        assert reads and all(thread != loop_thread for thread in reads), 'profile files were read on the owner loop'
    finally:
        set_multiplex_active(False)
