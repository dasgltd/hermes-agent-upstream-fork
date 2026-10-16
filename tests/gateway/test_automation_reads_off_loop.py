"""Automation storage reads never monopolize the gateway event-loop thread."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from hermes_state_runtime import list_session_admissions
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.asyncio
async def test_completion_batch_reads_off_loop_without_readmitting_duplicates(monkeypatch):
    runner = GatewayRunner(GatewayConfig())
    key = ('fixture',)
    runner.session_authority = object()
    runner._completion_notification_batch_window = 0
    runner._completion_notification_batch_tasks = {}
    loop_thread = threading.get_ident()
    seen = []

    def admitted(_runner, event):
        seen.append((event['name'], threading.get_ident()))
        return {'admission_id': 'prior'} if event['name'] == 'duplicate' else None

    monkeypatch.setattr('gateway.session_automation.completion_admission', admitted)
    monkeypatch.setattr('gateway.session_automation.producer_identity', lambda _runner, evt: evt['name'])
    runner._deliver_completion_notification = AsyncMock(return_value=True)
    runner._record_coalesced_completion_siblings = lambda _events: None
    entries = [('fixture', {'name': name}, asyncio.get_running_loop().create_future())
               for name in ('duplicate', 'fresh')]
    runner._completion_notification_batches = {key: entries}
    await runner._flush_process_completion_batch(key)
    assert seen and all(thread != loop_thread for _, thread in seen), 'completion reads blocked the owner loop'
    assert [name for name, _ in seen] == ['duplicate', 'fresh']
    assert runner._deliver_completion_notification.await_args.args[1]['name'] == 'fresh'
    assert all(waiter.result() is True for _, _, waiter in entries)


@pytest_asyncio.fixture
async def messaging(monkeypatch):
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='read-test')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    human = MessageEvent(text='human', source=source, message_id='first')
    with native_callback(runner, human, get_hermes_home()):
        receipt = await authority.admit_native(human)
    route = runner.session_store._generate_session_key(source)
    event = MessageEvent(text='finished', source=source, internal=True,
                         metadata={'gateway_session_key': route, 'gateway_session_id': receipt.ref.session_id})
    try:
        yield SimpleNamespace(runner=runner, adapter=adapter, authority=authority, event=event,
                              sid=receipt.ref.session_id)
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_automation_snapshot_reads_off_loop_and_still_commits_to_its_owner(messaging, monkeypatch):
    owner = messaging.authority
    loop_thread = threading.get_ident()
    reads = []
    real = owner.db.get_compression_lineage

    def lineage(sid):
        reads.append(threading.get_ident())
        return real(sid)

    monkeypatch.setattr(owner.db, 'get_compression_lineage', lineage)
    receipt = await owner.admit_automation(messaging.adapter, messaging.event, 'finished')
    assert receipt.ref.session_id == messaging.sid
    assert reads and all(thread != loop_thread for thread in reads), 'snapshot read storage on the owner loop'
    assert [row['request_id'] for row in list_session_admissions(owner.db, session_id=messaging.sid)] == [
        'first', 'finished']
