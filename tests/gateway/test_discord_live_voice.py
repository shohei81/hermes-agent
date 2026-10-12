"""Tests for the Discord GPT-Live voice bridge (discord.voice_realtime with a gpt-live-* model).

The bridge (plugins/platforms/discord/live_voice.py) is exercised with a fake
WebSocket that replies to session.start with session.started.
"""

import asyncio
import base64
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

np = pytest.importorskip("numpy")

_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)
if _DISCORD_DIR not in sys.path:
    sys.path.insert(0, _DISCORD_DIR)

import live_voice as lv
import realtime_voice as rv


class FakeLiveWS:
    def __init__(self, first_reply=None):
        self.sent = []
        self._first_reply = first_reply or {"type": "session.started", "session": {}}

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def recv(self):
        return json.dumps(self._first_reply)

    async def close(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)
        raise StopAsyncIteration


async def _started_bridge(delegate=None, **kwargs):
    ws = FakeLiveWS()

    async def connect(url, api_key):
        assert url == lv.LIVE_URL
        assert api_key == "sk-test"
        return ws

    bridge = lv.LiveVoiceBridge(
        api_key="sk-test",
        delegate=delegate or AsyncMock(return_value="done"),
        connect=connect,
        **kwargs,
    )
    await bridge.start()
    return bridge, ws


class TestLiveVoiceBridge:
    @pytest.mark.asyncio
    async def test_start_uses_client_delegation(self):
        bridge, ws = await _started_bridge(instructions="Speak calmly.", context="Your name is poi.")
        try:
            start = ws.sent[0]
            assert start["type"] == "session.start"
            session = start["session"]
            assert session["model"] == "gpt-live-1"
            assert session["delegation"] == {"type": "client"}
            assert session["audio"] == {
                "format": {"type": "audio/pcm", "rate": 24000},
                "output": {"voice": "marin"},
            }
            assert session["instructions"] == "\n\n".join(
                [lv.DEFAULT_INSTRUCTIONS, "Speak calmly.", "Your name is poi."]
            )
            assert bridge.running
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_start_raises_on_error(self):
        ws = FakeLiveWS(first_reply={"type": "error", "error": {"message": "bad model"}})

        async def connect(url, api_key):
            return ws

        bridge = lv.LiveVoiceBridge(api_key="k", delegate=AsyncMock(), connect=connect)
        with pytest.raises(RuntimeError, match="bad model"):
            await bridge.start()

    @pytest.mark.asyncio
    async def test_send_audio_appends_downsampled_pcm(self):
        bridge, ws = await _started_bridge()
        try:
            await bridge.send_audio(b"\x00" * rv.FRAME_BYTES)
            event = ws.sent[-1]
            assert event["type"] == "session.input_audio.append"
            assert len(base64.b64decode(event["audio"])) == rv.FRAME_BYTES // 4
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_output_audio_is_queued_for_playback(self):
        bridge, _ws = await _started_bridge()
        try:
            await bridge._handle_event({
                "type": "session.output_audio.delta",
                "delta": base64.b64encode(b"\x00" * 960).decode(),
            })
            assert len(bridge.source.read()) == rv.FRAME_BYTES
            assert bridge.source.played_bytes == rv.FRAME_BYTES
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_delegation_sends_transcript_and_returns_commentary(self, monkeypatch):
        monkeypatch.setattr(lv, "DELEGATION_SETTLE_S", 0)
        delegate = AsyncMock(return_value="digest には新着が 3 件あるよ。")
        bridge, ws = await _started_bridge(delegate=delegate)
        try:
            for etype, delta in [
                ("session.input_transcript.delta", "ねえ、"),
                ("session.input_transcript.delta", "digest 見といて"),
                ("session.output_transcript.delta", "ちょっと見てみるね"),
            ]:
                await bridge._handle_event({"type": etype, "delta": delta})
            await bridge._handle_event({
                "type": "session.delegation.created",
                "delegation": {"id": "item_1", "type": "delegation", "target": "client"},
            })
            await asyncio.gather(*bridge._tool_tasks)

            task = delegate.await_args.args[0]
            assert "User: ねえ、digest 見といて" in task
            assert "You (voice): ちょっと見てみるね" in task
            assert ws.sent[-1] == {
                "type": "session.commentary.append",
                "delegation_id": "item_1",
                "content": "digest には新着が 3 件あるよ。",
            }
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_delegation_failure_is_reported(self, monkeypatch):
        monkeypatch.setattr(lv, "DELEGATION_SETTLE_S", 0)
        bridge, ws = await _started_bridge(delegate=AsyncMock(side_effect=RuntimeError("boom")))
        try:
            await bridge._handle_event({"type": "session.delegation.created", "delegation": {"id": "d"}})
            await asyncio.gather(*bridge._tool_tasks)
            assert ws.sent[-1]["delegation_id"] == "d"
            assert "boom" in ws.sent[-1]["content"]
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_commentary_is_capped(self, monkeypatch):
        monkeypatch.setattr(lv, "DELEGATION_SETTLE_S", 0)
        bridge, ws = await _started_bridge(delegate=AsyncMock(return_value="あ" * 5000))
        try:
            await bridge._handle_event({"type": "session.delegation.created", "delegation": {"id": "d"}})
            await asyncio.gather(*bridge._tool_tasks)
            assert len(ws.sent[-1]["content"]) == lv.MAX_COMMENTARY_CHARS
        finally:
            await bridge.close()

    def test_transcript_keeps_only_recent_turns(self):
        bridge = lv.LiveVoiceBridge(api_key="k", delegate=AsyncMock())
        for i in range(lv.TRANSCRIPT_TURNS * 10):
            bridge._add_transcript("user" if i % 2 == 0 else "assistant", f"t{i}")
        assert len(bridge._transcript) <= lv.TRANSCRIPT_TURNS * 4
        task = bridge._task_from_transcript()
        assert f"t{lv.TRANSCRIPT_TURNS * 10 - 1}" in task
        assert "t0\n" not in task


class TestAdapterPicksLiveBridge:
    @pytest.mark.asyncio
    async def test_gpt_live_model_uses_live_bridge(self, monkeypatch):
        from plugins.platforms.discord.adapter import DiscordAdapter

        monkeypatch.setenv("HERMES_REALTIME_OPENAI_API_KEY", "sk-test")
        adapter = object.__new__(DiscordAdapter)
        adapter._voice_delegate_callback = AsyncMock()
        adapter._voice_sources = {}
        adapter._voice_realtime_bridges = {}
        adapter._voice_listen_tasks = {}
        adapter._voice_text_channels = {}
        adapter._client = None
        adapter._load_voice_realtime_config = lambda: {
            "enabled": True, "model": "gpt-live-1", "voice": "marin", "instructions": "",
        }
        adapter._realtime_voice_context = lambda gid, vc: ""
        adapter._realtime_voice_loop = AsyncMock()
        started = []

        async def fake_start(self):
            started.append(type(self).__name__)

        monkeypatch.setattr(lv.LiveVoiceBridge, "start", fake_start)
        vc = MagicMock()
        vc.is_playing.return_value = False

        assert await adapter._start_realtime_voice(111, vc) is True
        assert started == ["LiveVoiceBridge"]
        vc.play.assert_called_once_with(adapter._voice_realtime_bridges[111].source)


class TestVoiceThread:
    def _adapter(self, channel):
        from plugins.platforms.discord.adapter import DiscordAdapter

        adapter = object.__new__(DiscordAdapter)
        adapter._voice_text_channels = {111: 123}
        adapter._voice_sources = {111: {"platform": "discord", "chat_id": "123", "chat_type": "group", "user_id": "9"}}
        adapter._client = MagicMock()
        adapter._client.get_channel = MagicMock(return_value=channel)
        return adapter

    @pytest.mark.asyncio
    async def test_call_gets_its_own_thread_session(self):
        from types import SimpleNamespace

        channel = MagicMock()
        channel.create_thread = AsyncMock(return_value=SimpleNamespace(id=456, name="Voice — 10/13 15:00"))
        adapter = self._adapter(channel)

        await adapter._open_voice_thread(111)

        assert channel.create_thread.await_args.kwargs["name"].startswith("Voice — ")
        assert adapter._voice_text_channels[111] == 456
        assert adapter._voice_sources[111] == {
            "platform": "discord", "chat_id": "456", "chat_name": "Voice — 10/13 15:00",
            "chat_type": "thread", "user_id": "9", "thread_id": "456", "parent_chat_id": "123",
        }

    @pytest.mark.asyncio
    async def test_thread_failure_keeps_text_channel(self):
        channel = MagicMock()
        channel.create_thread = AsyncMock(side_effect=RuntimeError("missing permissions"))
        adapter = self._adapter(channel)

        await adapter._open_voice_thread(111)

        assert adapter._voice_text_channels[111] == 123
        assert adapter._voice_sources[111]["chat_id"] == "123"

    @pytest.mark.asyncio
    async def test_channels_without_threads_are_left_alone(self):
        adapter = self._adapter(object())  # e.g. a DM channel

        await adapter._open_voice_thread(111)

        assert adapter._voice_text_channels[111] == 123
