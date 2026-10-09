"""Tests for the Discord Realtime voice mode (discord.voice_realtime).

The bridge (plugins/platforms/discord/realtime_voice.py) is exercised with a
fake WebSocket; adapter and runner wiring use the ``object.__new__`` pattern
from the rest of the voice suite.
"""

import asyncio
import base64
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

np = pytest.importorskip("numpy")

_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)
if _DISCORD_DIR not in sys.path:
    sys.path.insert(0, _DISCORD_DIR)

import realtime_voice as rv  # noqa: E402

from gateway.platforms.base import MessageType  # noqa: E402


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def close(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)
        raise StopAsyncIteration


def _pcm(*samples):
    return np.array(samples, dtype="<i2").tobytes()


async def _started_bridge(delegate=None):
    ws = FakeWS()

    async def connect(url, api_key):
        assert api_key == "sk-test"
        return ws

    bridge = rv.RealtimeVoiceBridge(
        api_key="sk-test",
        delegate=delegate or AsyncMock(return_value="done"),
        connect=connect,
    )
    await bridge.start()
    return bridge, ws


# =====================================================================
# Audio conversion + playback source
# =====================================================================

class TestAudioConversion:
    def test_discord_frame_to_realtime_is_quarter_size(self):
        assert len(rv.discord_to_realtime(b"\x00" * rv.FRAME_BYTES)) == rv.FRAME_BYTES // 4

    def test_realtime_to_discord_is_four_times_size(self):
        assert len(rv.realtime_to_discord(b"\x00" * 960)) == rv.FRAME_BYTES

    def test_round_trip_preserves_constant_signal(self):
        realtime = rv.realtime_to_discord(_pcm(1000, -500))
        assert np.frombuffer(realtime, dtype="<i2").tolist() == [1000] * 4 + [-500] * 4
        assert np.frombuffer(rv.discord_to_realtime(realtime), dtype="<i2").tolist() == [1000, -500]

    def test_stereo_channels_are_averaged(self):
        # two stereo frames (L, R): (100, 300), (100, 300) -> one mono sample 200
        assert np.frombuffer(rv.discord_to_realtime(_pcm(100, 300, 100, 300)), dtype="<i2").tolist() == [200]


class TestRealtimeAudioSource:
    def test_read_returns_silence_when_empty(self):
        src = rv.RealtimeAudioSource()
        assert src.read() == b"\x00" * rv.FRAME_BYTES
        assert src.played_bytes == 0

    def test_read_pads_partial_frame_and_counts_played(self):
        src = rv.RealtimeAudioSource()
        src.feed(b"\x01" * 100)
        frame = src.read()
        assert len(frame) == rv.FRAME_BYTES
        assert frame[:100] == b"\x01" * 100
        assert src.played_bytes == 100

    def test_clear_returns_unplayed_bytes(self):
        src = rv.RealtimeAudioSource()
        src.feed(b"\x01" * (rv.FRAME_BYTES * 2))
        src.read()
        assert src.clear() == rv.FRAME_BYTES
        assert src.read() == b"\x00" * rv.FRAME_BYTES


# =====================================================================
# Bridge protocol
# =====================================================================

class TestRealtimeVoiceBridge:
    @pytest.mark.asyncio
    async def test_start_configures_session_with_ask_hermes(self):
        bridge, ws = await _started_bridge()
        try:
            session = ws.sent[0]["session"]
            assert ws.sent[0]["type"] == "session.update"
            assert session["model"] == "gpt-realtime-2.1-mini"
            assert session["audio"]["input"]["turn_detection"] == {"type": "semantic_vad"}
            assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
            assert [t["name"] for t in session["tools"]] == ["ask_hermes"]
            assert bridge.running
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_send_audio_appends_downsampled_pcm(self):
        bridge, ws = await _started_bridge()
        try:
            await bridge.send_audio(b"\x00" * rv.FRAME_BYTES)
            msg = ws.sent[-1]
            assert msg["type"] == "input_audio_buffer.append"
            assert len(base64.b64decode(msg["audio"])) == rv.FRAME_BYTES // 4
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_output_audio_delta_is_queued_for_playback(self):
        bridge, _ws = await _started_bridge()
        try:
            await bridge._handle_event({
                "type": "response.output_audio.delta",
                "item_id": "item_1",
                "delta": base64.b64encode(b"\x00" * 960).decode(),
            })
            assert bridge.source.clear() == rv.FRAME_BYTES
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_function_call_delegates_and_returns_output(self):
        delegate = AsyncMock(return_value="明日は晴れです")
        bridge, ws = await _started_bridge(delegate)
        try:
            await bridge._handle_event({
                "type": "response.done",
                "response": {"output": [{
                    "type": "function_call",
                    "name": "ask_hermes",
                    "call_id": "call_1",
                    "arguments": json.dumps({"task": "明日の天気"}),
                }]},
            })
            await asyncio.gather(*list(bridge._tool_tasks))
            delegate.assert_awaited_once_with("明日の天気")
            output_msg, create_msg = ws.sent[-2], ws.sent[-1]
            assert output_msg["item"]["type"] == "function_call_output"
            assert output_msg["item"]["call_id"] == "call_1"
            assert json.loads(output_msg["item"]["output"]) == {"result": "明日は晴れです"}
            assert create_msg == {"type": "response.create"}
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_delegate_failure_is_reported_to_model(self):
        bridge, ws = await _started_bridge(AsyncMock(side_effect=RuntimeError("boom")))
        try:
            await bridge._run_tool({"name": "ask_hermes", "call_id": "c", "arguments": "{}"})
            assert "boom" in json.loads(ws.sent[-2]["item"]["output"])["result"]
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_barge_in_truncates_unplayed_audio(self):
        bridge, ws = await _started_bridge()
        try:
            await bridge._handle_event({
                "type": "response.output_audio.delta",
                "item_id": "item_1",
                "delta": base64.b64encode(b"\x00" * 960 * 3).decode(),
            })
            bridge.source.read()  # one 20 ms frame played
            await bridge._handle_event({"type": "input_audio_buffer.speech_started"})
            assert ws.sent[-1] == {
                "type": "conversation.item.truncate",
                "item_id": "item_1",
                "content_index": 0,
                "audio_end_ms": 20,
            }
            assert bridge.source.read() == b"\x00" * rv.FRAME_BYTES
        finally:
            await bridge.close()

    @pytest.mark.asyncio
    async def test_speech_started_without_pending_audio_sends_nothing(self):
        bridge, ws = await _started_bridge()
        try:
            sent_before = len(ws.sent)
            await bridge._handle_event({"type": "input_audio_buffer.speech_started"})
            assert len(ws.sent) == sent_before
        finally:
            await bridge.close()


# =====================================================================
# Adapter wiring
# =====================================================================

def _make_receiver():
    from plugins.platforms.discord.adapter import VoiceReceiver
    vc = MagicMock()
    vc.channel = None
    return VoiceReceiver(vc)


class TestDrainUserAudio:
    def test_returns_target_user_audio_and_drops_others(self):
        receiver = _make_receiver()
        receiver.map_ssrc(1, 42)
        receiver.map_ssrc(2, 99)
        receiver._buffers[1] = bytearray(b"\x01" * 10)
        receiver._buffers[2] = bytearray(b"\x02" * 10)
        assert receiver.drain_user_audio(42) == b"\x01" * 10
        assert len(receiver._buffers) == 0

    def test_zero_accepts_any_mapped_user(self):
        receiver = _make_receiver()
        receiver.map_ssrc(1, 42)
        receiver._buffers[1] = bytearray(b"\x01" * 10)
        receiver._buffers[3] = bytearray(b"\x03" * 10)  # unmapped, not inferable
        assert receiver.drain_user_audio(0) == b"\x01" * 10


class TestAdapterRealtimeGuards:
    def _make_adapter(self):
        from plugins.platforms.discord.adapter import DiscordAdapter
        adapter = object.__new__(DiscordAdapter)
        adapter._voice_realtime_bridges = {}
        adapter._voice_text_channels = {111: 123}
        adapter._voice_clients = {}
        return adapter

    def test_realtime_voice_active_for_chat(self):
        adapter = self._make_adapter()
        assert not adapter.realtime_voice_active_for_chat("123")
        adapter._voice_realtime_bridges[111] = object()
        assert adapter.realtime_voice_active_for_chat("123")
        assert not adapter.realtime_voice_active_for_chat("456")

    @pytest.mark.asyncio
    async def test_play_in_voice_channel_yields_to_realtime(self):
        adapter = self._make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        adapter._voice_clients[111] = vc
        adapter._voice_realtime_bridges[111] = object()
        assert await adapter.play_in_voice_channel(111, "/tmp/x.ogg") is False
        vc.play.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_falls_back_without_api_key(self, monkeypatch):
        monkeypatch.delenv("HERMES_REALTIME_OPENAI_API_KEY", raising=False)
        adapter = self._make_adapter()
        adapter._voice_delegate_callback = AsyncMock()
        assert await adapter._start_realtime_voice(111, MagicMock()) is False


# =====================================================================
# Runner delegation
# =====================================================================

def _make_runner(tmp_path):
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._VOICE_MODE_PATH = tmp_path / "gateway_voice_mode.json"
    runner._session_db = None
    runner.session_store = MagicMock()
    runner._is_user_authorized = lambda source: True
    return runner


def _discord_adapter_mock():
    adapter = AsyncMock()
    adapter._voice_text_channels = {111: 123}
    adapter._voice_sources = {}
    adapter._client = MagicMock()
    adapter._client.get_channel = MagicMock(return_value=AsyncMock())
    return adapter


class TestRealtimeDelegate:
    @pytest.mark.asyncio
    async def test_runs_agent_turn_and_returns_reply(self, tmp_path):
        from gateway.config import Platform
        runner = _make_runner(tmp_path)
        adapter = _discord_adapter_mock()
        runner.adapters[Platform.DISCORD] = adapter
        runner._handle_message = AsyncMock(return_value="予定は3件です")

        reply = await runner._handle_voice_realtime_delegate(111, 42, "今日の予定は？")

        assert reply == "予定は3件です"
        event = runner._handle_message.call_args[0][0]
        assert event.text.startswith("[Voice call]")
        assert event.text.endswith("今日の予定は？")
        assert event.message_type == MessageType.TEXT
        assert event.source.chat_id == "123"
        assert event.raw_message.guild_id == 111
        adapter.send.assert_awaited_once_with("123", "予定は3件です")

    @pytest.mark.asyncio
    async def test_unauthorized_user_is_refused(self, tmp_path):
        from gateway.config import Platform
        runner = _make_runner(tmp_path)
        runner._is_user_authorized = lambda source: False
        runner.adapters[Platform.DISCORD] = _discord_adapter_mock()
        runner._handle_message = AsyncMock()

        reply = await runner._handle_voice_realtime_delegate(111, 42, "hi")

        assert "not authorized" in reply
        runner._handle_message.assert_not_called()

    def test_voice_reply_suppressed_while_realtime_active(self, tmp_path):
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        from gateway.session import SessionSource
        runner = _make_runner(tmp_path)
        adapter = MagicMock()
        adapter.realtime_voice_active_for_chat = MagicMock(return_value=True)
        runner.adapters[Platform.DISCORD] = adapter
        runner._voice_mode[runner._voice_key(Platform.DISCORD, "123")] = "all"
        event = MessageEvent(
            text="hi",
            source=SessionSource(platform=Platform.DISCORD, chat_id="123", user_id="42"),
        )
        assert runner._should_send_voice_reply(event, "hello", []) is False


# =====================================================================
# Voice follow (discord.voice_follow)
# =====================================================================

class TestVoiceFollow:
    def _make_adapter(self, monkeypatch, cfg=None):
        from plugins.platforms.discord.adapter import DiscordAdapter
        adapter = object.__new__(DiscordAdapter)
        adapter._allowed_user_ids = {"42"}
        adapter._voice_clients = {}
        adapter._on_voice_disconnect = MagicMock()
        adapter._client = MagicMock()
        adapter._client.get_channel = MagicMock(return_value=SimpleNamespace(name="home"))
        adapter.build_source = MagicMock(return_value=SimpleNamespace(to_dict=lambda: {"user_id": "42"}))
        adapter.join_voice_channel = AsyncMock(return_value=True)
        adapter.leave_voice_channel = AsyncMock()
        merged = {"enabled": True, "user_id": "", "text_channel_id": ""}
        merged.update(cfg or {})
        adapter._load_voice_follow_config = lambda: merged
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "123")
        return adapter

    @staticmethod
    def _member(user_id=42, guild_id=111):
        return SimpleNamespace(id=user_id, display_name="me", guild=SimpleNamespace(id=guild_id))

    @staticmethod
    def _state(channel):
        return SimpleNamespace(channel=channel)

    def test_target_defaults_to_single_allowed_user_and_home_channel(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        assert adapter._voice_follow_target() == (42, 123)

    def test_target_none_when_disabled(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch, {"enabled": False})
        assert adapter._voice_follow_target() is None

    def test_target_none_when_user_ambiguous(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        adapter._allowed_user_ids = {"42", "43"}
        assert adapter._voice_follow_target() is None

    @pytest.mark.asyncio
    async def test_joins_when_followed_user_enters(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        vc_channel = SimpleNamespace(name="General")
        await adapter._follow_voice_state(self._member(), self._state(None), self._state(vc_channel))
        adapter.join_voice_channel.assert_awaited_once_with(
            vc_channel, text_channel_id=123, source={"user_id": "42"}
        )

    @pytest.mark.asyncio
    async def test_moves_when_followed_user_switches(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        a, b = SimpleNamespace(name="A"), SimpleNamespace(name="B")
        await adapter._follow_voice_state(self._member(), self._state(a), self._state(b))
        assert adapter.join_voice_channel.call_args[0][0] is b

    @pytest.mark.asyncio
    async def test_leaves_when_followed_user_leaves(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        adapter._voice_clients[111] = MagicMock()
        await adapter._follow_voice_state(self._member(), self._state(SimpleNamespace(name="A")), self._state(None))
        adapter.leave_voice_channel.assert_awaited_once_with(111)
        adapter._on_voice_disconnect.assert_called_once_with("123")

    @pytest.mark.asyncio
    async def test_ignores_other_users_and_mute_toggles(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        ch = SimpleNamespace(name="A")
        await adapter._follow_voice_state(self._member(user_id=99), self._state(None), self._state(ch))
        await adapter._follow_voice_state(self._member(), self._state(ch), self._state(ch))
        adapter.join_voice_channel.assert_not_called()
        adapter.leave_voice_channel.assert_not_called()


class TestReceiverKeyRefresh:
    def test_picks_up_secret_key_after_silent_reconnect(self):
        from plugins.platforms.discord.adapter import VoiceReceiver
        vc = MagicMock()
        vc._connection.secret_key = [1] * 32
        vc._connection.ssrc = 10
        receiver = VoiceReceiver(vc)
        receiver.start()
        assert receiver._secret_key == bytes([1] * 32)

        vc._connection.secret_key = [2] * 32  # discord.py reconnected
        vc._connection.ssrc = 11
        receiver._sync_transport_keys()

        assert receiver._secret_key == bytes([2] * 32)
        assert receiver._bot_ssrc == 11


class TestRealtimeVoiceContext:
    @pytest.mark.asyncio
    async def test_bridge_appends_context_to_instructions(self):
        ws = FakeWS()

        async def connect(url, api_key):
            return ws

        bridge = rv.RealtimeVoiceBridge(
            api_key="k", delegate=AsyncMock(), context="Your name is poi.", connect=connect
        )
        await bridge.start()
        try:
            instructions = ws.sent[0]["session"]["instructions"]
            assert instructions.startswith(rv.DEFAULT_INSTRUCTIONS)
            assert instructions.endswith("Your name is poi.")
        finally:
            await bridge.close()

    def test_adapter_context_has_soul_memory_and_location(self, monkeypatch):
        from plugins.platforms.discord.adapter import DiscordAdapter
        import agent.prompt_builder as pb
        import tools.memory_tool as mt

        monkeypatch.setattr(pb, "load_soul_md", lambda: "Your name is poi.")

        class FakeStore:
            def load_from_disk(self):
                pass

            def format_for_system_prompt(self, target):
                return {"user": "USER PROFILE: Shohei", "memory": "MEMORY: notes"}[target]

        monkeypatch.setattr(mt, "MemoryStore", FakeStore)
        adapter = object.__new__(DiscordAdapter)
        adapter._voice_text_channels = {111: 123}
        adapter._client = MagicMock()
        adapter._client.get_channel = MagicMock(return_value=SimpleNamespace(name="agent-log"))
        vc = SimpleNamespace(channel=SimpleNamespace(name="一般", guild=SimpleNamespace(name="Shohei81")))

        ctx = adapter._realtime_voice_context(111, vc)

        for expected in ("Your name is poi.", "USER PROFILE: Shohei", "MEMORY: notes",
                         "'一般'", "'Shohei81'", "#agent-log", "Current time:"):
            assert expected in ctx
