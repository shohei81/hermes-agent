"""OpenAI Realtime voice bridge for Discord voice channels.

Two-layer voice mode: a Realtime speech-to-speech model holds the live
conversation (streaming audio, server-side turn detection, barge-in), and
delegates anything that needs real work to the full Hermes agent through a
single ``ask_hermes`` function call.  The agent's reply is returned as the
function output and the Realtime model speaks it.

Audio plumbing:

* Discord receive: 48 kHz stereo s16le  ->  Realtime input: 24 kHz mono s16le
* Realtime output: 24 kHz mono s16le    ->  Discord playback: 48 kHz stereo

The API key is read from ``HERMES_REALTIME_OPENAI_API_KEY`` rather than
``OPENAI_API_KEY`` so enabling this mode never silently switches other
voice tools (STT/TTS) onto paid OpenAI endpoints.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from typing import Any, Awaitable, Callable, Optional

import numpy as np

try:
    import discord
except ImportError:  # pragma: no cover - discord is a hard dep of the adapter
    discord = None

logger = logging.getLogger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime"
API_KEY_ENV = "HERMES_REALTIME_OPENAI_API_KEY"
REALTIME_RATE = 24000
FRAME_BYTES = 3840  # 20 ms of 48 kHz stereo s16le (discord.opus.Encoder.FRAME_SIZE)
BYTES_PER_MS = 192  # 48 kHz * 2 channels * 2 bytes / 1000
MAX_TOOL_OUTPUT_CHARS = 4000

DEFAULT_INSTRUCTIONS = (
    "You are the live voice of the user's personal agent, talking with them in "
    "a Discord voice channel. You ARE that agent: use the name, personality "
    "and knowledge described below, and never say you are a separate voice "
    "assistant. Reply in the user's language, briefly and naturally, like a "
    "phone call. Handle small talk yourself. For anything that needs tools, "
    "lookups, files, the web, calendars, other Discord channels, memory "
    "updates or careful reasoning, call ask_hermes with a self-contained task "
    "description: it runs your full agent with all of its tools and "
    "permissions, so never claim you lack access before trying it. Before "
    "calling it, say a short filler such as 'ちょっと調べるね'. When the "
    "result comes back, summarise it conversationally instead of reading it "
    "verbatim."
)

ASK_HERMES_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "ask_hermes",
    "description": (
        "Delegate a task to the full Hermes agent, which has tools, memory "
        "and the user's ongoing text conversation. Returns its reply."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Self-contained request for Hermes, including any needed context.",
            }
        },
        "required": ["task"],
    },
}


def discord_to_realtime(pcm: bytes) -> bytes:
    """48 kHz stereo s16le -> 24 kHz mono s16le (channel average + 2x decimation)."""
    s = np.frombuffer(pcm[: len(pcm) // 8 * 8], dtype="<i2").astype(np.int32)
    mono = (s[0::2] + s[1::2]) // 2
    return ((mono[0::2] + mono[1::2]) // 2).astype("<i2").tobytes()


def realtime_to_discord(pcm: bytes) -> bytes:
    """24 kHz mono s16le -> 48 kHz stereo s16le (sample repeat)."""
    m = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
    return np.repeat(m, 4).astype("<i2").tobytes()


_AudioSourceBase = discord.AudioSource if discord is not None else object


class RealtimeAudioSource(_AudioSourceBase):
    """Continuous Discord audio source fed by Realtime output deltas.

    ``read`` is polled every 20 ms by discord.py's sender thread and returns
    silence when nothing is queued, so playback never stops between replies.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._lock = threading.Lock()
        self.played_bytes = 0  # bytes of the current response item already sent

    def is_opus(self) -> bool:
        return False

    def feed(self, pcm: bytes) -> None:
        with self._lock:
            self._buf.extend(pcm)

    def clear(self) -> int:
        """Drop queued audio; return how many bytes were still unplayed."""
        with self._lock:
            pending = len(self._buf)
            self._buf.clear()
        return pending

    def reset_played(self) -> None:
        with self._lock:
            self.played_bytes = 0

    def read(self) -> bytes:
        with self._lock:
            if not self._buf:
                return b"\x00" * FRAME_BYTES
            frame = bytes(self._buf[:FRAME_BYTES])
            del self._buf[:FRAME_BYTES]
            self.played_bytes += len(frame)
        return frame.ljust(FRAME_BYTES, b"\x00")


async def _default_connect(url: str, api_key: str):
    from websockets.asyncio.client import connect

    return await connect(
        url,
        additional_headers={"Authorization": f"Bearer {api_key}"},
        max_size=None,
    )


class RealtimeVoiceBridge:
    """One Realtime WebSocket session bound to one Discord voice connection."""

    def __init__(
        self,
        *,
        api_key: str,
        delegate: Callable[[str], Awaitable[str]],
        model: str = "gpt-realtime-2.1-mini",
        voice: str = "marin",
        instructions: str = "",
        context: str = "",
        connect: Callable[[str, str], Awaitable[Any]] = _default_connect,
    ) -> None:
        self._api_key = api_key
        self._delegate = delegate
        self._model = model
        self._voice = voice
        self._instructions = "\n\n".join(
            part for part in (instructions or DEFAULT_INSTRUCTIONS, context) if part
        )
        self._connect = connect
        self._ws: Any = None
        self._recv_task: Optional[asyncio.Task] = None
        self._tool_tasks: set[asyncio.Task] = set()
        self._current_item: Optional[str] = None
        self.source = RealtimeAudioSource()

    @property
    def running(self) -> bool:
        return self._recv_task is not None and not self._recv_task.done()

    async def start(self) -> None:
        self._ws = await self._connect(f"{REALTIME_URL}?model={self._model}", self._api_key)
        await self._send({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self._model,
                "output_modalities": ["audio"],
                "instructions": self._instructions,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                        "turn_detection": {"type": "semantic_vad"},
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                        "voice": self._voice,
                    },
                },
                "tools": [ASK_HERMES_TOOL],
                "tool_choice": "auto",
            },
        })
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("Realtime voice session started (model=%s)", self._model)

    async def close(self) -> None:
        for task in [self._recv_task, *self._tool_tasks]:
            if task:
                task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self.source.clear()

    async def send_audio(self, pcm48: bytes) -> None:
        """Append Discord PCM (48 kHz stereo) to the Realtime input buffer."""
        if not pcm48 or self._ws is None:
            return
        await self._send({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(discord_to_realtime(pcm48)).decode("ascii"),
        })

    async def _send(self, payload: dict) -> None:
        await self._ws.send(json.dumps(payload))

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                await self._handle_event(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Realtime voice session ended: %s", e)

    async def _handle_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "response.output_audio.delta":
            item_id = event.get("item_id")
            if item_id != self._current_item:
                self._current_item = item_id
                self.source.reset_played()
            self.source.feed(realtime_to_discord(base64.b64decode(event.get("delta", ""))))
        elif etype == "input_audio_buffer.speech_started":
            await self._interrupt()
        elif etype == "response.done":
            for item in (event.get("response") or {}).get("output") or []:
                if item.get("type") == "function_call":
                    task = asyncio.create_task(self._run_tool(item))
                    self._tool_tasks.add(task)
                    task.add_done_callback(self._tool_tasks.discard)
        elif etype == "error":
            logger.warning("Realtime voice error: %s", event.get("error"))

    async def _interrupt(self) -> None:
        """User barged in: stop local playback and trim what the model 'said'."""
        unplayed = self.source.clear()
        if self._current_item and unplayed:
            await self._send({
                "type": "conversation.item.truncate",
                "item_id": self._current_item,
                "content_index": 0,
                "audio_end_ms": self.source.played_bytes // BYTES_PER_MS,
            })
        self._current_item = None

    async def _run_tool(self, item: dict) -> None:
        name = item.get("name")
        try:
            args = json.loads(item.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        if name != "ask_hermes":
            result = f"Unknown tool: {name}"
        else:
            try:
                result = await self._delegate(str(args.get("task", "")))
            except Exception as e:
                logger.warning("ask_hermes delegation failed: %s", e, exc_info=True)
                result = f"Hermes failed: {e}"
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": item.get("call_id"),
                "output": json.dumps({"result": (result or "")[:MAX_TOOL_OUTPUT_CHARS]}, ensure_ascii=False),
            },
        })
        await self._send({"type": "response.create"})
