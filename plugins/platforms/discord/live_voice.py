"""OpenAI GPT-Live voice bridge for Discord voice channels.

Full-duplex variant of :mod:`realtime_voice`.  GPT-Live listens and speaks at
the same time and decides turn-taking itself (backchannels, yielding when
interrupted), so there is no turn detection, truncation or response control
here.  Anything that needs real work goes to the full Hermes agent through
*client delegation*:

1. GPT-Live emits ``session.delegation.created`` (an id only, no task text).
2. The bridge builds the task from the transcript it has been accumulating
   and runs Hermes in the background, so the conversation keeps going.
3. The result goes back as ``session.commentary.append`` for that
   delegation id; GPT-Live paraphrases it when it fits the conversation.

Audio is the same 24 kHz mono PCM16 as Realtime, so playback and resampling
are shared with :mod:`realtime_voice`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Awaitable, Callable, List, Optional

try:
    from realtime_voice import (
        REALTIME_RATE,
        RealtimeAudioSource,
        discord_to_realtime,
        realtime_to_discord,
    )
except ImportError:
    from .realtime_voice import (
        REALTIME_RATE,
        RealtimeAudioSource,
        discord_to_realtime,
        realtime_to_discord,
    )

logger = logging.getLogger(__name__)

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
MAX_COMMENTARY_CHARS = 600  # appends are capped at 500 tokens; details live in the text channel
TRANSCRIPT_TURNS = 8  # recent turns handed to Hermes with each delegation
DELEGATION_SETTLE_S = 1.0  # delegation can arrive before the user's sentence is transcribed

DEFAULT_INSTRUCTIONS = """\
あなたはユーザー専属のエージェントの「声」として、Discord のボイスチャンネルでユーザーと通話している。あなたはそのエージェント本人であり、別の音声アシスタントとして振る舞わない。名前・人格・記憶は下に書かれたものを使う。親しい友人で、かつ頼れるアシスタントと電話しているような、自然な会話をする。
ユーザーの言葉づかいに合わせ、くだけた日本語にはくだけた日本語で返す。書き言葉ではなく話し言葉で話す。

# Backchannel policy
相づちは適度に入れる（うん、そうそう、なるほど、あー など）。相手の話を遮らない程度に、聞いていることが伝わるように。同じ相づちを続けて使わない。

# Interruption policy
ユーザーが話し始めたら話すのをやめて聞く。短い相づちは割り込みではないので、そのまま話し続けてよい。

# Delegation policy
## Backend tools
裏にいるエージェント本体（Hermes）は全ツールと権限を持っている。この Discord サーバーの全チャンネルの読み書き、ユーザーのファイルやメモ、スクリプトやコマンドの実行、Web 検索、予定やリマインダー、記憶の更新ができる。「アクセスできない」とは言わず、必要なら任せる。
## Delegate to the backend when
- チャンネル、メモ、ファイル、予定、ニュース、Web など、記憶だけでは答えられないことを聞かれたとき
- 何かを実行・投稿・保存・設定してほしいと頼まれたとき
- 進行中の依頼に訂正や追加があったとき
- 読み取りだけの調べものは確認を取らずにすぐ任せる
## Do not delegate to the backend when
- 挨拶や雑談、すでに伝えた結果の言い直し
- 依頼の中身が曖昧で、先に一言聞き返したほうがいいとき
任せるときは「ちょっと見てみるね」のような短い一言だけ言い、結果が出るまで内容を推測で答えない。待っている間は普通に会話を続けてよい。結果が届いたら要点を 1〜2 文で自分の言葉で伝える。詳細はテキストチャンネルに投稿されているので、長い内容は読み上げずにそう伝える。

# Length
普段の返事は 1〜2 文。一覧、見出し、URL、ID、コードは読み上げない。長い話はやり取りに分ける。

# Unclear audio
聞き取れないときや雑音だけのときは推測せず、短く聞き返す。背景の雑音には反応しない。
"""


async def _default_connect(url: str, api_key: str):
    from websockets.asyncio.client import connect

    return await connect(
        url,
        additional_headers={"Authorization": f"Bearer {api_key}"},
        max_size=None,
    )


class LiveVoiceBridge:
    """One GPT-Live WebSocket session bound to one Discord voice connection.

    Same surface as ``RealtimeVoiceBridge`` (start/close/send_audio/source/
    running) so the adapter can use either.
    """

    def __init__(
        self,
        *,
        api_key: str,
        delegate: Callable[[str], Awaitable[str]],
        model: str = "gpt-live-1",
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
            part for part in (DEFAULT_INSTRUCTIONS, instructions, context) if part
        )
        self._connect = connect
        self._ws: Any = None
        self._recv_task: Optional[asyncio.Task] = None
        self._tool_tasks: set[asyncio.Task] = set()
        self._transcript: list[list[str]] = []  # [role, text] turns, oldest first
        self.source = RealtimeAudioSource()

    @property
    def running(self) -> bool:
        return self._recv_task is not None and not self._recv_task.done()

    async def start(self) -> None:
        self._ws = await self._connect(LIVE_URL, self._api_key)
        await self._send({
            "type": "session.start",
            "session": {
                "model": self._model,
                "instructions": self._instructions,
                "audio": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_RATE},
                    "output": {"voice": self._voice},
                },
                "delegation": {"type": "client"},
            },
        })
        # Audio and commands are only accepted after session.started.
        while True:
            event = json.loads(await self._ws.recv())
            if event.get("type") == "session.started":
                break
            if event.get("type") == "error":
                raise RuntimeError(f"GPT-Live session.start failed: {event.get('error')}")
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("GPT-Live voice session started (model=%s)", self._model)

    async def close(self) -> None:
        for task in [self._recv_task, *self._tool_tasks]:
            if task:
                task.cancel()
        if self._ws is not None:
            try:
                await self._send({"type": "session.close"})
                await self._ws.close()
            except Exception:
                pass
        self.source.clear()

    async def send_audio(self, pcm48: bytes) -> None:
        """Append Discord PCM (48 kHz stereo) to the Live input stream."""
        if not pcm48 or self._ws is None:
            return
        await self._send({
            "type": "session.input_audio.append",
            "audio": base64.b64encode(discord_to_realtime(pcm48)).decode("ascii"),
        })

    async def _send(self, payload: dict) -> None:
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                await self._handle_event(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("GPT-Live voice session ended: %s", e)

    async def _handle_event(self, event: dict) -> None:
        etype = event.get("type")
        if etype == "session.output_audio.delta":
            self.source.feed(realtime_to_discord(base64.b64decode(event.get("delta", ""))))
        elif etype == "session.input_transcript.delta":
            self._add_transcript("user", event.get("delta", ""))
        elif etype == "session.output_transcript.delta":
            self._add_transcript("assistant", event.get("delta", ""))
        elif etype == "session.delegation.created":
            delegation_id = (event.get("delegation") or {}).get("id")
            if delegation_id:
                task = asyncio.create_task(self._run_delegation(delegation_id))
                self._tool_tasks.add(task)
                task.add_done_callback(self._tool_tasks.discard)
        elif etype == "session.closed":
            logger.info("GPT-Live voice session closed: %s", event.get("reason"))
        elif etype == "error":
            logger.warning("GPT-Live voice error: %s", event.get("error"))

    def _add_transcript(self, role: str, delta: str) -> None:
        if not delta:
            return
        if self._transcript and self._transcript[-1][0] == role:
            self._transcript[-1][1] += delta
        else:
            self._transcript.append([role, delta])
            del self._transcript[:-TRANSCRIPT_TURNS * 4]

    def _task_from_transcript(self) -> str:
        names = {"user": "User", "assistant": "You (voice)"}
        lines = [
            f"{names[role]}: {text.strip()}"
            for role, text in self._transcript[-TRANSCRIPT_TURNS:]
            if text.strip()
        ]
        return (
            "Your voice layer handed you the user's latest request. Recent voice "
            "transcript (speech recognition, may contain mistakes):\n" + "\n".join(lines)
        )

    async def _run_delegation(self, delegation_id: str) -> None:
        await asyncio.sleep(DELEGATION_SETTLE_S)
        try:
            result = await self._delegate(self._task_from_transcript())
        except Exception as e:
            logger.warning("GPT-Live delegation failed: %s", e, exc_info=True)
            result = f"Hermes failed: {e}"
        await self._send({
            "type": "session.commentary.append",
            "delegation_id": delegation_id,
            "content": (result or "Done, nothing to report.")[:MAX_COMMENTARY_CHARS],
        })
