#!/usr/bin/env python3

import argparse
import asyncio
import logging
import time
import re
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Set, Dict, Union, Tuple
import aiohttp
from wyoming.asr import Transcript, Transcribe
from wyoming.audio import AudioChunk, AudioStop
from wyoming.client import AsyncClient
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "en"

# Absolute STT budget, starting when the Wyoming request begins.
# Configurable via the Home Assistant add-on's stt_timeout_seconds option.
DEFAULT_STT_TIMEOUT_SECONDS = 8.0
MAX_AUDIO_BYTES = 2_000_000
MAX_QUEUED_CHUNKS = 512
CLEANUP_TIMEOUT_SECONDS = 2.0


@dataclass
class STTEntity:
    entity_id: str
    supported_languages: Set[str]
    _supported_lang_map: Dict[Tuple[str, Union[str, None]], str] = field(
        default_factory=dict
    )
    _language_map: Dict[str, str] = field(default_factory=dict)

    def get_best_language(self, language: str) -> Optional[str]:
        language = language.strip()

        if language in self.supported_languages:
            return language

        best_language = self._language_map.get(language)
        if best_language is not None:
            return best_language
        if not self._supported_lang_map:
            # {(family, region): language}
            for supported_lang in self.supported_languages:
                supported_lang_parts = re.split(r"[-_]", supported_lang)
                supported_lang_family = supported_lang_parts[0].lower()
                supported_lang_region = (
                    supported_lang_parts[1].upper()
                    if len(supported_lang_parts) > 1
                    else None
                )
                self._supported_lang_map[
                    (supported_lang_family, supported_lang_region)
                ] = supported_lang

        language_parts = re.split(r"[-_]", language)
        lang_family = language_parts[0].lower()
        lang_region = language_parts[1].upper() if len(language_parts) > 1 else None

        # Exact match
        best_language = self._supported_lang_map.get((lang_family, lang_region))
        if best_language is None:
            # Special cases
            if (lang_family == "en") and (lang_region is None):
                best_language = self._supported_lang_map.get(("en", "US"))

        if best_language is None:
            # Family only
            best_language = self._supported_lang_map.get((lang_family, None))

        if best_language is not None:
            self._language_map[language] = best_language
            return best_language

        return None

@dataclass
class RaceRequest:
    """State belonging to one Wyoming transcription request."""

    request_id: str
    language: str
    http_session: aiohttp.ClientSession
    queues: Dict[str, asyncio.Queue] = field(default_factory=dict)
    tasks: Dict[str, asyncio.Task] = field(default_factory=dict)
    # Entries arrive in actual provider-completion order, not entity order.
    results: asyncio.Queue = field(default_factory=asyncio.Queue)
    watchdog: Optional[asyncio.Task] = None
    audio_stopped: bool = False
    audio_bytes: int = 0
    started: float = field(default_factory=time.monotonic)


async def main() -> None:
    """Runs fallback ASR server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--hass-token", required=True, help="Long-lived access token for Home Assistant"
    )
    parser.add_argument(
        "--hass-http-uri",
        default="http://homeassistant.local:8123/api",
        help="URI of Home Assistant HTTP API",
    )
    parser.add_argument(
        "stt_entity_id",
        nargs="+",
        help="Ids of speech-to-text entities in fallback order",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    parser.add_argument(
        "--stt-timeout-seconds", type=float, default=DEFAULT_STT_TIMEOUT_SECONDS,
        help="Absolute timeout for a single STT request in seconds",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    # Do not log args: they contain the Home Assistant access token.

    # Get STT entity info
    entities: List[STTEntity] = []
    headers = {"Authorization": f"Bearer {args.hass_token}"}
    async with aiohttp.ClientSession() as session:
        for entity_id in args.stt_entity_id:
            _LOGGER.debug("Getting info for STT entity: %s", entity_id)
            async with session.get(
                f"{args.hass_http_uri}/stt/{entity_id}", headers=headers
            ) as resp:
                if resp.status != 200:
                    _LOGGER.warning(
                        "Failed to get entity info: %s, status=%s",
                        entity_id,
                        resp.status,
                    )
                    continue

                info = await resp.json()
                # Check required audio format
                if (
                    (16000 not in info["sample_rates"])
                    or (16 not in info["bit_rates"])
                    or (1 not in info["channels"])
                    or ("wav" not in info["formats"])
                    or ("pcm" not in info["codecs"])
                ):
                    _LOGGER.warning(
                        "Skipping '%s': 16Khz 16-bit mono PCM is not supported",
                        entity_id,
                    )
                    _LOGGER.warning("%s: %s", entity_id, info)
                    continue
                entities.append(
                    STTEntity(
                        entity_id=entity_id,
                        supported_languages=set(info["languages"]),
                    )
                )

    _LOGGER.debug("Entities: %s", entities)

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")

    try:
        await server.run(
            partial(
                FallbackEventHandler, entities, args.hass_token, args.hass_http_uri,
                args.stt_timeout_seconds,
            )
        )
    except KeyboardInterrupt:
        pass
# -----------------------------------------------------------------------------


class FallbackEventHandler(AsyncEventHandler):
    """Handle one Wyoming client; dispatch each request to all STT providers."""

    def __init__(
        self,
        entities: List[STTEntity],
        hass_token: str,
        hass_http_uri: str,
        stt_timeout_seconds: float = DEFAULT_STT_TIMEOUT_SECONDS,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.entities = entities
        self.hass_token = hass_token
        self.hass_http_uri = hass_http_uri
        self.stt_timeout_seconds = stt_timeout_seconds
        self.client_id = str(time.monotonic_ns())
        self._language = DEFAULT_LANGUAGE
        self._request: Optional[RaceRequest] = None
        self._ignore_audio_until_stop = False
        self._info_event: Optional[Event] = None

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming events with an absolute STT deadline."""
        try:
            if Describe.is_type(event.type):
                await self._write_info()
                return True

            if Transcribe.is_type(event.type):
                transcribe = Transcribe.from_event(event)
                if self._request is not None:
                    _LOGGER.warning("Ignoring overlapping Transcribe on client %s", self.client_id)
                else:
                    self._ignore_audio_until_stop = False
                    self._language = transcribe.language or DEFAULT_LANGUAGE
                    self._start_request()
                return True

            if AudioChunk.is_type(event.type):
                if self._ignore_audio_until_stop:
                    return True
                if self._request is None:
                    self._start_request()
                req = self._request
                if req is None or req.audio_stopped:
                    return True

                chunk = AudioChunk.from_event(event).audio
                req.audio_bytes += len(chunk)
                if req.audio_bytes > MAX_AUDIO_BYTES:
                    _LOGGER.warning("[%s] Audio size limit exceeded", req.request_id)
                    self._ignore_audio_until_stop = True
                    await self._finish_request(req, "")
                    return True

                for entity_id, queue in req.queues.items():
                    task = req.tasks[entity_id]
                    if task.done():
                        continue
                    try:
                        queue.put_nowait(chunk)
                    except asyncio.QueueFull:
                        # A non-consuming provider must not exhaust RAM or block
                        # the Wyoming event handler. Exclude this provider only.
                        _LOGGER.warning(
                            "[%s] %s stopped consuming audio; dropping provider",
                            req.request_id,
                            entity_id,
                        )
                        task.cancel()
                return True

            if AudioStop.is_type(event.type):
                if self._ignore_audio_until_stop:
                    self._ignore_audio_until_stop = False
                    return True
                req = self._request
                if req is None:
                    await self._write_empty_transcript()
                    return True
                req.audio_stopped = True
                if req.watchdog is not None:
                    req.watchdog.cancel()
                _LOGGER.info(
                    "[%s] AudioStop received: %d bytes, %.1fs STT budget remaining",
                    req.request_id,
                    req.audio_bytes,
                    max(0.0, req.started + self.stt_timeout_seconds - time.monotonic()),
                )
                for entity_id, queue in req.queues.items():
                    if req.tasks[entity_id].done():
                        continue
                    try:
                        queue.put_nowait(None)
                    except asyncio.QueueFull:
                        _LOGGER.warning(
                            "[%s] %s audio queue full at AudioStop; dropping provider",
                            req.request_id,
                            entity_id,
                        )
                        req.tasks[entity_id].cancel()

                text = await self._first_success(req)
                await self._finish_request(req, text)
                return True

        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.exception("Error handling Wyoming event")
            req = self._request
            if req is not None:
                self._ignore_audio_until_stop = True
                await self._finish_request(req, "")
        return True

    def _start_request(self) -> None:
        """Launch one streaming HTTP request per language-compatible entity."""
        assert self._request is None
        req = RaceRequest(
            request_id=str(time.monotonic_ns()),
            language=self._language,
            http_session=aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=None, connect=5, sock_connect=5, sock_read=None
                )
            ),
        )
        self._request = req
        for entity in self.entities:
            language = entity.get_best_language(req.language)
            if language is None:
                _LOGGER.debug(
                    "[%s] %s doesn't support %s",
                    req.request_id,
                    entity.entity_id,
                    req.language,
                )
                continue
            queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUED_CHUNKS)
            req.queues[entity.entity_id] = queue
            req.tasks[entity.entity_id] = asyncio.create_task(
                self._run_provider(req, entity.entity_id, language, queue)
            )
            _LOGGER.info(
                "[%s] Started STT: %s (%s)", req.request_id, entity.entity_id, language
            )

        req.watchdog = asyncio.create_task(self._recording_watchdog(req))
        if not req.tasks:
            _LOGGER.warning("[%s] No STT providers support %s", req.request_id, req.language)

    async def _run_provider(
        self,
        req: RaceRequest,
        entity_id: str,
        language: str,
        queue: asyncio.Queue,
    ) -> None:
        """Stream the live audio to one provider and publish its final result."""
        headers = {
            "Authorization": f"Bearer {self.hass_token}",
            "Content-Type": "audio/wav",
            "X-Speech-Content": (
                f"language={language};format=wav;codec=pcm;"
                "bit_rate=16;sample_rate=16000;channel=1"
            ),
        }

        async def audio_stream():
            while True:
                chunk = await queue.get()
                if chunk is None:
                    return
                yield chunk

        text = ""
        try:
            async with req.http_session.post(
                f"{self.hass_http_uri}/stt/{entity_id}",
                headers=headers,
                data=audio_stream(),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    text = (result.get("text") or "").strip()
                else:
                    _LOGGER.warning(
                        "[%s] %s HTTP error %s",
                        req.request_id,
                        entity_id,
                        response.status,
                    )
        except asyncio.CancelledError:
            # A dropped provider counts as failed, not as an indefinite wait.
            req.results.put_nowait((entity_id, ""))
            raise
        except Exception:
            _LOGGER.exception("[%s] Error in %s", req.request_id, entity_id)
        # Only report completed results. Cancelled providers are discarded.
        _LOGGER.info(
            "[%s] %s finished: %s",
            req.request_id,
            entity_id,
            "transcript" if text else "empty/failed",
        )
        req.results.put_nowait((entity_id, text))

    async def _first_success(self, req: RaceRequest) -> str:
        """Wait for first nonempty result; never let a provider hang block us."""
        remaining = len(req.tasks)
        # AudioStop must not restart the STT deadline.
        deadline = req.started + self.stt_timeout_seconds
        while remaining:
            seconds_left = deadline - asyncio.get_running_loop().time()
            if seconds_left <= 0:
                break
            try:
                entity_id, text = await asyncio.wait_for(
                    req.results.get(), timeout=seconds_left
                )
            except asyncio.TimeoutError:
                break
            remaining -= 1
            if text:
                _LOGGER.info(
                    "[%s] STT winner: %s (%.2fs since request started)",
                    req.request_id,
                    entity_id,
                    time.monotonic() - req.started,
                )
                return text

        if remaining:
            _LOGGER.warning(
                "[%s] STT result deadline reached; %d provider(s) still pending",
                req.request_id,
                remaining,
            )
        else:
            _LOGGER.warning("[%s] All STT providers failed/returned empty", req.request_id)
        return ""

    async def _recording_watchdog(self, req: RaceRequest) -> None:
        try:
            await asyncio.sleep(self.stt_timeout_seconds)
            if self._request is req and not req.audio_stopped:
                _LOGGER.warning(
                    "[%s] No AudioStop after %.1fs: ending failed request",
                    req.request_id,
                    self.stt_timeout_seconds,
                )
                self._ignore_audio_until_stop = True
                await self._finish_request(req, "")
        except asyncio.CancelledError:
            pass

    async def _finish_request(self, req: RaceRequest, text: str) -> None:
        """Emit exactly one terminal Wyoming transcript and tear down providers."""
        if self._request is not req:
            return
        self._request = None
        self._language = DEFAULT_LANGUAGE
        if req.watchdog is not None and req.watchdog is not asyncio.current_task():
            req.watchdog.cancel()
        try:
            try:
                await asyncio.wait_for(
                    self.write_event(Transcript(text=text).event()),
                    timeout=CLEANUP_TIMEOUT_SECONDS,
                )
            except (asyncio.TimeoutError, OSError):
                _LOGGER.warning("[%s] Could not deliver Wyoming transcript", req.request_id)
        finally:
            for task in req.tasks.values():
                if not task.done():
                    task.cancel()
            if req.tasks:
                done, pending = await asyncio.wait(
                    req.tasks.values(), timeout=CLEANUP_TIMEOUT_SECONDS
                )
                if pending:
                    _LOGGER.warning(
                        "[%s] %d cancelled provider(s) did not stop promptly",
                        req.request_id,
                        len(pending),
                    )
            try:
                await asyncio.wait_for(
                    req.http_session.close(), timeout=CLEANUP_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                _LOGGER.warning("[%s] Timed out closing HTTP session", req.request_id)

    async def _write_empty_transcript(self) -> None:
        await self.write_event(Transcript(text="").event())

    async def _write_info(self) -> None:
        if self._info_event is not None:
            await self.write_event(self._info_event)
            return

        supported_langs: Set[str] = set()
        for entity in self.entities:
            supported_langs.update(entity.supported_languages)
        info = Info(
            asr=[
                AsrProgram(
                    name="stt-fallback",
                    attribution=Attribution(
                        name="The Home Assistant Authors",
                        url="http://github.com/OHF-voice",
                    ),
                    description="Automatic fallback for Home Assistant speech-to-text",
                    installed=True,
                    version="1.0.0",
                    models=[
                        AsrModel(
                            name="fallback",
                            attribution=Attribution(
                                name="The Home Assistant Authors",
                                url="http://github.com/OHF-voice",
                            ),
                            installed=True,
                            description="Fallback model",
                            version=None,
                            languages=list(sorted(supported_langs)),
                        )
                    ],
                )
            ]
        )
        self._info_event = info.event()
        await self.write_event(self._info_event)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
