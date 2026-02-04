"""Real-time transcription pipeline: orchestrates audio, VAD, ASR, and UI.

RealtimeTranscriber coordinates the async pipeline. All domain logic
is delegated to specialized modules (RingBuffer, VoiceActivityDetector,
transcribe, analyze_intent, ui render functions).
"""

import asyncio
import io
import re
import shutil
import signal
import sys
import time
from typing import Any

import numpy as np
import sounddevice as sd
from rich.console import Console
from rich.live import Live

from dictate.analysis import IntentResult, analyze_intent
from dictate.audio.ring_buffer import RingBuffer
from dictate.audio.vad import VadConfig, VoiceActivityDetector
from dictate.constants import (
    DEFAULT_ASR_MODEL,
    DEFAULT_AUDIO_QUEUE_MAXSIZE,
    DEFAULT_LANGUAGE,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_BUFFER_SECONDS,
    DEFAULT_MIN_WORDS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TRANSCRIBE_INTERVAL,
    DEFAULT_VAD_FRAME_MS,
    DEFAULT_VAD_MODE,
    DEFAULT_VAD_SILENCE_MS,
)
from dictate.env import LOGGER, suppress_output
from dictate.model import load_qwen3_asr
from dictate.protocols import FeatureExtractorLike, TokenizerLike
from dictate.transcribe import transcribe
from dictate.ui import UiState, render_layout


def is_meaningful(text: str) -> bool:
    """Filter out noise so we do not finalize junk output."""
    cleaned = re.sub(r"[^\w]", "", text)
    return len(cleaned) >= 2


class RealtimeTranscriber:
    """Async pipeline orchestrator for real-time speech transcription."""

    def __init__(
        self,
        model_path: str = DEFAULT_ASR_MODEL,
        language: str = DEFAULT_LANGUAGE,
        transcribe_interval: float = DEFAULT_TRANSCRIBE_INTERVAL,
        vad_frame_ms: int = DEFAULT_VAD_FRAME_MS,
        vad_mode: int = DEFAULT_VAD_MODE,
        vad_silence_ms: int = DEFAULT_VAD_SILENCE_MS,
        min_words: int = DEFAULT_MIN_WORDS,
        analyze: bool = False,
        llm_model: str | None = None,
        device: int | None = None,
        no_ui: bool = False,
        context: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.language = language
        self.transcribe_interval = transcribe_interval
        self.context = context
        self.min_words = min_words
        self.analyze = analyze
        self.llm_model_name = llm_model or DEFAULT_LLM_MODEL
        self.device = device
        self.no_ui = no_ui
        self.sample_rate = DEFAULT_SAMPLE_RATE

        self.audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(
            maxsize=DEFAULT_AUDIO_QUEUE_MAXSIZE
        )
        self.loop: asyncio.AbstractEventLoop | None = None

        # Models (loaded in run())
        self.model: Any = None
        self.tokenizer: TokenizerLike | None = None
        self.feature_extractor: FeatureExtractorLike | None = None
        self.llm: Any = None
        self.llm_tokenizer: TokenizerLike | None = None

        # Audio components
        self.ring_buffer = RingBuffer.create(
            DEFAULT_MAX_BUFFER_SECONDS, self.sample_rate
        )
        self.vad = VoiceActivityDetector(
            VadConfig(
                frame_ms=vad_frame_ms,
                mode=vad_mode,
                silence_ms=vad_silence_ms,
                sample_rate=self.sample_rate,
            )
        )
        self.last_transcribed_sample = 0

        # Concurrency
        self.buffer_lock = asyncio.Lock()
        self.gpu_lock = asyncio.Lock()

        # Transcript state
        self.current_transcript = ""
        self.last_transcript = ""
        self.pending_analysis: tuple[str, IntentResult | None] | None = None

        # UI
        self.ui_state = UiState(
            language=language,
            vad_mode=vad_mode,
            vad_frame_ms=vad_frame_ms,
            model_path=model_path,
            llm_model_name=self.llm_model_name,
            analyze_enabled=analyze,
        )
        self.console_out = Console()
        self.console_ui = Console(stderr=True, force_terminal=True)
        self.live: Live | None = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Keep callback lightweight by deferring work to the async loop."""
        data = indata.reshape(-1).copy()
        self.loop.call_soon_threadsafe(
            lambda: (
                self.audio_queue.put_nowait(data)
                if not self.audio_queue.full()
                else None
            )
        )

    def _update_ui(self, force: bool = False) -> None:
        if not self.live:
            return
        self.live.update(render_layout(self.ui_state), refresh=force)

    def _log_info(self, message: str, *args: Any) -> None:
        """Log only when the live UI is disabled to avoid terminal clutter."""
        if not self.live:
            LOGGER.info(message, *args)

    def _run_transcribe(self, audio: np.ndarray) -> str:
        """Run ASR on audio, return transcript text."""
        if len(audio) < self.sample_rate * 0.3:
            return ""
        parts: list[str] = []
        with suppress_output():
            for token in transcribe(
                self.model,
                self.tokenizer,
                self.feature_extractor,
                audio,
                self.language,
                context=self.context,
            ):
                parts.append(token)
        return "".join(parts).strip()

    async def _handle_turn_complete(self) -> None:
        """Emit a final turn result once VAD indicates completion."""
        if (
            not self.current_transcript
            or self.current_transcript == self.last_transcript
        ):
            return
        if not is_meaningful(self.current_transcript):
            return
        if len(self.current_transcript.split()) < self.min_words:
            return

        self.ui_state.turn_complete = True
        final_transcript = self.current_transcript

        analysis_result: IntentResult | None = None
        if self.analyze:
            self.ui_state.status = "Analyzing"
            self._update_ui()
            async with self.gpu_lock:
                start = time.perf_counter()
                analysis_result = await asyncio.to_thread(
                    analyze_intent,
                    final_transcript,
                    self.llm,
                    self.llm_tokenizer,
                )
                self.ui_state.analysis_ms = (
                    time.perf_counter() - start
                ) * 1000

        self.pending_analysis = (final_transcript, analysis_result)
        self.last_transcript = final_transcript
        self.current_transcript = ""
        self.ui_state.status = "Listening"

        async with self.buffer_lock:
            self.ring_buffer.reset()
            self.vad.reset()

        self.last_transcribed_sample = self.ring_buffer.total_samples_written
        self.ui_state.buffer_seconds = 0.0
        self.ui_state.turn_complete = False

    async def _processor(self) -> None:
        """Coordinate capture/VAD/ASR in one loop to avoid races."""
        min_new_samples = int(self.sample_rate * 0.2)
        last_transcribe_time = self.loop.time()

        while True:
            self.ui_state.queue_size = self.audio_queue.qsize()
            frame: np.ndarray | None = None
            try:
                frame = await asyncio.wait_for(
                    self.audio_queue.get(), timeout=0.05
                )
            except asyncio.TimeoutError:
                pass

            if frame is not None:
                async with self.buffer_lock:
                    self.ring_buffer.append(frame)
                    turn_complete = self.vad.process(frame)
                    self.ui_state.vad_state = self.vad.state
                    self.ui_state.buffer_seconds = (
                        self.ring_buffer.filled_seconds
                    )

                if turn_complete:
                    await self._handle_turn_complete()

            now = self.loop.time()
            if now - last_transcribe_time >= self.transcribe_interval:
                samples_since = (
                    self.ring_buffer.total_samples_written
                    - self.last_transcribed_sample
                )
                if samples_since >= min_new_samples:
                    async with self.buffer_lock:
                        audio_int16 = self.ring_buffer.get_recent(
                            DEFAULT_MAX_BUFFER_SECONDS
                        )

                    if audio_int16.size >= int(self.sample_rate * 0.3):
                        if not self.gpu_lock.locked():
                            async with self.gpu_lock:
                                self.ui_state.status = "Transcribing"
                                self._update_ui()
                                audio = (
                                    audio_int16.astype(np.float32) / 32768.0
                                )
                                start = time.perf_counter()
                                text = await asyncio.to_thread(
                                    self._run_transcribe, audio
                                )
                                self.ui_state.asr_ms = (
                                    time.perf_counter() - start
                                ) * 1000
                            if text and text != self.current_transcript:
                                self.current_transcript = text
                            self.ui_state.status = "Listening"
                        self.last_transcribed_sample = (
                            self.ring_buffer.total_samples_written
                        )

                last_transcribe_time = now

    async def _display(self) -> None:
        """Keep UI responsive without blocking the ASR pipeline."""
        last_displayed = ""
        is_tty = sys.stdout.isatty()

        while True:
            await asyncio.sleep(0.1)

            if self.pending_analysis:
                final_transcript, analysis = self.pending_analysis
                self.pending_analysis = None

                self.ui_state.history.append((final_transcript, analysis))
                self.ui_state.partial_transcript = ""
                self._update_ui(force=True)

                if not self.live:
                    if is_tty:
                        sys.stdout.write("\r\033[K")
                        self.console_out.print(
                            f"[bold green]>[/bold green] {final_transcript}"
                        )
                        if analysis:
                            if analysis.intent or analysis.action:
                                self.console_out.print(
                                    f"  [cyan]Intent:[/cyan] {analysis.intent}"
                                )
                                if (
                                    analysis.entities
                                    and analysis.entities.lower() != "none"
                                ):
                                    self.console_out.print(
                                        f"  [cyan]Entities:[/cyan] {analysis.entities}"
                                    )
                                if analysis.action:
                                    self.console_out.print(
                                        f"  [cyan]Action:[/cyan] {analysis.action}"
                                    )
                            self.console_out.print()
                    else:
                        sys.stdout.write(f"{final_transcript}\n")
                        sys.stdout.flush()

                last_displayed = ""
                continue

            if self.current_transcript and is_meaningful(
                self.current_transcript
            ):
                if (
                    self.current_transcript != last_displayed
                    and not self.ui_state.turn_complete
                ):
                    self.ui_state.partial_transcript = (
                        self.current_transcript
                    )
                    if not self.live and is_tty:
                        width = (
                            shutil.get_terminal_size(
                                fallback=(80, 20)
                            ).columns
                            - 5
                        )
                        width = max(width, 10)
                        display_text = self.current_transcript
                        if len(display_text) > width:
                            display_text = (
                                "..." + display_text[-(width - 3) :]
                            )
                        sys.stdout.write(
                            f"\r\033[K  \033[2m{display_text}\033[0m"
                        )
                        sys.stdout.flush()
                    last_displayed = self.current_transcript

            if self.live:
                self._update_ui()

    async def run(self) -> None:
        """Wire models, stream, and tasks, then manage their lifecycle."""
        self.loop = asyncio.get_running_loop()
        if not self.no_ui:
            self.live = Live(
                render_layout(self.ui_state),
                console=self.console_ui,
                refresh_per_second=10,
                transient=False,
            )
            self.live.start()

        self.ui_state.status = "Loading ASR model..."
        self._update_ui(force=True)
        self._log_info("Loading ASR model...")
        self.model, self.tokenizer, self.feature_extractor = (
            await asyncio.to_thread(load_qwen3_asr, self.model_path)
        )

        # Prime caches and trigger one-time warnings off-screen.
        def warmup() -> None:
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                dummy_audio = (
                    np.random.randn(self.sample_rate).astype(np.float32)
                    * 0.01
                )
                list(
                    transcribe(
                        self.model,
                        self.tokenizer,
                        self.feature_extractor,
                        dummy_audio,
                        self.language,
                        context=self.context,
                    )
                )
            finally:
                sys.stderr = old_stderr

        await asyncio.to_thread(warmup)

        if self.analyze:
            self.ui_state.status = "Loading LLM..."
            self._update_ui(force=True)
            self._log_info("Loading LLM...")
            from mlx_lm.utils import load as load_llm

            self.llm, self.llm_tokenizer = await asyncio.to_thread(
                load_llm, self.llm_model_name
            )

        stream_kwargs: dict[str, Any] = {}
        if self.device is not None:
            stream_kwargs["device"] = self.device
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.vad.frame_samples,
            channels=1,
            dtype="int16",
            callback=self._audio_callback,
            **stream_kwargs,
        )

        info = (
            f"Language: {self.language} | "
            f"VAD: mode {self.vad._config.mode}, {self.vad._config.frame_ms}ms"
        )
        if self.analyze:
            info += " | Analysis: enabled"
        self._log_info("Ready - %s", info)
        self._log_info("Listening... (Ctrl+C to stop)")
        self.ui_state.status = "Listening"
        self._update_ui(force=True)

        stream.start()

        tasks = [
            asyncio.create_task(self._processor()),
            asyncio.create_task(self._display()),
        ]

        stop_event = asyncio.Event()

        def signal_handler() -> None:
            if not stop_event.is_set():
                self._log_info("Stopping...")
                stop_event.set()

        signal_handler_installed = False
        try:
            self.loop.add_signal_handler(signal.SIGINT, signal_handler)
            signal_handler_installed = True
        except NotImplementedError:
            signal_handler_installed = False

        try:
            await stop_event.wait()
        finally:
            if signal_handler_installed:
                try:
                    self.loop.remove_signal_handler(signal.SIGINT)
                except Exception:
                    pass

            for t in tasks:
                t.cancel()

            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                pass

            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

            if self.live:
                self.live.stop()
