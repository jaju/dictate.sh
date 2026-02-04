"""Textual TUI for notes mode: speak, accumulate, commit through LLM rewrite.

Left panel shows debounced, vocab-corrected speech. Press Enter to commit
accumulated text through the rewrite LLM. Right panel shows rewritten output.
Space to start/stop recording (push-to-talk). q quits and saves.

This module directly owns audio capture, VAD, and ASR — it does NOT use
RealtimeTranscriber. The ASR model is pre-loaded by notes.run_notes_pipeline()
and passed in via constructor to avoid subprocess/fd conflicts with Textual.
"""

import asyncio
import contextlib
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
from rich.rule import Rule
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, RichLog, Static

from dictate.audio.ring_buffer import RingBuffer
from dictate.audio.vad import VadConfig, VoiceActivityDetector
from dictate.constants import (
    DEFAULT_AUDIO_QUEUE_MAXSIZE,
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_MAX_BUFFER_SECONDS,
    DEFAULT_SAMPLE_RATE,
)
from dictate.notes import NotesConfig, append_turn, write_session_header
from dictate.protocols import FeatureExtractorLike, TokenizerLike
from dictate.rewrite import RewriteResult, apply_vocab, rewrite_transcript
from dictate.transcribe import is_meaningful, transcribe


class DiscardConfirmScreen(ModalScreen[bool]):
    """Modal confirmation for discarding accumulated text."""

    CSS = """
    DiscardConfirmScreen {
        align: center middle;
    }
    #discard-dialog {
        width: 50;
        height: 5;
        border: thick $error;
        background: $surface;
        padding: 1 2;
        content-align: center middle;
    }
    """

    BINDINGS = [
        Binding("y", "confirm", "Yes", priority=True),
        Binding("n", "cancel", "No", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

    def compose(self) -> ComposeResult:
        yield Static(
            "Discard accumulated text? (y/n)", id="discard-dialog"
        )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class DictateNotesApp(App):
    """Textual TUI for voice-driven notes with manual commit."""

    TITLE = "Dictate Notes"

    CSS = """
    #middle {
        height: 1fr;
    }
    #speech-panel {
        width: 40%;
        border: solid $primary;
        padding: 1 2;
        overflow-y: auto;
    }
    #output-panel {
        width: 60%;
        border: solid $success;
        padding: 1 2;
    }
    #status-bar {
        height: 1;
        dock: bottom;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    """

    BINDINGS = [
        Binding("enter", "commit", "Commit", priority=True),
        Binding("space", "toggle_recording", "Record/Stop", priority=True),
        Binding("escape", "discard", "Discard", priority=True),
        Binding("q", "quit_app", "Quit", priority=True),
    ]

    def __init__(
        self,
        model: Any,
        tokenizer: TokenizerLike,
        feature_extractor: FeatureExtractorLike,
        language: str,
        context: str | None,
        transcribe_interval: float,
        vad_frame_ms: int,
        vad_mode: int,
        vad_silence_ms: int,
        min_words: int,
        device: int | None,
        notes_config: NotesConfig,
        energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
        logit_bias: dict[int, float] | None = None,
    ) -> None:
        super().__init__()
        # Pre-loaded ASR model components
        self._model = model
        self._tokenizer = tokenizer
        self._feature_extractor = feature_extractor
        self._language = language
        self._asr_context = context
        self._logit_bias = logit_bias
        self._transcribe_interval = transcribe_interval
        self._min_words = min_words
        self._device = device
        self._notes_config = notes_config
        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._energy_threshold = energy_threshold

        # Audio components (created in on_mount)
        self._ring_buffer: RingBuffer | None = None
        self._vad: VoiceActivityDetector | None = None
        self._audio_queue: asyncio.Queue[np.ndarray] | None = None
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._processor_task: asyncio.Task[None] | None = None

        # VAD config for deferred creation
        self._vad_config = VadConfig(
            frame_ms=vad_frame_ms,
            mode=vad_mode,
            silence_ms=vad_silence_ms,
            sample_rate=self._sample_rate,
        )

        # Concurrency
        self._gpu_lock = asyncio.Lock()

        # Transcript state
        self._current_transcript = ""
        self._last_transcribed_sample = 0
        self._turn_accumulator: list[str] = []
        self._current_partial: str = ""
        self._last_partial_change: float = 0.0
        self._last_partial_raw: str = ""
        self._recording: bool = False
        self._turn_count: int = 0
        self._committing: bool = False
        self._vad_state: str = "silence"
        self._buffer_seconds: float = 0.0
        self._asr_ms: float | None = None

    def _state_header(self) -> str:
        """Return Rich-markup header line reflecting current recording state."""
        if self._recording:
            return "[bold green]\u2500\u2500 \u25cf Listening (Space to stop) \u2500\u2500[/bold green]"
        return "[dim]\u2500\u2500 Paused (Space to record) \u2500\u2500[/dim]"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="middle"):
            yield Static(self._state_header(), id="speech-panel")
            yield RichLog(id="output-panel", wrap=True, highlight=True)
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        output = self.query_one("#output-panel", RichLog)
        path_display = _short_path(self._notes_config.output_path)
        output.write(Text(path_display, style="bold cyan"))
        output.write(Rule())

        write_session_header(self._notes_config.output_path)

        # Capture the running event loop for call_soon_threadsafe
        self._loop = asyncio.get_running_loop()

        # Create audio components
        self._ring_buffer = RingBuffer.create(
            DEFAULT_MAX_BUFFER_SECONDS, self._sample_rate
        )
        self._vad = VoiceActivityDetector(self._vad_config)
        self._audio_queue = asyncio.Queue(maxsize=DEFAULT_AUDIO_QUEUE_MAXSIZE)

        # Create audio stream (stopped — user presses Space to record)
        stream_kwargs: dict[str, Any] = {}
        if self._device is not None:
            stream_kwargs["device"] = self._device
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._vad.frame_samples,
            channels=1,
            dtype="int16",
            callback=self._audio_callback,
            **stream_kwargs,
        )

        # Launch processor as asyncio task on Textual's event loop
        self._processor_task = asyncio.create_task(self._processor())

        self._update_status_bar()
        self.set_interval(0.1, self._refresh_display)

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Sounddevice thread callback — enqueue audio for async processing."""
        if self._loop is None or self._audio_queue is None:
            return
        data = indata.reshape(-1).copy()
        self._loop.call_soon_threadsafe(
            lambda: (
                self._audio_queue.put_nowait(data)
                if not self._audio_queue.full()
                else None
            )
        )

    async def _processor(self) -> None:
        """Async loop on Textual's event loop: audio -> VAD -> ASR."""
        assert self._ring_buffer is not None
        assert self._vad is not None
        assert self._audio_queue is not None

        min_new_samples = int(self._sample_rate * 0.2)
        last_transcribe_time = self._loop.time()

        while True:
            frame: np.ndarray | None = None
            try:
                frame = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=0.05
                )
            except asyncio.TimeoutError:
                pass

            if frame is not None:
                # RMS energy gate — skip near-silent frames before VAD
                rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
                if rms < self._energy_threshold:
                    continue

                self._ring_buffer.append(frame)
                turn_complete = self._vad.process(frame)
                self._vad_state = self._vad.state
                self._buffer_seconds = self._ring_buffer.filled_seconds

                if turn_complete:
                    await self._handle_turn_complete()

            now = self._loop.time()
            if now - last_transcribe_time >= self._transcribe_interval:
                samples_since = (
                    self._ring_buffer.total_samples_written
                    - self._last_transcribed_sample
                )
                if samples_since >= min_new_samples:
                    audio_int16 = self._ring_buffer.get_recent(
                        DEFAULT_MAX_BUFFER_SECONDS
                    )

                    if audio_int16.size >= int(self._sample_rate * 0.3):
                        if not self._gpu_lock.locked():
                            async with self._gpu_lock:
                                audio = (
                                    audio_int16.astype(np.float32) / 32768.0
                                )
                                start = time.perf_counter()
                                text = await asyncio.to_thread(
                                    self._run_transcribe, audio
                                )
                                self._asr_ms = (
                                    time.perf_counter() - start
                                ) * 1000
                            if text and text != self._current_transcript:
                                self._current_transcript = text
                    self._last_transcribed_sample = (
                        self._ring_buffer.total_samples_written
                    )

                last_transcribe_time = now

    def _run_transcribe(self, audio: np.ndarray) -> str:
        """Run ASR inference — Python-level suppress only (no os.dup2)."""
        if len(audio) < self._sample_rate * 0.3:
            return ""
        parts: list[str] = []
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            for token in transcribe(
                self._model,
                self._tokenizer,
                self._feature_extractor,
                audio,
                self._language,
                context=self._asr_context,
                logit_bias=self._logit_bias,
            ):
                parts.append(token)
        return "".join(parts).strip()

    async def _handle_turn_complete(self) -> None:
        """Finalize transcript when VAD indicates turn completion."""
        if not self._current_transcript:
            return
        if not is_meaningful(self._current_transcript):
            return
        if len(self._current_transcript.split()) < self._min_words:
            return

        transcript = self._current_transcript
        vocab = self._notes_config.rewrite.vocab
        if vocab:
            transcript = apply_vocab(transcript, vocab)

        self._turn_accumulator.append(transcript)
        self._turn_count += 1
        self._current_transcript = ""
        self._current_partial = ""
        self._last_partial_raw = ""

        # Reset buffer and VAD for next turn
        self._ring_buffer.reset()
        self._vad.reset()
        self._last_transcribed_sample = self._ring_buffer.total_samples_written
        self._buffer_seconds = 0.0

    def _update_status_bar(self) -> None:
        bar = self.query_one("#status-bar", Static)
        parts: list[str] = []

        vad_icon = "\u25cf" if self._vad_state == "speech" else "\u25cb"
        parts.append(f"VAD: {vad_icon} {self._vad_state}")
        parts.append(f"Buffer: {self._buffer_seconds:.1f}s")
        if self._asr_ms is not None:
            parts.append(f"ASR: {self._asr_ms:.0f}ms")
        parts.append(f"Turns: {self._turn_count}")

        parts.append("|")
        if self._committing:
            parts.append("Rewriting...")
        elif self._recording:
            parts.append("\u25cf Recording")
        else:
            parts.append("\u25a0 Stopped")

        parts.append("|")
        parts.append("\u2423 Record/Stop  \u23ce Commit  q Quit")
        bar.update(" ".join(parts))

    def _refresh_display(self) -> None:
        self._update_status_bar()

        raw_partial = self._current_transcript
        now = time.monotonic()

        # Debounce: update displayed partial only when stable for 0.5s
        if raw_partial != self._last_partial_raw:
            self._last_partial_raw = raw_partial
            self._last_partial_change = now
        elif now - self._last_partial_change >= 0.5 and raw_partial:
            vocab = self._notes_config.rewrite.vocab
            self._current_partial = (
                apply_vocab(raw_partial, vocab) if vocab else raw_partial
            )

        # Compose left panel content
        panel = self.query_one("#speech-panel", Static)
        lines: list[str] = [self._state_header()]

        for turn_text in self._turn_accumulator:
            lines.append(turn_text)

        if self._turn_accumulator and self._current_partial:
            lines.append("---")

        if self._current_partial:
            lines.append(self._current_partial)

        panel.update("\n\n".join(lines))

    def action_commit(self) -> None:
        text_parts = list(self._turn_accumulator)
        if self._current_partial:
            text_parts.append(self._current_partial)

        if not text_parts:
            return

        combined = "\n\n".join(text_parts)
        self._turn_accumulator.clear()
        self._current_partial = ""
        self._last_partial_raw = ""
        self._committing = True
        self._update_status_bar()

        # Update left panel immediately
        panel = self.query_one("#speech-panel", Static)
        panel.update("[dim]Rewriting...[/dim]")

        self._run_rewrite(combined)

    @work(exclusive=True, thread=True)
    def _run_rewrite(self, text: str) -> None:
        result = rewrite_transcript(text, self._notes_config.rewrite)
        self.call_from_thread(self._on_rewrite_done, result)

    def _on_rewrite_done(self, result: RewriteResult) -> None:
        output = self.query_one("#output-panel", RichLog)
        if result.error:
            output.write(
                Text(f"[rewrite error: {result.error}]", style="bold red")
            )
            output.write(Text(result.original))
        else:
            output.write(Text(result.rewritten))
        output.write(Rule())

        append_turn(self._notes_config.output_path, result)

        self._committing = False
        self._update_status_bar()

    def action_toggle_recording(self) -> None:
        if not self._stream:
            return

        if self._recording:
            self._stream.stop()
            self._recording = False
        else:
            # Reset buffer/VAD so stale audio doesn't bleed into new recording
            if self._ring_buffer:
                self._ring_buffer.reset()
                # Sync: total_samples_written is NOT cleared by reset()
                self._last_transcribed_sample = (
                    self._ring_buffer.total_samples_written
                )
            if self._vad:
                self._vad.reset()
            self._current_transcript = ""
            self._buffer_seconds = 0.0
            self._stream.start()
            self._recording = True
        self._update_status_bar()

    def action_discard(self) -> None:
        if not self._turn_accumulator and not self._current_partial:
            self.notify("Nothing to discard", severity="warning", timeout=2)
            return
        self.push_screen(
            DiscardConfirmScreen(), callback=self._on_discard_confirmed
        )

    def _on_discard_confirmed(self, result: bool) -> None:
        if not result:
            return
        self._turn_accumulator.clear()
        self._current_partial = ""
        self._last_partial_raw = ""
        self._current_transcript = ""
        # Reset audio state to discard buffered audio
        if self._ring_buffer:
            self._ring_buffer.reset()
            self._last_transcribed_sample = (
                self._ring_buffer.total_samples_written
            )
        if self._vad:
            self._vad.reset()
        self._buffer_seconds = 0.0

    def action_quit_app(self) -> None:
        # Save any uncommitted text raw to file
        text_parts = list(self._turn_accumulator)
        if self._current_partial:
            text_parts.append(self._current_partial)

        if text_parts:
            combined = "\n\n".join(text_parts)
            result = RewriteResult(
                original=combined,
                rewritten="",
                model=self._notes_config.rewrite.model,
                error="uncommitted (quit)",
            )
            append_turn(self._notes_config.output_path, result)

        # Stop audio stream
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

        # Cancel processor task
        if self._processor_task:
            self._processor_task.cancel()

        self.exit()


def _short_path(path: Path) -> str:
    """Shorten a path for display, replacing home with ~."""
    home = Path.home()
    try:
        return f"~/{path.relative_to(home)}"
    except ValueError:
        return str(path)
