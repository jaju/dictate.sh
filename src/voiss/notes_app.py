"""Textual TUI for notes mode: speak, accumulate, edit, commit through LLM rewrite.

Both panels are TextAreas that start read-only and non-focusable. Tab/click
selects a panel (orange border, no cursor). Press `e` to enter edit mode
(red border, cursor visible, full typing). Ctrl+S saves, Escape cancels.

This module directly owns audio capture, VAD, and ASR — it does NOT use
RealtimeTranscriber. The ASR model is pre-loaded by notes.run_notes_pipeline()
and passed in via constructor to avoid subprocess/fd conflicts with Textual.
"""

import asyncio
import contextlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Click
from textual.screen import ModalScreen
from textual.widgets import Header, ListItem, ListView, Markdown, Static, TextArea

from voiss.audio.ring_buffer import RingBuffer
from voiss.audio.vad import VadConfig, VoiceActivityDetector
from voiss.constants import (
    DEFAULT_AUDIO_QUEUE_MAXSIZE,
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_MAX_BUFFER_SECONDS,
    DEFAULT_SAMPLE_RATE,
)
from voiss.notes import NotesConfig, append_raw, append_turn, write_session_header
from voiss.protocols import FeatureExtractorLike, TokenizerLike
from voiss.rewrite import PostprocessResult, apply_vocab, postprocess_transcript
from voiss.transcribe import is_meaningful, transcribe

# Maps selected panel name to (container id, editor id)
_PANEL_IDS: dict[str, tuple[str, str]] = {
    "left": ("left-container", "speech-editor"),
    "right": ("right-container", "output-editor"),
}

_HISTORY_PREVIEW_MAX = 80


def _make_preview(text: str) -> str:
    """Collapse whitespace and truncate for one-line history preview."""
    oneline = " ".join(text.split())
    if len(oneline) > _HISTORY_PREVIEW_MAX:
        return oneline[: _HISTORY_PREVIEW_MAX - 1] + "\u2026"
    return oneline


@dataclass(slots=True)
class HistoryEntry:
    """A committed turn stored in the history index."""

    raw_text: str
    preview: str


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


class QuitConfirmScreen(ModalScreen[bool]):
    """Modal confirmation for quitting the app."""

    CSS = """
    QuitConfirmScreen {
        align: center middle;
    }
    #quit-dialog {
        width: 50;
        height: 5;
        border: thick $warning;
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
            "Quit and save? (y/n)", id="quit-dialog"
        )

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)


class VoissNotesApp(App):
    """Textual TUI for voice-driven notes with modal editable panels."""

    TITLE = "Voiss Notes"

    CSS = """
    #middle {
        height: 1fr;
    }
    #left-container {
        width: 40%;
        border: solid $primary;
    }
    #left-container.-selected {
        border: solid $accent;
    }
    #left-container.-editing {
        border: heavy $error;
        border-title-align: right;
        border-title-color: $error;
        border-subtitle-align: right;
        border-subtitle-color: $error;
    }
    #speech-header {
        height: auto;
        padding: 0 1;
    }
    #history-list {
        height: auto;
        max-height: 30%;
        min-height: 3;
        border-bottom: solid $primary-lighten-3;
        overflow-y: auto;
    }
    #history-list.-locked {
        opacity: 0.35;
    }
    #history-list > ListItem {
        height: 1;
        padding: 0 1;
    }
    #speech-editor {
        height: 1fr;
        border: none;
        overflow-y: auto;
    }
    #left-container.-previewing #speech-editor {
        border: dashed $warning;
    }
    #right-container {
        width: 60%;
        border: solid $primary;
    }
    #right-container.-selected {
        border: solid $accent;
    }
    #right-container.-editing {
        border: heavy $error;
        border-title-align: right;
        border-title-color: $error;
        border-subtitle-align: right;
        border-subtitle-color: $error;
    }
    #output-editor {
        display: none;
        height: 1fr;
        border: none;
        overflow-y: auto;
    }
    #output-display {
        height: 1fr;
        overflow-y: auto;
    }
    #right-container.-editing #output-editor {
        display: block;
    }
    #right-container.-editing #output-display {
        display: none;
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
        Binding("r", "commit_rewrite", "Rewrite", priority=True),
        Binding("space", "toggle_recording", "Record/Stop", priority=True),
        Binding("e", "edit_panel", "Edit", priority=True),
        Binding("E", "edit_external", "Ext Edit", priority=True),
        Binding("escape", "escape_action", "Esc", priority=True),
        Binding("q", "quit_app", "Quit", priority=True),
        Binding("tab", "cycle_panel", "Tab", priority=True),
        Binding("ctrl+s", "save_edit", "Save", priority=True, show=False),
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
        max_buffer_seconds: int = DEFAULT_MAX_BUFFER_SECONDS,
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
        self._max_buffer_seconds = max_buffer_seconds

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

        # Right panel dirty flag for debounced file writes
        self._output_dirty: bool = False

        # Panel selection & modal editing state
        self._selected_panel: str | None = None  # "left" or "right"
        self._editing: bool = False
        self._editing_target: str | None = None  # "speech-editor" or "output-editor"
        self._edit_snapshot: str = ""  # for cancel/revert

        # History index state
        self._history: list[HistoryEntry] = []
        self._previewing_history: bool = False
        self._previewing_index: int | None = None

    def _state_header(self) -> str:
        """Return Rich-markup header line reflecting current recording state."""
        if self._recording:
            return "[bold green]\u2500\u2500 \u25cf Listening (Space to stop) \u2500\u2500[/bold green]"
        return "[dim]\u2500\u2500 Paused (Space to record) \u2500\u2500[/dim]"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="middle"):
            with Vertical(id="left-container"):
                yield Static(self._state_header(), id="speech-header")
                yield ListView(id="history-list")
                yield TextArea("", id="speech-editor", read_only=True)
            with Vertical(id="right-container"):
                yield Markdown("", id="output-display")
                yield TextArea("", id="output-editor", read_only=True)
        yield Static("", id="status-bar")

    def on_mount(self) -> None:
        # Disable focus on both TextAreas — no cursor, no arrow keys
        self.query_one("#speech-editor", TextArea).can_focus = False
        self.query_one("#output-editor", TextArea).can_focus = False
        self.query_one("#output-display", Markdown).can_focus = False

        # History list starts locked and dimmed (empty)
        history_lv = self.query_one("#history-list", ListView)
        history_lv.can_focus = False
        history_lv.add_class("-locked")

        write_session_header(self._notes_config.output_path)

        # Load notes file into right panel (both widgets)
        try:
            content = self._notes_config.output_path.read_text()
            self.query_one("#output-editor", TextArea).text = content
            self.query_one("#output-display", Markdown).update(content)
        except FileNotFoundError:
            pass

        # Capture the running event loop for call_soon_threadsafe
        self._loop = asyncio.get_running_loop()

        # Create audio components
        self._ring_buffer = RingBuffer.create(
            self._max_buffer_seconds, self._sample_rate
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
        self.set_interval(1.0, self._save_output_if_dirty)

    # ── Click handling ───────────────────────────────────────────────

    def on_click(self, event: Click) -> None:
        """Select panel on click (when not editing)."""
        if self._editing:
            return
        left = self.query_one("#left-container")
        right = self.query_one("#right-container")
        x, y = event.screen_x, event.screen_y
        if left.region.contains(x, y):
            self._select_panel("left")
        elif right.region.contains(x, y):
            self._select_panel("right")

    # ── Panel selection ──────────────────────────────────────────────

    def _select_panel(self, panel: str) -> None:
        """Select a panel (orange border). Does NOT enter edit mode."""
        left_c = self.query_one("#left-container")
        right_c = self.query_one("#right-container")
        left_c.remove_class("-selected")
        right_c.remove_class("-selected")
        container_id, _ = _PANEL_IDS[panel]
        self.query_one(f"#{container_id}").add_class("-selected")
        self._selected_panel = panel

        # Auto-focus history list when selecting left panel with history available
        if panel == "left" and not self._recording and self._history:
            lv = self.query_one("#history-list", ListView)
            lv.focus()

        self._update_status_bar()

    # ── Audio ────────────────────────────────────────────────────────

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
                        self._max_buffer_seconds
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
        vocab = self._notes_config.vocab
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

    # ── Display ──────────────────────────────────────────────────────

    def _update_status_bar(self) -> None:
        bar = self.query_one("#status-bar", Static)
        parts: list[str] = []

        vad_icon = "\u25cf" if self._vad_state == "speech" else "\u25cb"
        parts.append(f"VAD: {vad_icon} {self._vad_state}")
        fill_pct = (self._buffer_seconds / self._max_buffer_seconds) * 100 if self._max_buffer_seconds else 0
        if fill_pct >= 90:
            buf_str = f"[bold red]Buffer: {self._buffer_seconds:.0f}/{self._max_buffer_seconds}s ({fill_pct:.0f}%)[/bold red]"
        elif fill_pct >= 70:
            buf_str = f"[yellow]Buffer: {self._buffer_seconds:.0f}/{self._max_buffer_seconds}s ({fill_pct:.0f}%)[/yellow]"
        else:
            buf_str = f"Buffer: {self._buffer_seconds:.0f}/{self._max_buffer_seconds}s"
        parts.append(buf_str)
        if self._asr_ms is not None:
            parts.append(f"ASR: {self._asr_ms:.0f}ms")
        parts.append(f"Turns: {self._turn_count}")
        if self._history:
            parts.append(f"History: {len(self._history)}")
        if self._previewing_history and self._previewing_index is not None:
            parts.append(f"Viewing #{self._previewing_index + 1}")

        parts.append("|")
        if self._editing:
            parts.append("^S Save  \u238b Cancel")
        elif self._previewing_history:
            parts.append(
                "\u2191\u2193 Navigate  \u238b Exit preview"
            )
        else:
            parts.append(
                "\u2423 Rec  \u21e5 Select  e Edit  E Ext Edit  \u23ce Raw  r Rewrite  \u238b Discard  q Quit"
            )
        bar.update(" ".join(parts))

        # Header subtitle — mode indicator
        if self._editing:
            panel = "speech" if self._editing_target == "speech-editor" else "notes"
            self.sub_title = f"Editing {panel}"
        elif self._committing:
            self.sub_title = "Rewriting\u2026"
        elif self._recording:
            self.sub_title = "\u25cf Recording"
        else:
            self.sub_title = ""

    def _refresh_display(self) -> None:
        self._update_status_bar()

        # Update speech header
        header = self.query_one("#speech-header", Static)
        header.update(self._state_header())

        raw_partial = self._current_transcript
        now = time.monotonic()

        # Debounce: update displayed partial only when stable for 0.5s
        if raw_partial != self._last_partial_raw:
            self._last_partial_raw = raw_partial
            self._last_partial_change = now
        elif now - self._last_partial_change >= 0.5 and raw_partial:
            vocab = self._notes_config.vocab
            self._current_partial = (
                apply_vocab(raw_partial, vocab) if vocab else raw_partial
            )

        # Only update left editor when NOT being edited and NOT previewing history
        if not (self._editing and self._editing_target == "speech-editor"):
            if not self._previewing_history:
                self._refresh_live_text()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Track right-panel edits for debounced file save."""
        if event.text_area.id == "output-editor":
            self._output_dirty = True

    def _save_output_if_dirty(self) -> None:
        """Debounced write of right panel content to notes file."""
        if self._output_dirty:
            output = self.query_one("#output-editor", TextArea)
            self._notes_config.output_path.write_text(output.text)
            self._output_dirty = False

    def _reload_right_panel(self) -> None:
        """Reload file content into both right-panel widgets."""
        try:
            content = self._notes_config.output_path.read_text()
            self.query_one("#output-editor", TextArea).text = content
            md = self.query_one("#output-display", Markdown)
            md.update(content)
            md.scroll_end(animate=False)
            self._output_dirty = False
        except FileNotFoundError:
            pass

    # ── Edit mode helpers ────────────────────────────────────────────

    def _exit_edit_mode(self) -> None:
        """Leave edit mode: restore read-only, disable focus, keep selected."""
        if self._editing_target:
            editor = self.query_one(f"#{self._editing_target}", TextArea)
            editor.read_only = True
            editor.can_focus = False

            # Swap container class from -editing back to -selected
            container_id = (
                "left-container"
                if self._editing_target == "speech-editor"
                else "right-container"
            )
            container = self.query_one(f"#{container_id}")
            container.remove_class("-editing")
            container.add_class("-selected")
            container.border_title = ""
            container.border_subtitle = ""

            # Sync TextArea back to Markdown on right panel exit
            if self._editing_target == "output-editor":
                md = self.query_one("#output-display", Markdown)
                md.update(editor.text)

        self.set_focus(None)
        self._editing = False
        self._editing_target = None
        self._edit_snapshot = ""
        self._update_status_bar()

    def _insert_into_editor(self, char: str) -> None:
        """Insert a character into the active editing TextArea."""
        if self._editing_target:
            editor = self.query_one(f"#{self._editing_target}", TextArea)
            editor.insert(char)

    def _read_left_panel(self) -> str:
        """Read text from the left panel."""
        if self._editing_target == "speech-editor":
            editor = self.query_one("#speech-editor", TextArea)
            return editor.text.strip()

        text_parts = list(self._turn_accumulator)
        if self._current_partial:
            text_parts.append(self._current_partial)
        return "\n\n".join(text_parts)

    def _sync_left_editor_to_accumulator(self) -> None:
        """Sync left TextArea content back to the turn accumulator."""
        editor = self.query_one("#speech-editor", TextArea)
        text = editor.text.strip()
        self._turn_accumulator.clear()
        if text:
            self._turn_accumulator.append(text)
        self._current_partial = ""
        self._last_partial_raw = ""
        self._current_transcript = ""

    # ── History helpers ──────────────────────────────────────────────

    def _add_history_entry(self, text: str) -> None:
        """Create a HistoryEntry and append it to the history ListView."""
        preview = _make_preview(text)
        entry = HistoryEntry(raw_text=text, preview=preview)
        self._history.append(entry)
        lv = self.query_one("#history-list", ListView)
        lv.append(ListItem(Static(preview)))

    def _exit_history_preview(self) -> None:
        """Exit history preview mode and restore live TextArea content."""
        self._previewing_history = False
        self._previewing_index = None
        lv = self.query_one("#history-list", ListView)
        lv.index = None
        container = self.query_one("#left-container")
        container.remove_class("-previewing")
        self._refresh_live_text()

    def _refresh_live_text(self) -> None:
        """Rebuild TextArea from turn accumulator + current partial."""
        editor = self.query_one("#speech-editor", TextArea)
        lines: list[str] = []
        for turn_text in self._turn_accumulator:
            lines.append(turn_text)
        if self._turn_accumulator and self._current_partial:
            lines.append("---")
        if self._current_partial:
            lines.append(self._current_partial)
        new_text = "\n\n".join(lines)
        if editor.text != new_text:
            editor.text = new_text

    def _show_history_preview(self, index: int) -> None:
        """Show a history entry's full text in the TextArea."""
        self._previewing_history = True
        self._previewing_index = index
        editor = self.query_one("#speech-editor", TextArea)
        editor.text = self._history[index].raw_text
        container = self.query_one("#left-container")
        container.add_class("-previewing")
        self._update_status_bar()

    def _update_history_lock(self) -> None:
        """Lock/unlock the history list based on recording state."""
        lv = self.query_one("#history-list", ListView)
        if self._recording or not self._history:
            lv.can_focus = False
            lv.add_class("-locked")
        else:
            lv.can_focus = True
            lv.remove_class("-locked")

    # ── Commit helpers ───────────────────────────────────────────────

    def _prepare_commit(self) -> str | None:
        """Shared commit preparation: read left panel, clear state, reset audio.

        Returns the combined text, or None if editing or nothing to commit.
        """
        if self._editing:
            return None

        if self._previewing_history:
            self.notify(
                "Exit history preview first (\u238b)",
                severity="warning",
                timeout=2,
            )
            return None

        combined = self._read_left_panel()
        if not combined:
            return None

        # Sync and clear state
        self._turn_accumulator.clear()
        self._current_transcript = ""
        self._current_partial = ""
        self._last_partial_raw = ""
        self._committing = True

        # Clear left editor and add to history
        editor = self.query_one("#speech-editor", TextArea)
        editor.text = ""
        self._add_history_entry(combined)
        self._update_history_lock()

        # Reset audio state so stale buffer doesn't re-populate the panel
        if self._ring_buffer:
            self._ring_buffer.reset()
            self._last_transcribed_sample = (
                self._ring_buffer.total_samples_written
            )
        if self._vad:
            self._vad.reset()
        self._buffer_seconds = 0.0

        self._update_status_bar()
        return combined

    # ── Action handlers ──────────────────────────────────────────────

    def action_cycle_panel(self) -> None:
        """Tab: cycle selected panel (or insert tab in edit mode)."""
        if self._editing:
            self._insert_into_editor("    ")
            return
        if self._selected_panel == "left":
            self._select_panel("right")
        else:
            self._select_panel("left")

    def action_edit_panel(self) -> None:
        """Enter edit mode on the selected panel."""
        if self._editing:
            self._insert_into_editor("e")
            return

        if self._selected_panel is None:
            self.notify(
                "Select a panel first (Tab or click)",
                severity="warning",
                timeout=2,
            )
            return

        if self._selected_panel == "left" and self._previewing_history:
            self.notify(
                "Exit history preview first (\u238b)",
                severity="warning",
                timeout=2,
            )
            return

        container_id, editor_id = _PANEL_IDS[self._selected_panel]
        editor = self.query_one(f"#{editor_id}", TextArea)
        container = self.query_one(f"#{container_id}")

        self._editing = True
        self._editing_target = editor_id
        self._edit_snapshot = editor.text

        editor.read_only = False
        editor.can_focus = True
        editor.focus()

        container.remove_class("-selected")
        container.add_class("-editing")
        container.border_title = "EDIT"
        container.border_subtitle = "^S Save  \u238b Cancel"

        self._update_status_bar()

    def action_edit_external(self) -> None:
        """Open focused panel content in $EDITOR."""
        if self._editing:
            self._insert_into_editor("E")
            return
        if self._selected_panel is None:
            self.notify(
                "Select a panel first (Tab or click)",
                severity="warning",
                timeout=2,
            )
            return
        if self._selected_panel == "left" and self._previewing_history:
            self.notify(
                "Exit history preview first (\u238b)",
                severity="warning",
                timeout=2,
            )
            return
        self.run_worker(self._edit_in_external_editor(), exclusive=True)

    async def _edit_in_external_editor(self) -> None:
        import os
        import shlex
        import subprocess
        import tempfile

        # Read current content from the selected panel
        if self._selected_panel == "left":
            content = self._read_left_panel()
        else:
            editor_widget = self.query_one("#output-editor", TextArea)
            content = editor_widget.text

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, prefix="voiss-"
        ) as f:
            f.write(content)
            tmp_path = f.name

        try:
            editor_cmd = os.environ.get("EDITOR", "vi")
            cmd_parts = shlex.split(editor_cmd)
            cmd_parts.append(tmp_path)
            with self.suspend():
                subprocess.call(cmd_parts)
            result = Path(tmp_path).read_text()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Apply result back
        if self._selected_panel == "left":
            self._turn_accumulator.clear()
            if result.strip():
                self._turn_accumulator.append(result.strip())
            self._current_partial = ""
            self._current_transcript = ""
            left_editor = self.query_one("#speech-editor", TextArea)
            left_editor.text = result.strip()
        else:
            output_editor = self.query_one("#output-editor", TextArea)
            output_editor.text = result
            self._output_dirty = True
            self._save_output_if_dirty()
            md = self.query_one("#output-display", Markdown)
            md.update(result)

    def action_save_edit(self) -> None:
        """Save edit and exit edit mode (Ctrl+S)."""
        if not self._editing:
            return

        if self._editing_target == "speech-editor":
            self._sync_left_editor_to_accumulator()
        elif self._editing_target == "output-editor":
            self._save_output_if_dirty()

        self._exit_edit_mode()
        self.notify("Saved", severity="information", timeout=1)

    def action_escape_action(self) -> None:
        """Context-sensitive Escape: cancel edit, exit preview, or discard."""
        if self._editing:
            # Revert to snapshot
            if self._editing_target:
                editor = self.query_one(f"#{self._editing_target}", TextArea)
                editor.text = self._edit_snapshot
            self._exit_edit_mode()
            self.notify("Edit cancelled", severity="warning", timeout=1)
            return

        if self._previewing_history:
            self._exit_history_preview()
            self._update_status_bar()
            return

        # Normal mode: discard
        combined = self._read_left_panel()
        if not combined:
            self.notify("Nothing to discard", severity="warning", timeout=2)
            return
        self.push_screen(
            DiscardConfirmScreen(), callback=self._on_discard_confirmed
        )

    def action_commit(self) -> None:
        """Enter: commit raw text (with vocab corrections, no LLM rewrite)."""
        if self._editing:
            self._insert_into_editor("\n")
            return

        combined = self._prepare_commit()
        if not combined:
            return

        # Apply vocab corrections (no LLM)
        vocab = self._notes_config.vocab
        if vocab:
            combined = apply_vocab(combined, vocab)

        self._save_output_if_dirty()
        append_raw(self._notes_config.output_path, combined)
        self._reload_right_panel()
        self._committing = False
        self._update_status_bar()

    def action_commit_rewrite(self) -> None:
        """r: commit with LLM rewrite."""
        if self._editing:
            self._insert_into_editor("r")
            return

        combined = self._prepare_commit()
        if not combined:
            return

        self._run_rewrite(combined)

    @work(exclusive=True, thread=True)
    def _run_rewrite(self, text: str) -> None:
        result = postprocess_transcript(
            text, self._notes_config.postprocess, self._notes_config.vocab,
        )
        self.call_from_thread(self._on_rewrite_done, result)

    def _on_rewrite_done(self, result: PostprocessResult) -> None:
        # Flush any pending user edits to file before appending
        self._save_output_if_dirty()

        # Append structured turn to file
        append_turn(self._notes_config.output_path, result)

        self._reload_right_panel()
        self._committing = False
        self._update_status_bar()

    def action_toggle_recording(self) -> None:
        if self._editing:
            self._insert_into_editor(" ")
            return

        if not self._stream:
            return

        if self._recording:
            self._stream.stop()
            self._recording = False
            self._update_history_lock()
        else:
            # Exit history preview if active
            if self._previewing_history:
                self._exit_history_preview()
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
            self._update_history_lock()
        self._update_status_bar()

    def _on_discard_confirmed(self, result: bool) -> None:
        if not result:
            return
        self._turn_accumulator.clear()
        self._current_partial = ""
        self._last_partial_raw = ""
        self._current_transcript = ""

        # Clear left editor
        editor = self.query_one("#speech-editor", TextArea)
        editor.text = ""

        # Reset audio state to discard buffered audio
        if self._ring_buffer:
            self._ring_buffer.reset()
            self._last_transcribed_sample = (
                self._ring_buffer.total_samples_written
            )
        if self._vad:
            self._vad.reset()
        self._buffer_seconds = 0.0

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Show history entry in TextArea when highlighted."""
        if self._recording or self._editing:
            return
        lv = self.query_one("#history-list", ListView)
        idx = lv.index
        if idx is not None and 0 <= idx < len(self._history):
            self._show_history_preview(idx)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Suppress Enter on history list — user must exit preview first."""
        event.prevent_default()
        event.stop()
        self.notify(
            "Exit history preview first (\u238b)",
            severity="warning",
            timeout=2,
        )

    def action_quit_app(self) -> None:
        if self._editing:
            self._insert_into_editor("q")
            return

        self.push_screen(
            QuitConfirmScreen(), callback=self._on_quit_confirmed
        )

    def _on_quit_confirmed(self, result: bool) -> None:
        if not result:
            return

        # Flush any pending right-panel edits
        self._save_output_if_dirty()

        # Save any uncommitted left-panel text raw to file
        combined = self._read_left_panel()
        if combined:
            result_rw = PostprocessResult(
                original=combined,
                rewritten="",
                model=self._notes_config.postprocess.model or "",
                error="uncommitted (quit)",
            )
            append_turn(self._notes_config.output_path, result_rw)

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
