"""Terminal UI rendering for voiss.

All render functions are pure: they take a UiState snapshot and return
Rich renderables. No side effects, no mutation.
"""

from dataclasses import dataclass, field

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from voiss.analysis import IntentResult


@dataclass(slots=True)
class UiState:
    """Snapshot of pipeline state consumed by render functions."""

    status: str = "Starting"
    partial_transcript: str = ""
    history: list[tuple[str, IntentResult | None]] = field(default_factory=list)
    max_history: int = 50
    vad_state: str = "silence"
    buffer_seconds: float = 0.0
    queue_size: int = 0
    asr_ms: float | None = None
    analysis_ms: float | None = None
    language: str = "English"
    vad_mode: int = 2
    vad_frame_ms: int = 30
    model_path: str = ""
    llm_model_name: str = ""
    analyze_enabled: bool = False
    turn_complete: bool = False


def short_model_name(name: str | None) -> str:
    """Extract the last path segment for display."""
    if not name:
        return "--"
    return name.split("/")[-1]


def render_status_panel(state: UiState) -> Panel:
    """Render the top status bar."""
    status = Text()
    status.append("Status: ", style="bold")
    status_style = "green" if state.status == "Listening" else "yellow"
    status.append(state.status, style=status_style)
    status.append(" | ")
    status.append(f"Language: {state.language}")
    status.append(" | ")
    status.append(f"VAD: mode {state.vad_mode}, {state.vad_frame_ms}ms")
    status.append(" | ")
    status.append(f"ASR: {short_model_name(state.model_path)}")
    if state.analyze_enabled:
        status.append(" | ")
        status.append(f"LLM: {short_model_name(state.llm_model_name)}")
    return Panel(status, title="Status", padding=(0, 1))


def render_transcript_panel(state: UiState) -> Panel:
    """Render the transcript history and partial output."""
    body = Text()
    for transcript, analysis in state.history[-state.max_history :]:
        body.append("> ", style="bold green")
        body.append(transcript)
        body.append("\n")
        if analysis:
            if analysis.intent:
                body.append("Intent: ", style="cyan")
                body.append(analysis.intent)
                body.append("\n")
            if analysis.entities and analysis.entities.lower() != "none":
                body.append("Entities: ", style="cyan")
                body.append(analysis.entities)
                body.append("\n")
            if analysis.action:
                body.append("Action: ", style="cyan")
                body.append(analysis.action)
                body.append("\n")
        body.append("\n")

    if state.partial_transcript and not state.turn_complete:
        body.append("... ", style="dim")
        body.append(state.partial_transcript, style="dim")
        body.append("\n")

    if not body.plain:
        body.append("Waiting for speech...", style="dim")
    return Panel(body, title="Transcript", padding=(0, 1))


def render_stats_panel(state: UiState) -> Panel:
    """Render the stats table."""
    stats = Table.grid(expand=True, padding=(0, 1))
    stats.add_column(justify="right", style="cyan")
    stats.add_column()
    stats.add_row("VAD", state.vad_state)
    stats.add_row("Buffer", f"{state.buffer_seconds:.1f}s")
    stats.add_row("Queue", str(state.queue_size))
    stats.add_row(
        "ASR",
        f"{state.asr_ms:.0f} ms" if state.asr_ms is not None else "--",
    )
    stats.add_row(
        "Analysis",
        f"{state.analysis_ms:.0f} ms"
        if state.analysis_ms is not None
        else "--",
    )
    return Panel(stats, title="Stats", padding=(0, 1))


def render_layout(state: UiState) -> Layout:
    """Compose the full terminal layout from state."""
    layout = Layout()
    layout.split_column(
        Layout(render_status_panel(state), name="status", size=3),
        Layout(render_transcript_panel(state), name="transcript", ratio=2),
        Layout(render_stats_panel(state), name="stats", size=7),
    )
    return layout
