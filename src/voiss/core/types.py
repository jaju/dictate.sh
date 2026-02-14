"""Core data types shared across voiss modules."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IntentResult:
    """Immutable result of intent analysis."""

    intent: str = ""
    entities: str = ""
    action: str = ""
