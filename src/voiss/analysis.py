"""LLM-based intent analysis for transcribed speech.

Analyzes completed speech turns to extract intent, entities, and
suggested actions using a small local LLM.
"""

import re
from dataclasses import dataclass
from typing import Any, Final

from voiss.env import suppress_output
from voiss.protocols import TokenizerLike

INTENT_EXPLAIN_PROMPT: Final = """Analyze this speech and respond with exactly 3 lines:
INTENT: <primary intent in 2-3 words>
ENTITIES: <key items or names, comma-separated, or "none">
ACTION: <what should happen next, one short sentence>

Speech: "{text}" /no_think"""


@dataclass(frozen=True, slots=True)
class IntentResult:
    """Immutable result of intent analysis."""

    intent: str = ""
    entities: str = ""
    action: str = ""


def analyze_intent(
    text: str,
    llm: Any,
    llm_tokenizer: TokenizerLike,
    prompt: str | None = None,
) -> IntentResult:
    """Analyze transcribed text for intent, entities, and action.

    Uses the provided LLM to generate a structured analysis.
    *prompt* overrides the default ``INTENT_EXPLAIN_PROMPT`` when supplied
    (must contain a ``{text}`` placeholder).
    Returns a frozen IntentResult with parsed fields.
    """
    from mlx_lm.generate import generate

    template = prompt or INTENT_EXPLAIN_PROMPT
    messages = [
        {"role": "user", "content": template.format(text=text)}
    ]
    prompt = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    with suppress_output():
        response = generate(
            llm, llm_tokenizer, prompt, max_tokens=100, verbose=False
        )

    # Strip reasoning tags some models emit.
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    response = response.strip()

    intent = ""
    entities = ""
    action = ""
    for line in response.split("\n"):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("INTENT:"):
            intent = line[7:].strip()
        elif upper.startswith("ENTITIES:"):
            entities = line[9:].strip()
        elif upper.startswith("ACTION:"):
            action = line[7:].strip()

    return IntentResult(intent=intent, entities=entities, action=action)
