"""
Loader for BigSpin invisible-failure annotation files.

Parses the pandas-serialised JSON format produced by the BigSpin pipeline
and returns a list of BigSpinRecord objects, each pairing the raw BigSpin
annotations with a Conversation object ready for the culture-chat pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models import Conversation, Turn


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BigSpinSignal:
    """One BigSpin failure signal on a conversation."""
    signal_type: str
    severity: int                  # 1–3 as in BigSpin schema
    evidence: str
    turn: int
    notes: str


@dataclass
class BigSpinRecord:
    """A single row from the BigSpin annotation file."""
    conversation_id: int
    model: str
    overall_quality: str           # good | acceptable | poor | critical | None
    signal_count: int
    transcript_summary: str
    primary_failure_mode: str | None
    signals: list[BigSpinSignal]
    conversation: Conversation     # culture-chat Conversation object


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_pandas_field(value: Any) -> dict:
    """
    Each field in the BigSpin JSON is stored as a pandas-serialised dict
    (a JSON string of {str_index: value}).  Handle both already-parsed
    dicts and raw strings.
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    # Replace Python None literals before JSON-parsing
    cleaned = re.sub(r"\bNone\b", "null", value)
    # Replace single quotes around keys/values that JSON doesn't accept
    # (only needed if the field was serialised with repr instead of json.dumps)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def _parse_transcript(text: str) -> list[Turn]:
    """
    Parse a transcript with 'User:' / 'Bot:' prefixes into Turn objects.
    Turns are separated by these prefixes on new lines.
    """
    turns: list[Turn] = []
    current_role: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("User:"):
            if current_role is not None:
                turns.append(Turn(role=current_role, content="\n".join(current_lines).strip()))
            current_role = "user"
            current_lines = [line[len("User:"):].strip()]
        elif line.startswith("Bot:"):
            if current_role is not None:
                turns.append(Turn(role=current_role, content="\n".join(current_lines).strip()))
            current_role = "assistant"
            current_lines = [line[len("Bot:"):].strip()]
        else:
            current_lines.append(line)

    if current_role is not None and current_lines:
        turns.append(Turn(role=current_role, content="\n".join(current_lines).strip()))

    return turns


def _parse_signals(signals_dict: dict) -> list[BigSpinSignal]:
    """Convert a {signal_type: {severity, evidence, turn, notes}} dict."""
    out: list[BigSpinSignal] = []
    for sig_type, details in signals_dict.items():
        if not isinstance(details, dict):
            continue
        out.append(BigSpinSignal(
            signal_type=sig_type,
            severity=int(details.get("severity", 1)),
            evidence=str(details.get("evidence", "")),
            turn=int(details.get("turn", 0)),
            notes=str(details.get("notes", "")),
        ))
    return out


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_bigspin(path: str | Path) -> list[BigSpinRecord]:
    """
    Load a BigSpin annotation JSON file and return a list of BigSpinRecords.

    Parameters
    ----------
    path : str | Path
        Path to the JSON file (e.g. wildchat_bigspin_annotations_2k_opus_v1.json).

    Returns
    -------
    list[BigSpinRecord]
        One record per conversation, with the BigSpin annotations and a
        culture-chat Conversation object populated from the transcript.
    """
    with open(path) as f:
        raw = json.load(f)

    conv_ids       = _parse_pandas_field(raw.get("conversation_id", {}))
    models_field   = _parse_pandas_field(raw.get("model", {}))
    quality_field  = _parse_pandas_field(raw.get("overall_quality", {}))
    sig_count_field= _parse_pandas_field(raw.get("signal_count", {}))
    summary_field  = _parse_pandas_field(raw.get("transcript_summary", {}))
    primary_field  = _parse_pandas_field(raw.get("primary_failure_mode", {}))
    signals_field  = _parse_pandas_field(raw.get("signals_json", {}))
    transcript_field = _parse_pandas_field(raw.get("transcript", {}))

    records: list[BigSpinRecord] = []

    for idx in conv_ids:
        conv_id   = int(conv_ids[idx]) if conv_ids.get(idx) is not None else -1
        transcript_text = transcript_field.get(idx, "")
        turns = _parse_transcript(transcript_text) if transcript_text else []

        signals_raw = signals_field.get(idx, {})
        signals = _parse_signals(signals_raw) if isinstance(signals_raw, dict) else []

        conversation = Conversation(
            conversation_id=str(conv_id),
            turns=turns,
            metadata={
                "bigspin_quality":       quality_field.get(idx),
                "bigspin_signals":       list(signals_raw.keys()) if isinstance(signals_raw, dict) else [],
                "bigspin_primary":       primary_field.get(idx),
                "bigspin_signal_count":  sig_count_field.get(idx, 0),
                "source":                "bigspin",
            },
        )

        records.append(BigSpinRecord(
            conversation_id=conv_id,
            model=str(models_field.get(idx, "")),
            overall_quality=str(quality_field.get(idx, "")),
            signal_count=int(sig_count_field.get(idx) or 0),
            transcript_summary=str(summary_field.get(idx, "")),
            primary_failure_mode=primary_field.get(idx),
            signals=signals,
            conversation=conversation,
        ))

    return records
