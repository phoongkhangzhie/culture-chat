"""
WildChat dataset loader.

Streams the WildChat-1M dataset from HuggingFace and yields conversations
that meet the minimum turn count threshold.

WildChat schema reference:
  https://huggingface.co/datasets/allenai/WildChat-1M
  Each row has a `conversation` field — a list of dicts with keys:
    role:     "user" | "assistant"
    content:  str
  Plus top-level fields: conversation_id, language, country, ...
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

from src.models import Conversation, Turn

logger = logging.getLogger(__name__)


def _row_to_conversation(row: dict) -> Conversation:
    turns = [
        Turn(role=msg["role"], content=msg["content"])
        for msg in row.get("conversation", [])
        if msg.get("content")  # skip empty turns
    ]
    return Conversation(
        conversation_id=str(row.get("conversation_id", "")),
        turns=turns,
        language=row.get("language"),
        country=row.get("country"),
        metadata={
            k: row[k]
            for k in ("model", "toxic", "redacted", "state")
            if k in row
        },
    )


def load_wildchat(
    min_turns: int = 6,
    split: str = "train",
    max_conversations: Optional[int] = None,
    language_filter: Optional[str] = None,
) -> Iterator[Conversation]:
    """
    Stream WildChat conversations that satisfy the filters.

    Args:
        min_turns:          Minimum number of turns (user + assistant combined).
                            Default is 6 (i.e. at least 3 user + 3 assistant turns).
        split:              Dataset split — "train" is the only split in WildChat-1M.
        max_conversations:  Hard cap on how many conversations to yield.
        language_filter:    If set, only yield conversations whose `language`
                            field matches (case-insensitive, e.g. "English").
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the `datasets` package: pip install datasets"
        ) from exc

    dataset = load_dataset(
        "allenai/WildChat-1M",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    yielded = 0
    for row in dataset:
        conv = _row_to_conversation(row)

        if len(conv.turns) < min_turns:
            continue

        if language_filter and (
            conv.language or ""
        ).lower() != language_filter.lower():
            continue

        yield conv
        yielded += 1

        if max_conversations is not None and yielded >= max_conversations:
            logger.info("Reached max_conversations=%d, stopping.", max_conversations)
            break
