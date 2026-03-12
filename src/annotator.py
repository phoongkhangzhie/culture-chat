"""
Cultural annotation pipeline for WildChat conversations.

Given a conversation, calls an Anthropic model and returns a structured
Annotation object using the CultureScope taxonomy.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import anthropic

from src.models import Annotation, Conversation, DimensionMatch
from src.prompts import SYSTEM_PROMPT, build_annotation_prompt
from src.taxonomy import CULTURAL_DIMENSIONS

logger = logging.getLogger(__name__)


class Annotator:
    """
    Wraps the Anthropic client and annotates WildChat conversations.

    Usage::

        annotator = Annotator()
        annotation = annotator.annotate(conversation)
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        client: Optional[anthropic.Anthropic] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = client or anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(self, conversation: Conversation) -> Annotation:
        """Annotate a single conversation. Returns an Annotation."""
        user_prompt = build_annotation_prompt(
            conversation.turns, CULTURAL_DIMENSIONS
        )
        raw = self._call_with_retry(user_prompt)
        parsed = self._parse_response(raw, conversation.conversation_id)
        parsed.num_turns = len(conversation.turns)
        parsed.metadata["model"] = self.model
        parsed.metadata["annotated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if conversation.country:
            parsed.metadata["country"] = conversation.country
        if conversation.language:
            parsed.metadata["language"] = conversation.language
        return parsed

    def annotate_batch(
        self,
        conversations: list[Conversation],
        on_error: str = "skip",  # "skip" | "raise"
    ) -> list[Annotation]:
        """
        Annotate a list of conversations sequentially.

        Args:
            conversations: List of Conversation objects to annotate.
            on_error:       What to do on a single failure — "skip" logs a
                            warning and continues; "raise" re-raises the error.
        """
        results: list[Annotation] = []
        for i, conv in enumerate(conversations):
            logger.info(
                "Annotating conversation %d/%d  id=%s",
                i + 1,
                len(conversations),
                conv.conversation_id,
            )
            try:
                results.append(self.annotate(conv))
            except Exception as exc:
                if on_error == "raise":
                    raise
                logger.warning(
                    "Skipping conversation %s due to error: %s",
                    conv.conversation_id,
                    exc,
                )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, user_prompt: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return message.content[0].text
            except anthropic.RateLimitError as exc:
                logger.warning("Rate limited (attempt %d/%d). Retrying in %ss.", attempt, self.max_retries, self.retry_delay * attempt)
                time.sleep(self.retry_delay * attempt)
                last_exc = exc
            except anthropic.APIError as exc:
                logger.warning("API error (attempt %d/%d): %s", attempt, self.max_retries, exc)
                time.sleep(self.retry_delay)
                last_exc = exc
        raise RuntimeError(f"All {self.max_retries} attempts failed") from last_exc

    def _parse_response(self, raw: str, conversation_id: str) -> Annotation:
        """Parse the model's JSON response into an Annotation."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Model returned invalid JSON for conversation {conversation_id}:\n{raw}"
            ) from exc

        dimension_matches = [
            DimensionMatch(
                dimension_key=d.get("dimension_key", ""),
                dimension_name=d.get("dimension_name", ""),
                category=d.get("category", ""),
                indicators=d.get("indicators", []),
                confidence=float(d.get("confidence", 0.0)),
            )
            for d in data.get("relevant_dimensions", [])
        ]

        return Annotation(
            conversation_id=conversation_id,
            num_turns=0,  # filled in by annotate()
            is_culturally_relevant=bool(data.get("is_culturally_relevant", False)),
            relevant_dimensions=dimension_matches,
            reasoning=data.get("reasoning", ""),
            metadata={},
        )
