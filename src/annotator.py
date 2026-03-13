"""
Cultural annotation pipeline for WildChat conversations.

Supports two backends:
  - "anthropic"  — Anthropic API (Claude models)
  - "vllm"       — local vLLM server via OpenAI-compatible API

Features:
  - Throttling: enforces a minimum interval between requests (default 5 RPM).
  - Resumable:  annotate_batch() writes each result immediately and skips
                conversations already present in the output file, so a
                stopped run can be resumed by re-running the same command.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from src.models import Annotation, Conversation, DimensionMatch
from src.prompts import SYSTEM_PROMPT, build_annotation_prompt
from src.taxonomy import CULTURAL_DIMENSIONS

logger = logging.getLogger(__name__)


class Annotator:
    """
    Annotates WildChat conversations using either the Anthropic API or a
    local vLLM server.

    Usage::

        # Anthropic (default)
        annotator = Annotator(backend="anthropic", model="claude-sonnet-4-6")

        # Local vLLM
        annotator = Annotator(
            backend="vllm",
            model="Qwen/Qwen2.5-7B-Instruct",
            base_url="http://localhost:8000/v1",
        )

        annotator.annotate_batch(conversations, output_path=Path("output/annotations.jsonl"))
    """

    def __init__(
        self,
        backend: str = "anthropic",         # "anthropic" | "vllm"
        model: str = "claude-sonnet-4-6",
        base_url: str = "http://localhost:8000/v1",  # used only for vllm
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        requests_per_minute: int = 5,
    ) -> None:
        if backend not in ("anthropic", "vllm"):
            raise ValueError(f"backend must be 'anthropic' or 'vllm', got {backend!r}")

        self.backend = backend
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.min_interval = 60.0 / requests_per_minute
        self._last_request_time: float = 0.0

        if backend == "anthropic":
            import anthropic as _anthropic
            self._anthropic_client = _anthropic.Anthropic()
            self._openai_client = None
        else:
            from openai import OpenAI
            self._openai_client = OpenAI(base_url=base_url, api_key="EMPTY")
            self._anthropic_client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(self, conversation: Conversation) -> Annotation:
        """Annotate a single conversation. Throttles to respect RPM limit."""
        self._throttle()
        user_prompt = build_annotation_prompt(
            conversation.turns, CULTURAL_DIMENSIONS
        )
        raw = self._call_with_retry(user_prompt)
        parsed = self._parse_response(raw, conversation.conversation_id)
        parsed.num_turns = len(conversation.turns)
        parsed.metadata["model"] = self.model
        parsed.metadata["backend"] = self.backend
        parsed.metadata["annotated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if conversation.country:
            parsed.metadata["country"] = conversation.country
        if conversation.language:
            parsed.metadata["language"] = conversation.language
        return parsed

    def annotate_batch(
        self,
        conversations: list[Conversation],
        output_path: Path,
        on_error: str = "skip",  # "skip" | "raise"
    ) -> list[Annotation]:
        """
        Annotate conversations sequentially, writing each result to output_path
        immediately (JSONL, one annotation per line).

        Resumable: if output_path already exists, conversations whose IDs are
        already in the file are skipped. Re-run the same command to continue
        from where it left off.

        Args:
            conversations: Conversations to annotate.
            output_path:   JSONL file to write/append annotations to.
            on_error:      "skip" logs a warning and continues; "raise" re-raises.
        """
        done_ids: set[str] = set()
        if output_path.exists():
            with open(output_path) as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line)["conversation_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
            if done_ids:
                logger.info("Resuming — %d conversations already annotated, skipping.", len(done_ids))

        remaining = [c for c in conversations if c.conversation_id not in done_ids]
        total = len(conversations)
        skipped = len(done_ids)

        results: list[Annotation] = []
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a") as f:
            for i, conv in enumerate(remaining):
                logger.info(
                    "Annotating %d/%d (skipped %d)  id=%s",
                    skipped + i + 1,
                    total,
                    skipped,
                    conv.conversation_id,
                )
                try:
                    ann = self.annotate(conv)
                    f.write(ann.model_dump_json() + "\n")
                    f.flush()
                    results.append(ann)
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

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        wait = self.min_interval - elapsed
        if wait > 0:
            logger.debug("Throttling: sleeping %.1fs to respect RPM limit.", wait)
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    def _call_with_retry(self, user_prompt: str) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.backend == "anthropic":
                    return self._call_anthropic(user_prompt)
                else:
                    return self._call_vllm(user_prompt)
            except Exception as exc:
                # Anthropic rate-limit gets a longer back-off
                import anthropic as _anthropic
                if isinstance(exc, _anthropic.RateLimitError):
                    wait = self.retry_delay * attempt
                    logger.warning("Rate limited (attempt %d/%d). Retrying in %.0fs.", attempt, self.max_retries, wait)
                    time.sleep(wait)
                else:
                    logger.warning("API error (attempt %d/%d): %s", attempt, self.max_retries, exc)
                    time.sleep(self.retry_delay)
                last_exc = exc
        raise RuntimeError(f"All {self.max_retries} attempts failed") from last_exc

    def _call_anthropic(self, user_prompt: str) -> str:
        message = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    def _call_vllm(self, user_prompt: str) -> str:
        response = self._openai_client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def _parse_response(self, raw: str, conversation_id: str) -> Annotation:
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
