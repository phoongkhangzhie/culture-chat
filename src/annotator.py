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

from src.models import (
    Annotation, Conversation, DimensionMatch,
    EmergentPattern, FailureAnalysis, FailureMode,
    FailureObservation, OpenAnalysis, SynthesisReport,
)
from src.prompts import (
    FAILURE_ANALYSIS_SYSTEM_PROMPT,
    MERGE_SYSTEM_PROMPT,
    OPEN_ANALYSIS_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_annotation_prompt,
    build_failure_analysis_prompt,
    build_merge_prompt,
    build_open_analysis_prompt,
    build_synthesis_prompt,
)
from src.taxonomy import CULTURAL_DIMENSIONS

logger = logging.getLogger(__name__)


class Annotator:
    """
    Annotates WildChat conversations using either the Anthropic API or a
    local vLLM server.

    Usage::

        # Anthropic (default)
        annotator = Annotator(backend="anthropic", model="claude-sonnet-4-6")

        # OpenAI
        annotator = Annotator(backend="openai", model="gpt-4o")

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
        backend: str = "anthropic",         # "anthropic" | "openai" | "vllm"
        model: str = "claude-sonnet-4-6",
        base_url: str = "http://localhost:8000/v1",  # used only for vllm
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        requests_per_minute: int = 5,
    ) -> None:
        if backend not in ("anthropic", "openai", "vllm"):
            raise ValueError(f"backend must be 'anthropic', 'openai', or 'vllm', got {backend!r}")

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
        elif backend == "openai":
            from openai import OpenAI
            self._openai_client = OpenAI()  # reads OPENAI_API_KEY from env
            self._anthropic_client = None
        else:  # vllm
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
        on_error: str = "skip",  # "skip" | "raise",
        max_annotations: int = None
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
                if max_annotations and len(results) >= max_annotations:
                    print(f"Annotated {max_annotations} conversations.")
                    break

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
    # Failure analysis (culturally-relevant conversations only)
    # ------------------------------------------------------------------

    def analyse_failures(
        self,
        conversation: Conversation,
        annotation: Annotation,
    ) -> FailureAnalysis:
        """
        Analyse a single culturally-relevant conversation for assistant failure modes.
        Only call this on conversations where annotation.is_culturally_relevant is True.
        """
        self._throttle()
        user_prompt = build_failure_analysis_prompt(conversation.turns, annotation)
        raw = self._call_with_retry(user_prompt, system=FAILURE_ANALYSIS_SYSTEM_PROMPT)
        result = self._parse_failure_response(raw, conversation.conversation_id)
        result.num_turns = len(conversation.turns)
        result.metadata["model"] = self.model
        result.metadata["backend"] = self.backend
        result.metadata["analysed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if conversation.country:
            result.metadata["country"] = conversation.country
        if conversation.language:
            result.metadata["language"] = conversation.language
        return result

    def analyse_failures_batch(
        self,
        conversations: list[Conversation],
        annotations: list[Annotation],
        output_path: Path,
        on_error: str = "skip",
    ) -> list[FailureAnalysis]:
        """
        Run failure analysis on a list of culturally-relevant conversations.
        Skips conversations that already appear in output_path (resumable).

        Args:
            conversations: Conversation objects (must match annotations by position).
            annotations:   Corresponding Annotation objects.
            output_path:   JSONL file to write/append FailureAnalysis results to.
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
                logger.info("Resuming — %d conversations already analysed, skipping.", len(done_ids))

        pairs = [
            (conv, ann)
            for conv, ann in zip(conversations, annotations)
            if conv.conversation_id not in done_ids and ann.is_culturally_relevant
        ]
        skipped_done  = len(done_ids)
        skipped_neutral = sum(1 for _, ann in zip(conversations, annotations)
                              if not ann.is_culturally_relevant)
        total = len(conversations)

        logger.info(
            "Failure analysis: %d to analyse, %d already done, %d culturally-neutral (skipped).",
            len(pairs), skipped_done, skipped_neutral,
        )

        results: list[FailureAnalysis] = []
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a") as f:
            for i, (conv, ann) in enumerate(pairs):
                logger.info(
                    "Analysing %d/%d  id=%s",
                    skipped_done + i + 1,
                    total - skipped_neutral,
                    conv.conversation_id,
                )
                try:
                    fa = self.analyse_failures(conv, ann)
                    f.write(fa.model_dump_json() + "\n")
                    f.flush()
                    results.append(fa)
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
    # Open-ended analysis (Option B, Pass 1)
    # ------------------------------------------------------------------

    def open_analyse(
        self,
        conversation: Conversation,
        annotation: Annotation,
        max_tokens: Optional[int] = None,
    ) -> OpenAnalysis:
        """Open-ended cultural failure analysis — no predefined failure taxonomy."""
        self._throttle()
        user_prompt = build_open_analysis_prompt(conversation.turns, annotation)
        raw = self._call_with_retry(user_prompt, system=OPEN_ANALYSIS_SYSTEM_PROMPT, max_tokens=max_tokens)
        result = self._parse_open_analysis(raw, conversation.conversation_id)
        result.num_turns = len(conversation.turns)
        result.metadata["model"] = self.model
        result.metadata["backend"] = self.backend
        result.metadata["analysed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if conversation.country:
            result.metadata["country"] = conversation.country
        if conversation.language:
            result.metadata["language"] = conversation.language
        return result

    def open_analyse_batch(
        self,
        conversations: list[Conversation],
        annotations: list[Annotation],
        output_path: Path,
        on_error: str = "skip",
        max_tokens: Optional[int] = None,
    ) -> list[OpenAnalysis]:
        """
        Run open-ended analysis on culturally-relevant conversations.
        Resumable — skips conversation IDs already present in output_path.
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
                logger.info("Resuming — %d conversations already analysed, skipping.", len(done_ids))

        pairs = [
            (conv, ann)
            for conv, ann in zip(conversations, annotations)
            if conv.conversation_id not in done_ids and ann.is_culturally_relevant
        ]
        skipped_neutral = sum(1 for _, ann in zip(conversations, annotations)
                              if not ann.is_culturally_relevant)
        logger.info(
            "Open analysis: %d to analyse, %d already done, %d neutral (skipped).",
            len(pairs), len(done_ids), skipped_neutral,
        )

        results: list[OpenAnalysis] = []
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "a") as f:
            for i, (conv, ann) in enumerate(pairs):
                logger.info("Open-analysing %d/%d  id=%s", len(done_ids) + i + 1,
                            len(conversations) - skipped_neutral, conv.conversation_id)
                try:
                    oa = self.open_analyse(conv, ann, max_tokens=max_tokens)
                    f.write(oa.model_dump_json() + "\n")
                    f.flush()
                    results.append(oa)
                except Exception as exc:
                    if on_error == "raise":
                        raise
                    logger.warning("Skipping %s: %s", conv.conversation_id, exc)

        return results

    # ------------------------------------------------------------------
    # Synthesis (Option B, Pass 2)
    # ------------------------------------------------------------------

    def synthesise(
        self,
        open_analyses: list[OpenAnalysis],
        output_path: Path,
        max_tokens: Optional[int] = None,
        batch_size: int = 40,
    ) -> SynthesisReport:
        """
        Synthesise emergent failure patterns across all open analyses.

        If the number of analyses exceeds batch_size, a map-reduce approach is
        used: each batch is synthesised independently, then the intermediate
        reports are merged in a single follow-up call.

        Intermediate batch reports are saved alongside output_path as
        <stem>_batch_0.json, <stem>_batch_1.json, … so the merge step can be
        re-run without repeating the per-batch calls.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        analyses_with_issues = [oa for oa in open_analyses if oa.has_issues]

        if len(analyses_with_issues) <= batch_size:
            # Single-pass — fits in one prompt
            logger.info("Synthesis: single pass (%d analyses).", len(analyses_with_issues))
            return self._synthesise_chunk(open_analyses, output_path, max_tokens)

        # Map phase — synthesise each batch
        chunks = [
            analyses_with_issues[i: i + batch_size]
            for i in range(0, len(analyses_with_issues), batch_size)
        ]
        logger.info(
            "Synthesis: %d analyses → %d batches of ≤%d.",
            len(analyses_with_issues), len(chunks), batch_size,
        )
        stem = output_path.stem
        intermediate: list[SynthesisReport] = []
        for i, chunk in enumerate(chunks):
            batch_path = output_path.parent / f"{stem}_batch_{i}.json"
            if batch_path.exists():
                logger.info("Batch %d already synthesised, loading from %s.", i, batch_path)
                intermediate.append(SynthesisReport.model_validate_json(batch_path.read_text()))
            else:
                logger.info("Synthesising batch %d/%d (%d analyses)…", i + 1, len(chunks), len(chunk))
                report = self._synthesise_chunk(chunk, batch_path, max_tokens)
                intermediate.append(report)

        # Reduce phase — merge intermediate reports
        logger.info("Merging %d intermediate reports…", len(intermediate))
        self._throttle()
        merge_prompt = build_merge_prompt(intermediate)
        raw = self._call_with_retry(merge_prompt, system=MERGE_SYSTEM_PROMPT, max_tokens=max_tokens)
        merged = self._parse_synthesis(raw, open_analyses)
        merged.metadata["model"] = self.model
        merged.metadata["backend"] = self.backend
        merged.metadata["synthesised_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        merged.metadata["batches"] = len(chunks)
        output_path.write_text(merged.model_dump_json(indent=2))
        return merged

    def _synthesise_chunk(
        self,
        open_analyses: list[OpenAnalysis],
        output_path: Path,
        max_tokens: Optional[int] = None,
    ) -> SynthesisReport:
        """Run synthesis on a single chunk and save to output_path."""
        self._throttle()
        user_prompt = build_synthesis_prompt(open_analyses)
        raw = self._call_with_retry(user_prompt, system=SYNTHESIS_SYSTEM_PROMPT, max_tokens=max_tokens)
        report = self._parse_synthesis(raw, open_analyses)
        report.metadata["model"] = self.model
        report.metadata["backend"] = self.backend
        report.metadata["synthesised_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.model_dump_json(indent=2))
        return report

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

    def _call_with_retry(self, user_prompt: str, system: str = SYSTEM_PROMPT, max_tokens: Optional[int] = None) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.backend == "anthropic":
                    return self._call_anthropic(user_prompt, system=system, max_tokens=max_tokens)
                else:
                    return self._call_openai_compatible(user_prompt, system=system, max_tokens=max_tokens)
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

    def _call_anthropic(
        self,
        user_prompt: str,
        system: str = SYSTEM_PROMPT,
        max_tokens: int | None = None,
    ) -> str:
        message = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    def _call_openai_compatible(
        self,
        user_prompt: str,
        system: str = SYSTEM_PROMPT,
        max_tokens: int | None = None,
    ) -> str:
        """Shared call for both OpenAI API and vLLM (OpenAI-compatible)."""
        tokens = max_tokens or self.max_tokens
        # OpenAI API requires max_completion_tokens; vLLM uses max_tokens
        token_kwarg = (
            {"max_completion_tokens": tokens}
            if self.backend == "openai"
            else {"max_tokens": tokens}
        )
        response = self._openai_client.chat.completions.create(
            model=self.model,
            **token_kwarg,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
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

    def _parse_failure_response(self, raw: str, conversation_id: str) -> FailureAnalysis:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Model returned invalid JSON for failure analysis of {conversation_id}:\n{raw}"
            ) from exc

        failures = [
            FailureMode(
                failure_type=f.get("failure_type", "unknown"),
                severity=f.get("severity", "low"),
                turn_indices=f.get("turn_indices", []),
                evidence=f.get("evidence", []),
                description=f.get("description", ""),
                recommended_response=f.get("recommended_response", ""),
            )
            for f in data.get("failures", [])
        ]

        return FailureAnalysis(
            conversation_id=conversation_id,
            num_turns=0,  # filled in by analyse_failures()
            has_failures=bool(data.get("has_failures", False)),
            failures=failures,
            overall_severity=data.get("overall_severity") or None,
            summary=data.get("summary", ""),
            metadata={},
        )

    def _parse_open_analysis(self, raw: str, conversation_id: str) -> OpenAnalysis:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            # Likely truncated at max_tokens — log the tail so the user can diagnose
            tail = raw[-200:] if len(raw) > 200 else raw
            logger.error(
                "JSON truncated for conversation %s (response length %d chars). "
                "Try --max-tokens 8192. Last 200 chars: ...%s",
                conversation_id, len(raw), tail,
            )
            raise ValueError(
                f"Model returned invalid JSON for open analysis of {conversation_id} "
                f"(likely truncated at max_tokens — try --max-tokens 8192)"
            ) from exc

        observations = [
            FailureObservation(
                turn_indices=o.get("turn_indices", []),
                evidence=o.get("evidence", []),
                observation=o.get("observation", ""),
                severity=o.get("severity", "low"),
                recommended_response=o.get("recommended_response", ""),
            )
            for o in data.get("observations", [])
        ]
        return OpenAnalysis(
            conversation_id=conversation_id,
            num_turns=0,
            has_issues=bool(data.get("has_issues", False)),
            observations=observations,
            overall_severity=data.get("overall_severity") or None,
            summary=data.get("summary", ""),
            metadata={},
        )

    def _parse_synthesis(self, raw: str, open_analyses: list[OpenAnalysis]) -> SynthesisReport:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            data: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            tail = raw[-300:] if len(raw) > 300 else raw
            logger.error(
                "Synthesis JSON truncated (response length %d chars). "
                "Try --max-tokens 8192 or higher. Last 300 chars: ...%s",
                len(raw), tail,
            )
            raise ValueError(
                f"Model returned invalid JSON for synthesis "
                f"(likely truncated at max_tokens — try --max-tokens 8192)"
            ) from exc

        patterns = [
            EmergentPattern(
                name=p.get("name", ""),
                description=p.get("description", ""),
                frequency=int(p.get("frequency", 0)),
                severity_distribution=p.get("severity_distribution", {}),
                example_conversation_ids=p.get("example_conversation_ids", []),
                example_evidence=p.get("example_evidence", []),
            )
            for p in data.get("patterns", [])
        ]
        total = len(open_analyses)
        with_issues = sum(1 for oa in open_analyses if oa.has_issues)
        total_obs = sum(len(oa.observations) for oa in open_analyses)
        return SynthesisReport(
            total_conversations=total,
            total_with_issues=with_issues,
            total_observations=total_obs,
            patterns=patterns,
            uncategorised_observations=data.get("uncategorised_observations", []),
            synthesis_summary=data.get("synthesis_summary", ""),
            metadata={},
        )
