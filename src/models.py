from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared taxonomy type (re-exported here so annotator imports stay local)
# ---------------------------------------------------------------------------

class CultureDimension(BaseModel):
    name: str
    category: str
    description: str
    keywords: list[str] = []


# ---------------------------------------------------------------------------
# WildChat conversation types
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    role: str          # "user" or "assistant"
    content: str


class Conversation(BaseModel):
    conversation_id: str
    turns: list[Turn]
    language: Optional[str] = None
    country: Optional[str] = None   # country field in WildChat if available
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Annotation output types
# ---------------------------------------------------------------------------

class DimensionMatch(BaseModel):
    """A single cultural dimension that was detected in the conversation."""

    dimension_key: str
    dimension_name: str
    indicators: list[str] = Field(
        description="Verbatim or near-verbatim text spans from the conversation that signal this dimension"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Annotator confidence that this dimension applies (0–1)"
    )


class Annotation(BaseModel):
    """Full cultural annotation for one conversation."""

    conversation_id: str
    num_turns: int
    is_culturally_relevant: bool = Field(
        description="True if the conversation contains culturally-specific content; False if culturally-neutral"
    )
    relevant_dimensions: list[DimensionMatch] = Field(
        default_factory=list,
        description="Populated only when is_culturally_relevant is True"
    )
    reasoning: str = Field(
        description="Brief explanation of the annotation decision"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnnotationBatch(BaseModel):
    """Container for a batch of annotations (one output file)."""

    annotations: list[Annotation]
    batch_metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Failure analysis output types
# ---------------------------------------------------------------------------

class FailureMode(BaseModel):
    """A single assistant failure instance detected in a conversation."""

    failure_type: str = Field(
        description=(
            "Category of failure. One of: cultural_ignorance, western_centric_default, "
            "stereotyping, inappropriate_advice, idiom_misinterpretation, cultural_dismissal, "
            "harmful_content, incorrect_cultural_information, language_failure, "
            "regulatory_legal_mismatch, refusal_overcorrection"
        )
    )
    severity: str = Field(
        description="Severity level: low | medium | high | critical"
    )
    turn_indices: list[int] = Field(
        description="0-based indices of the assistant turn(s) where the failure occurs"
    )
    evidence: list[str] = Field(
        description="Verbatim or near-verbatim text spans from the assistant turn(s) that demonstrate the failure"
    )
    description: str = Field(
        description="Specific explanation of why this response is a failure given the cultural context"
    )
    recommended_response: str = Field(
        description="Brief description of how a culturally-aware assistant should have responded instead"
    )


class FailureAnalysis(BaseModel):
    """Full failure analysis for one culturally-relevant conversation."""

    conversation_id: str
    num_turns: int
    has_failures: bool
    failures: list[FailureMode] = Field(default_factory=list)
    overall_severity: Optional[str] = Field(
        default=None,
        description="Worst severity across all failures: low | medium | high | critical | None"
    )
    summary: str = Field(
        description="1–3 sentence summary of the assistant's overall cultural competence in this conversation"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Open-ended (self-discovery) failure analysis types  — Option B, Pass 1
# ---------------------------------------------------------------------------

class FailureObservation(BaseModel):
    """A single open-ended observation of an assistant failure — no predefined taxonomy."""

    turn_indices: list[int] = Field(
        description="0-based indices of the assistant turn(s) where the issue occurs"
    )
    evidence: list[str] = Field(
        description="Verbatim or near-verbatim text spans from the assistant turn(s)"
    )
    observation: str = Field(
        description="Free-text description of what went wrong and why it matters culturally"
    )
    severity: str = Field(
        description="Estimated severity: low | medium | high | critical"
    )
    recommended_response: str = Field(
        description="What a culturally-aware assistant should have done instead"
    )


class OpenAnalysis(BaseModel):
    """Open-ended failure analysis for one culturally-relevant conversation (Pass 1)."""

    conversation_id: str
    num_turns: int
    has_issues: bool
    observations: list[FailureObservation] = Field(default_factory=list)
    overall_severity: Optional[str] = Field(default=None)
    summary: str = Field(
        description="1–3 sentence assessment of the assistant's cultural competence"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Synthesis types — Option B, Pass 2
# ---------------------------------------------------------------------------

class EmergentPattern(BaseModel):
    """A failure pattern identified inductively across many conversations."""

    name: str = Field(description="Short snake_case name for the pattern")
    description: str = Field(description="What this pattern is and why it represents a failure")
    frequency: int = Field(description="Number of conversations where this pattern appears")
    severity_distribution: dict[str, int] = Field(
        description="Count of each severity level: {'low': N, 'medium': N, 'high': N, 'critical': N}"
    )
    example_conversation_ids: list[str] = Field(
        description="Up to 3 conversation IDs that best illustrate this pattern"
    )
    example_evidence: list[str] = Field(
        description="Representative verbatim evidence spans for this pattern"
    )


class SynthesisReport(BaseModel):
    """Emergent taxonomy synthesised from open-ended failure observations (Pass 2)."""

    total_conversations: int
    total_with_issues: int
    total_observations: int
    patterns: list[EmergentPattern]
    uncategorised_observations: list[str] = Field(
        default_factory=list,
        description="Observations that did not fit any identified pattern"
    )
    synthesis_summary: str = Field(
        description="3–5 sentence narrative of the key findings across all conversations"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fine-grained cultural failure tags  (BigSpin-compatible format)
# ---------------------------------------------------------------------------

class CulturalFailureTag(BaseModel):
    """
    A single cultural failure at the turn level.

    Mirrors BigSpin's per-signal schema {severity, evidence, turn, notes} but
    enriched with cultural dimension information and a free-text failure type
    discovered inductively from the conversation.
    """

    failure_type: str = Field(
        description=(
            "Name for this failure.  Use the closest matching synthesis pattern name "
            "if one was provided; otherwise coin a concise snake_case label (2–5 words)."
        )
    )
    cultural_failure_severity: int = Field(
        ge=1, le=4,
        description="1=low, 2=medium, 3=high, 4=critical  (matches BigSpin severity scale)"
    )
    evidence: str = Field(
        description="Verbatim or near-verbatim text span from the assistant turn"
    )
    turn: int = Field(
        description="0-based index of the assistant turn where the failure occurs"
    )
    notes: str = Field(
        description="Specific explanation of why this is a cultural failure in this context"
    )
    dimension_key: Optional[str] = Field(
        default=None,
        description="CultureScope dimension key most relevant to this failure (if applicable)"
    )
    dimension_name: Optional[str] = Field(
        default=None,
        description="Human-readable name of the matched CultureScope dimension"
    )
    recommended_response: str = Field(
        description="What a culturally-aware assistant should have said instead"
    )


class CulturalFailureAnnotation(BaseModel):
    """
    Full cultural failure annotation for one conversation.

    Combines cultural relevance annotation, open-ended observations, and
    fine-grained per-turn tags in BigSpin-compatible format.  Intended as
    the final output of the culture-chat × BigSpin intersection pipeline.
    """

    conversation_id: str
    num_turns: int

    # Cultural relevance layer
    is_culturally_relevant: bool
    cultural_dimensions: list[DimensionMatch] = Field(default_factory=list)
    annotation_reasoning: str = ""

    # Fine-grained failure tags (BigSpin-compatible)
    cultural_failures: list[CulturalFailureTag] = Field(default_factory=list)
    has_cultural_failures: bool = False
    overall_cultural_severity: Optional[int] = Field(
        default=None,
        description="Worst cultural_failure_severity across all tags (1–4), or None"
    )
    cultural_summary: str = Field(
        description="1–3 sentence summary of the assistant's cultural competence"
    )

    # BigSpin context (preserved from source file)
    bigspin_quality: Optional[str] = None
    bigspin_signals: list[str] = Field(default_factory=list)
    bigspin_primary_failure: Optional[str] = None

    metadata: dict[str, Any] = Field(default_factory=dict)
