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
