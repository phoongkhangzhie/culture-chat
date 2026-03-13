"""
Prompt templates for the WildChat cultural annotation pipeline.
"""

from __future__ import annotations

from src.models import CultureDimension, Turn

# ---------------------------------------------------------------------------
# Taxonomy summary injected into the system prompt
# ---------------------------------------------------------------------------

TAXONOMY_OVERVIEW = """\
You are an expert annotator trained in cross-cultural communication and the \
CultureScope taxonomy. The taxonomy has three layers:

  Layer 1 — Institutional Norms
    • Geography & Customs (population, geography, dates of significance)
    • Regulation & Policy (transportation rules, data formats, measurement
      units, financial market rules)

  Layer 2 — Behavioral Patterns
    • Personal Choices & Habits: daily life & travel, diet & health, education
      & knowledge, art & entertainment, personal etiquette, etiquette &
      courtesy, communication style, fixed expressions in language

  Layer 3 — Core Values
    • Social Relationship & Structures: family dynamics, household structures,
      gender roles
    • Values & Beliefs: cultural values, religion, do's & don'ts
"""

SYSTEM_PROMPT = (
    TAXONOMY_OVERVIEW
    + """
Your task: given a multi-turn conversation, decide whether it is
**culturally-relevant** or **culturally-neutral**.

Definitions
-----------
• Culturally-neutral  — The conversation could have taken place in any culture
  without modification. Generic technical questions, universal topics, or \
abstract discussions belong here.

• Culturally-relevant — The conversation contains content that is specific to,
  shaped by, or meaningfully reflects the norms, practices, values, or
  expressions of one or more cultures. This includes explicit cultural
  references AND implicit signals (e.g. writing norms for a local holiday,
  asking about a regional dish, referencing a culture-specific etiquette rule,
  using culturally-loaded idioms).

When annotating cultural relevance, identify the fine-grained CultureScope
dimensions that apply and quote the exact text spans (indicators) that support
each dimension.

Respond ONLY with valid JSON matching the schema provided in the user message.
"""
)


def build_annotation_prompt(
    turns: list[Turn],
    dimensions: dict[str, CultureDimension],
) -> str:
    """
    Build the user-turn prompt sent to the model.

    Args:
        turns:      The conversation turns to annotate.
        dimensions: The full CULTURAL_DIMENSIONS dict from taxonomy.py.
    """
    conversation_text = "\n".join(
        f"[{t.role.upper()}]: {t.content}" for t in turns
    )

    dimension_list = "\n".join(
        f'  "{key}": {dim.name} — {dim.description}'
        for key, dim in dimensions.items()
    )

    return f"""\
## Conversation to annotate

{conversation_text}

---

## Available CultureScope dimensions

{dimension_list}

---

## Required JSON output

Return a single JSON object with this exact structure:

{{
  "is_culturally_relevant": <true | false>,
  "relevant_dimensions": [
    {{
      "dimension_key": "<key from the list above>",
      "dimension_name": "<name>",
      "indicators": ["<verbatim or near-verbatim text span from conversation>"],
      "confidence": <0.0 – 1.0>
    }}
  ],
  "reasoning": "<1–3 sentence explanation of your decision>"
}}

Rules:
- If is_culturally_relevant is false, set relevant_dimensions to [].
- Only include dimensions with clear evidence from the conversation text.
- indicators must be short, specific spans (not the whole conversation).
- confidence reflects how strongly the conversation signals that dimension.
"""
