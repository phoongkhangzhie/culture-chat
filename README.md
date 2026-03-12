# culture-chat

A pipeline for annotating [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) conversations from a cultural perspective.

Given the [CultureScope](https://arxiv.org/abs/2402.10946) taxonomy, each conversation (≥ 6 turns) is classified as:

- **Culturally-neutral** — the conversation could occur in any culture without modification (generic technical questions, abstract discussions, universal topics).
- **Culturally-relevant** — the conversation reflects norms, practices, values, or expressions specific to one or more cultures, with the exact text spans (*indicators*) that signal each matched dimension.

---

## Repository Structure

```
culture-chat/
├── main.py          # CLI: annotate / sample / stats
├── config.py        # Shared defaults (model, token budget, min turns)
├── pyproject.toml
└── src/
    ├── models.py    # Pydantic types: Conversation, Annotation, DimensionMatch
    ├── taxonomy.py  # Full CultureScope taxonomy (~130 fine-grained dimensions)
    ├── loader.py    # Streams WildChat-1M from HuggingFace with turn-count filter
    ├── prompts.py   # System + user prompt templates sent to the model
    └── annotator.py # Anthropic API calls, JSON parsing, retry logic
```

---

## Setup

```bash
# Clone
git clone https://github.com/phoongkhangzhie/culture-chat.git
cd culture-chat

# Install dependencies (Python >=3.12)
pip install -e .

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-...
```

---

## Annotation Process — Step by Step

### Step 1 — Sample conversations from WildChat

WildChat-1M is a large dataset; it is faster to save a local sample first before annotating.

```bash
python main.py sample \
  --n 500 \
  --min-turns 6 \
  --language English \
  --output output/sample.jsonl
```

| Flag | Default | Description |
|---|---|---|
| `--n` | 100 | Number of conversations to save |
| `--min-turns` | 6 | Minimum combined user + assistant turns |
| `--language` | *(all)* | Filter by WildChat's `language` field (e.g. `English`, `Chinese`) |
| `--output` | required | Output JSONL path |

Each line of the output JSONL is a serialised `Conversation` object:

```json
{
  "conversation_id": "abc123",
  "turns": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "language": "English",
  "country": "US",
  "metadata": {"model": "gpt-4", "toxic": false}
}
```

---

### Step 2 — Annotate

```bash
python main.py annotate \
  --input output/sample.jsonl \
  --output output/annotations.jsonl \
  --model claude-opus-4-6
```

You can also skip Step 1 and stream directly from HuggingFace:

```bash
python main.py annotate \
  --output output/annotations.jsonl \
  --min-turns 6 \
  --max-conversations 200 \
  --language English
```

| Flag | Default | Description |
|---|---|---|
| `--input` | *(stream)* | Local JSONL from `sample`; omit to stream from HuggingFace |
| `--output` | required | Annotation output JSONL |
| `--model` | `claude-opus-4-6` | Anthropic model |
| `--max-tokens` | 2048 | Max tokens in the model response |
| `--min-turns` | 6 | Only used when streaming (ignored if `--input` is set) |
| `--max-conversations` | *(all)* | Hard cap on conversations to annotate |
| `--language` | *(all)* | Language filter when streaming |

---

### Step 3 — Inspect statistics

```bash
python main.py stats --input output/annotations.jsonl
```

Example output:

```
==================================================
  Total annotated:       500
  Culturally relevant:   312  (62.4%)
  Culturally neutral:    188  (37.6%)

  Top 10 dimensions:
    guest_hospitality                             47
    eating_habits                                 38
    religious_holidays                            31
    verbal_communication_style                    28
    collectivism                                  22
    ...
==================================================
```

---

## Annotation Pipeline — How It Works

```
Conversation (≥ 6 turns)
        │
        ▼
┌─────────────────────────────────────────────┐
│               Annotator                     │
│                                             │
│  1. build_annotation_prompt()               │
│     • formats the conversation as           │
│       [USER] / [ASSISTANT] turns            │
│     • appends all ~130 CultureScope         │
│       dimensions with descriptions          │
│     • specifies required JSON schema        │
│                                             │
│  2. _call_with_retry()                      │
│     • sends system + user prompt to         │
│       Anthropic API                         │
│     • retries on rate-limit / API errors    │
│                                             │
│  3. _parse_response()                       │
│     • parses model JSON                     │
│     • validates into Annotation object      │
└─────────────────────────────────────────────┘
        │
        ▼
  Annotation (JSONL line)
```

### What the model sees

**System prompt** — sets the annotator role, explains the three-layer CultureScope taxonomy, and defines the distinction between culturally-neutral and culturally-relevant.

**User prompt** — contains:
1. The full conversation formatted as `[USER]` / `[ASSISTANT]` turns
2. All ~130 CultureScope dimension keys with their names and descriptions
3. The required JSON output schema

### What the model returns

```json
{
  "is_culturally_relevant": true,
  "relevant_dimensions": [
    {
      "dimension_key": "shoe_etiquette",
      "dimension_name": "Shoe Etiquette During a Visit",
      "category": "Behavioral Patterns > Personal Choices & Habits > Personal Etiquette",
      "indicators": [
        "Should I take off my shoes before entering?",
        "In Japan it is customary to remove shoes at the genkan"
      ],
      "confidence": 0.95
    }
  ],
  "reasoning": "The conversation explicitly discusses the Japanese custom of removing shoes at the entrance, a well-defined cultural etiquette norm."
}
```

### Output schema

| Field | Type | Description |
|---|---|---|
| `conversation_id` | str | WildChat conversation identifier |
| `num_turns` | int | Number of turns in the conversation |
| `is_culturally_relevant` | bool | True if any cultural dimension was detected |
| `relevant_dimensions` | list | One entry per matched CultureScope dimension |
| `relevant_dimensions[].dimension_key` | str | Key from the CultureScope taxonomy |
| `relevant_dimensions[].dimension_name` | str | Human-readable dimension name |
| `relevant_dimensions[].category` | str | `Layer > Category > Topic Aspect` path |
| `relevant_dimensions[].indicators` | list[str] | Verbatim text spans from the conversation |
| `relevant_dimensions[].confidence` | float | 0–1 annotator confidence |
| `reasoning` | str | 1–3 sentence justification |
| `metadata.model` | str | Anthropic model used |
| `metadata.annotated_at` | str | ISO 8601 UTC timestamp |
| `metadata.country` | str | WildChat `country` field (if available) |
| `metadata.language` | str | WildChat `language` field (if available) |

---

## CultureScope Taxonomy

The taxonomy has three layers, seven categories, and ~130 fine-grained dimensions:

| Layer | Categories |
|---|---|
| **1 — Institutional Norms** | Geography & Customs, Regulation & Policy |
| **2 — Behavioral Patterns** | Daily Life & Travel, Diet & Health, Education & Knowledge, Art & Entertainment, Personal Etiquette, Etiquette & Courtesy, Communication Style, Fixed Expressions |
| **3 — Core Values** | Family Dynamics, Household Structures, Gender Roles, Cultural Values, Religion, Do's & Don'ts |

See [`src/taxonomy.py`](src/taxonomy.py) for the full list of dimension keys and descriptions.

---

## Configuration

Edit [`config.py`](config.py) or set environment variables:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required — your Anthropic API key |
| `CULTURE_CHAT_MODEL` | `claude-opus-4-6` | Model used for annotation |
| `CULTURE_CHAT_MAX_TOKENS` | `2048` | Max tokens in annotation response |
| `CULTURE_CHAT_MIN_TURNS` | `6` | Minimum conversation turns |
| `CULTURE_CHAT_OUTPUT_DIR` | `./output` | Default output directory |
