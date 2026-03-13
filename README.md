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
    ├── loader.py    # Loads WildChat-1M from HuggingFace with turn-count filter
    ├── prompts.py   # System + user prompt templates sent to the model
    └── annotator.py # API calls (Anthropic / OpenAI / vLLM), JSON parsing, retry logic
```

---

## Setup

```bash
# Clone
git clone https://github.com/phoongkhangzhie/culture-chat.git
cd culture-chat

# Install dependencies with uv (Python >=3.12)
uv sync

# Or with pip
pip install -e .

# Add your API key(s) to a .env file
echo "ANTHROPIC_API_KEY=sk-..." > .env
echo "OPENAI_API_KEY=sk-..."   >> .env
```

---

## Annotation Process — Step by Step

### Step 1 — Sample conversations from WildChat

WildChat-1M is a large dataset; saving a local sample first is faster than streaming during annotation.

```bash
python main.py sample \
  --n 500 \
  --min-turns 6 \
  --language English \
  --output output/sample.jsonl
```

| Flag | Default | Description |
|---|---|---|
| `--n` | *(all)* | Number of conversations to save |
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
  "country": "United States",
  "metadata": {}
}
```

---

### Step 2 — Annotate

```bash
python main.py annotate \
  --input  output/sample.jsonl \
  --output output/annotations.jsonl \
  --backend anthropic \
  --model   claude-sonnet-4-6
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
| `--backend` | `anthropic` | `anthropic`, `openai`, or `vllm` |
| `--model` | `claude-sonnet-4-6` | Model ID for the chosen backend |
| `--base-url` | `http://localhost:8000/v1` | vLLM server URL (only used with `--backend vllm`) |
| `--max-tokens` | 2048 | Max tokens in the model response |
| `--requests-per-minute` | 5 | API rate limit (requests/min) |
| `--min-turns` | 6 | Only used when streaming (ignored if `--input` is set) |
| `--max-conversations` | *(all)* | Hard cap on conversations to annotate |
| `--language` | *(all)* | Language filter when streaming |

The run is **resumable** — if it is interrupted, re-run the same command and already-annotated conversation IDs will be skipped automatically.

---

### Step 3 — Inspect statistics

```bash
# Print to terminal
python main.py stats --input output/annotations.jsonl

# Write to a file
python main.py stats --input output/annotations.jsonl --output output/stats.txt
```

The stats report includes:
- Overview (total, relevant %, avg dimensions per relevant conversation, avg turns)
- Full dimension breakdown ranked by frequency with average confidence
- Full country breakdown with relevance rate per country
- Model breakdown (when annotations from multiple models are mixed)

| Flag | Default | Description |
|---|---|---|
| `--input` | required | Annotation JSONL file |
| `--output` | *(terminal)* | Optional path to write the stats report as a text file |

---

## Backends

### Anthropic (default)

Uses the Anthropic API. Requires `ANTHROPIC_API_KEY` in your environment or `.env`.

```bash
python main.py annotate --backend anthropic --model claude-sonnet-4-6 ...
```

### OpenAI

Uses the OpenAI API. Requires `OPENAI_API_KEY`.

```bash
python main.py annotate --backend openai --model gpt-4o ...
```

### vLLM (local)

Runs against a locally-served vLLM endpoint via the OpenAI-compatible API.

```bash
python main.py annotate --backend vllm --model Qwen/Qwen3-30B-A3B --base-url http://localhost:8000/v1 ...
```

For a full end-to-end vLLM run (start server → annotate → stop server) use the helper script:

```bash
bash scripts/run.sh output/sample.jsonl output/annotations-vllm.jsonl
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
│     • sends system + user prompt to the     │
│       chosen backend                        │
│     • throttles to respect --requests-per-  │
│       minute limit                          │
│     • retries on rate-limit / API errors    │
│                                             │
│  3. _parse_response()                       │
│     • strips any markdown fences            │
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
| `relevant_dimensions[].indicators` | list[str] | Verbatim text spans from the conversation |
| `relevant_dimensions[].confidence` | float | 0–1 annotator confidence |
| `reasoning` | str | 1–3 sentence justification |
| `metadata.model` | str | Model used for annotation |
| `metadata.backend` | str | Backend used (`anthropic`, `openai`, or `vllm`) |
| `metadata.annotated_at` | str | ISO 8601 UTC timestamp |
| `metadata.country` | str | WildChat `country` field (if available) |
| `metadata.language` | str | WildChat `language` field (if available) |

---

## CultureScope Taxonomy

The taxonomy has three layers, five L2 subcategories, and ~130 fine-grained dimensions:

| Layer 1 | Layer 2 Subcategory |
|---|---|
| **Institutional Norms** | Geography & Customs |
| **Institutional Norms** | Regulation & Policy |
| **Behavioral Patterns** | Personal Choices & Habits |
| **Core Values and Social Structures** | Social Relationship and Structures |
| **Core Values and Social Structures** | Values and Beliefs |

See [`src/taxonomy.py`](src/taxonomy.py) for the full list of dimension keys and descriptions.

---

## Model Comparison Findings

Annotations were run on the same 100 English WildChat conversations (≥ 6 turns) using three models. Key findings:

| Model | Relevant % | Avg dims / relevant conv | Notes |
|---|---|---|---|
| Claude Sonnet 4.6 | 40.8% | 5.5 | Most liberal; longest reasoning; strong on societal/structural dimensions |
| GPT-4.5 | 32.3% | 4.9 | Mid-range; unique `cultural_acknowledgement` dimension; over-annotates fictional worldbuilding |
| Qwen3-30B-A3B | 28.4% | 6.9 | Most conservative; highest dims-per-flagged-conv; strong on humour/idioms; over-reads tech culture |

**Inter-model agreement** on 94 matched conversations (3-way Jaccard):

| Level | Avg Jaccard |
|---|---|
| Fine-grained dimension | 0.21 (agreed-relevant) / 0.08 (all non-neutral) |
| L2 subcategory | 0.54 / 0.20 |
| L1 layer | 0.61 / 0.23 |

The models agree much better on *which broad cultural domain* is present than on *which specific dimension* applies. **L2-level rollups are significantly more reliable** for downstream analysis than individual dimension counts.

---

## Configuration

Edit [`config.py`](config.py) or set environment variables:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `CULTURE_CHAT_BACKEND` | `anthropic` | Default backend |
| `CULTURE_CHAT_MODEL` | `claude-sonnet-4-6` | Default model |
| `CULTURE_CHAT_MAX_TOKENS` | `2048` | Max tokens in annotation response |
| `CULTURE_CHAT_MIN_TURNS` | `6` | Minimum conversation turns |
| `CULTURE_CHAT_OUTPUT_DIR` | `./output` | Default output directory |
