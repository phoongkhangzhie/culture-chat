"""culture-chat – CLI entry point.

Commands:

  annotate    Annotate WildChat conversations from a JSONL file or stream
  sample      Download and save a sample of WildChat conversations to JSONL
  stats       Print annotation statistics from an output file

Run any command with --help for options:
  python main.py annotate --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ---------------------------------------------------------------------------
# annotate command
# ---------------------------------------------------------------------------

def cmd_annotate(args: argparse.Namespace) -> None:
    from src.annotator import Annotator
    from src.loader import load_wildchat
    from src.models import Conversation, Turn

    annotator = Annotator(model=args.model, max_tokens=args.max_tokens)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conversations: list[Conversation]

    if args.input:
        # Load from a local JSONL file (produced by `sample` command)
        conversations = []
        with open(args.input) as f:
            for line in f:
                row = json.loads(line)
                turns = [Turn(**t) for t in row["turns"]]
                conversations.append(
                    Conversation(
                        conversation_id=row["conversation_id"],
                        turns=turns,
                        language=row.get("language"),
                        country=row.get("country"),
                        metadata=row.get("metadata", {}),
                    )
                )
        if args.max_conversations:
            conversations = conversations[: args.max_conversations]
    else:
        # Stream directly from HuggingFace
        conversations = list(
            load_wildchat(
                min_turns=args.min_turns,
                max_conversations=args.max_conversations,
                language_filter=args.language,
            )
        )

    annotations = annotator.annotate_batch(conversations, on_error="skip")

    with open(output_path, "w") as f:
        for ann in annotations:
            f.write(ann.model_dump_json() + "\n")

    print(f"Annotated {len(annotations)} conversations → {output_path}")
    _print_stats(annotations)


# ---------------------------------------------------------------------------
# sample command
# ---------------------------------------------------------------------------

def cmd_sample(args: argparse.Namespace) -> None:
    from src.loader import load_wildchat

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for conv in load_wildchat(
            min_turns=args.min_turns,
            max_conversations=args.n,
            language_filter=args.language,
        ):
            f.write(conv.model_dump_json() + "\n")
            count += 1

    print(f"Saved {count} conversations → {output_path}")


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------

def _print_stats(annotations: list) -> None:
    total = len(annotations)
    relevant = sum(1 for a in annotations if a.is_culturally_relevant)
    neutral = total - relevant

    print(f"\n{'='*50}")
    print(f"  Total annotated:       {total}")
    print(f"  Culturally relevant:   {relevant}  ({relevant/total*100:.1f}%)" if total else "  No annotations.")
    print(f"  Culturally neutral:    {neutral}  ({neutral/total*100:.1f}%)" if total else "")

    # Dimension frequency
    from collections import Counter
    dim_counter: Counter = Counter()
    for ann in annotations:
        for match in ann.relevant_dimensions:
            dim_counter[match.dimension_key] += 1

    if dim_counter:
        print(f"\n  Top 10 dimensions:")
        for key, cnt in dim_counter.most_common(10):
            print(f"    {key:<45} {cnt}")
    print("=" * 50)


def cmd_stats(args: argparse.Namespace) -> None:
    from src.models import Annotation

    annotations = []
    with open(args.input) as f:
        for line in f:
            annotations.append(Annotation.model_validate_json(line))

    _print_stats(annotations)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="culture-chat",
        description="Annotate WildChat conversations for cultural relevance using CultureScope.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # -- annotate --
    p_ann = sub.add_parser("annotate", help="Annotate conversations and write JSONL output")
    p_ann.add_argument("--input", "-i", help="Local JSONL file of conversations (from `sample`). If omitted, streams from HuggingFace.")
    p_ann.add_argument("--output", "-o", required=True, help="Output JSONL file for annotations")
    p_ann.add_argument("--model", default="claude-opus-4-6", help="Anthropic model to use")
    p_ann.add_argument("--max-tokens", type=int, default=2048)
    p_ann.add_argument("--min-turns", type=int, default=6, help="Minimum conversation turns (default 6)")
    p_ann.add_argument("--max-conversations", type=int, default=None, help="Cap on conversations to annotate")
    p_ann.add_argument("--language", default=None, help="Filter by language (e.g. 'English')")

    # -- sample --
    p_sam = sub.add_parser("sample", help="Download and save a WildChat sample to JSONL")
    p_sam.add_argument("--output", "-o", required=True, help="Output JSONL file")
    p_sam.add_argument("--n", type=int, default=100, help="Number of conversations to save")
    p_sam.add_argument("--min-turns", type=int, default=6)
    p_sam.add_argument("--language", default=None)

    # -- stats --
    p_sta = sub.add_parser("stats", help="Print statistics from an annotation JSONL file")
    p_sta.add_argument("--input", "-i", required=True, help="Annotation JSONL file")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    dispatch = {
        "annotate": cmd_annotate,
        "sample": cmd_sample,
        "stats": cmd_stats,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
