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

from dotenv import load_dotenv
load_dotenv()


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

    annotator = Annotator(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        requests_per_minute=args.requests_per_minute,
    )
    output_path = Path(args.output)

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

    annotations = annotator.annotate_batch(conversations, output_path=output_path, on_error="skip")

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
    from collections import Counter, defaultdict
    from statistics import mean, median

    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    total = len(annotations)
    if not total:
        console.print("[yellow]No annotations to report.[/yellow]")
        return

    relevant_anns = [a for a in annotations if a.is_culturally_relevant]
    neutral_anns  = [a for a in annotations if not a.is_culturally_relevant]
    relevant = len(relevant_anns)
    neutral  = len(neutral_anns)

    # ── Overview ────────────────────────────────────────────────────────────
    console.rule("[bold]Annotation Summary[/bold]")
    overview = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    overview.add_column(style="dim")
    overview.add_column(justify="right", style="bold")
    overview.add_column(justify="right", style="dim")
    overview.add_row("Total annotated",       str(total),    "")
    overview.add_row("Culturally relevant",   str(relevant), f"{relevant/total*100:.1f}%")
    overview.add_row("Culturally neutral",    str(neutral),  f"{neutral/total*100:.1f}%")

    # avg dimensions per relevant conversation
    dims_per_conv = [len(a.relevant_dimensions) for a in relevant_anns]
    if dims_per_conv:
        overview.add_row("Avg dimensions / relevant conv", f"{mean(dims_per_conv):.1f}", "")
        overview.add_row("Median dimensions / relevant conv", f"{median(dims_per_conv):.1f}", "")

    # avg turns
    all_turns = [a.num_turns for a in annotations if a.num_turns]
    if all_turns:
        rel_turns = [a.num_turns for a in relevant_anns if a.num_turns]
        neu_turns = [a.num_turns for a in neutral_anns  if a.num_turns]
        overview.add_row("Avg turns (all)",      f"{mean(all_turns):.1f}", "")
        if rel_turns:
            overview.add_row("Avg turns (relevant)", f"{mean(rel_turns):.1f}", "")
        if neu_turns:
            overview.add_row("Avg turns (neutral)",  f"{mean(neu_turns):.1f}", "")

    console.print(overview)

    # ── Dimension frequency ─────────────────────────────────────────────────
    dim_counts:    Counter = Counter()
    dim_conf_sums: dict    = defaultdict(float)
    dim_names:     dict    = {}

    for ann in annotations:
        for m in ann.relevant_dimensions:
            dim_counts[m.dimension_key] += 1
            dim_conf_sums[m.dimension_key] += m.confidence
            dim_names[m.dimension_key] = m.dimension_name

    if dim_counts:
        console.rule("[bold]Top 20 Dimensions[/bold]")
        dim_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        dim_table.add_column("Rank",       justify="right",  style="dim", width=5)
        dim_table.add_column("Key",        style="cyan",     no_wrap=True)
        dim_table.add_column("Name",       style="white")
        dim_table.add_column("Count",      justify="right",  style="bold green")
        dim_table.add_column("%",          justify="right",  style="dim")
        dim_table.add_column("Avg Conf",   justify="right",  style="yellow")

        for rank, (key, cnt) in enumerate(dim_counts.most_common(20), 1):
            avg_conf = dim_conf_sums[key] / cnt
            dim_table.add_row(
                str(rank),
                key,
                dim_names.get(key, ""),
                str(cnt),
                f"{cnt/total*100:.1f}%",
                f"{avg_conf:.2f}",
            )
        console.print(dim_table)

    # ── Country breakdown ────────────────────────────────────────────────────
    country_total:    Counter = Counter()
    country_relevant: Counter = Counter()
    for ann in annotations:
        c = ann.metadata.get("country") or "Unknown"
        country_total[c] += 1
        if ann.is_culturally_relevant:
            country_relevant[c] += 1

    if len(country_total) > 1:
        console.rule("[bold]Top 15 Countries[/bold]")
        cty_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        cty_table.add_column("Country",    style="blue")
        cty_table.add_column("Total",      justify="right")
        cty_table.add_column("Relevant",   justify="right", style="bold green")
        cty_table.add_column("Rel %",      justify="right", style="dim")
        for country, tot in country_total.most_common(15):
            rel = country_relevant[country]
            cty_table.add_row(country, str(tot), str(rel), f"{rel/tot*100:.1f}%")
        console.print(cty_table)

    # ── Model breakdown ──────────────────────────────────────────────────────
    model_total:    Counter = Counter()
    model_relevant: Counter = Counter()
    for ann in annotations:
        m = ann.metadata.get("model") or "unknown"
        model_total[m] += 1
        if ann.is_culturally_relevant:
            model_relevant[m] += 1

    if len(model_total) > 1:
        console.rule("[bold]Model Breakdown[/bold]")
        mdl_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        mdl_table.add_column("Model",    style="cyan")
        mdl_table.add_column("Total",    justify="right")
        mdl_table.add_column("Relevant", justify="right", style="bold green")
        mdl_table.add_column("Rel %",    justify="right", style="dim")
        for model, tot in model_total.most_common():
            rel = model_relevant[model]
            mdl_table.add_row(model, str(tot), str(rel), f"{rel/tot*100:.1f}%")
        console.print(mdl_table)


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
    # Shared --verbose flag added to every subparser via a parent parser
    verbose_parser = argparse.ArgumentParser(add_help=False)
    verbose_parser.add_argument("--verbose", "-v", action="store_true")

    parser = argparse.ArgumentParser(
        prog="culture-chat",
        description="Annotate WildChat conversations for cultural relevance using CultureScope.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # -- annotate --
    p_ann = sub.add_parser("annotate", parents=[verbose_parser], help="Annotate conversations and write JSONL output")
    p_ann.add_argument("--input", "-i", help="Local JSONL file of conversations (from `sample`). If omitted, streams from HuggingFace.")
    p_ann.add_argument("--output", "-o", required=True, help="Output JSONL file for annotations")
    p_ann.add_argument("--backend", default="anthropic", choices=["anthropic", "openai", "vllm"], help="Backend to use (default: anthropic)")
    p_ann.add_argument("--model", default="claude-sonnet-4-6", help="Model name — Claude ID for anthropic, GPT ID for openai, served model name for vllm")
    p_ann.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM base URL (only used with --backend vllm)")
    p_ann.add_argument("--max-tokens", type=int, default=2048)
    p_ann.add_argument("--requests-per-minute", type=int, default=5, help="Max API requests per minute (default 5)")
    p_ann.add_argument("--min-turns", type=int, default=6, help="Minimum conversation turns (default 6)")
    p_ann.add_argument("--max-conversations", type=int, default=None, help="Cap on conversations to annotate")
    p_ann.add_argument("--language", default=None, help="Filter by language (e.g. 'English')")

    # -- sample --
    p_sam = sub.add_parser("sample", parents=[verbose_parser], help="Download and save a WildChat sample to JSONL")
    p_sam.add_argument("--output", "-o", required=True, help="Output JSONL file")
    p_sam.add_argument("--n", type=int, default=None, help="Number of conversations to save")
    p_sam.add_argument("--min-turns", type=int, default=6)
    p_sam.add_argument("--language", default=None)

    # -- stats --
    p_sta = sub.add_parser("stats", parents=[verbose_parser], help="Print statistics from an annotation JSONL file")
    p_sta.add_argument("--input", "-i", required=True, help="Annotation JSONL file")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))

    dispatch = {
        "annotate": cmd_annotate,
        "sample": cmd_sample,
        "stats": cmd_stats,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
