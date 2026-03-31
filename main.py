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

def _load_conversations_from_jsonl(path: str) -> list:
    """Load conversations from a JSONL file.

    New sample files (produced by `sample`) always have a stable conv_NNNN ID.
    For older files that still have empty IDs, fall back to a positional ID so
    resumability still works.
    """
    from src.models import Conversation, Turn

    conversations = []
    with open(path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            conv_id = row.get("conversation_id") or f"conv_{i:04d}"
            turns = [Turn(**t) for t in row["turns"]]
            conversations.append(
                Conversation(
                    conversation_id=conv_id,
                    turns=turns,
                    language=row.get("language"),
                    country=row.get("country"),
                    metadata=row.get("metadata", {}),
                )
            )
    return conversations


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
        conversations = _load_conversations_from_jsonl(args.input)
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
            if not conv.conversation_id:
                conv.conversation_id = f"conv_{count:04d}"
            f.write(conv.model_dump_json() + "\n")
            count += 1

    print(f"Saved {count} conversations → {output_path}")


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------

def _print_stats(annotations: list, output_path: Path | None = None) -> None:
    from collections import Counter, defaultdict
    from statistics import mean, median

    from rich.console import Console
    from rich.table import Table
    from rich import box

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(output_path, "w")
        console = Console(file=fh, highlight=False)
    else:
        fh = None
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
        console.rule("[bold]Dimensions[/bold]")
        dim_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        dim_table.add_column("Rank",       justify="right",  style="dim", width=5)
        dim_table.add_column("Key",        style="cyan",     no_wrap=True)
        dim_table.add_column("Name",       style="white")
        dim_table.add_column("Count",      justify="right",  style="bold green")
        dim_table.add_column("%",          justify="right",  style="dim")
        dim_table.add_column("Avg Conf",   justify="right",  style="yellow")

        for rank, (key, cnt) in enumerate(dim_counts.most_common(), 1):
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
        console.rule("[bold]Countries[/bold]")
        cty_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        cty_table.add_column("Country",    style="blue")
        cty_table.add_column("Total",      justify="right")
        cty_table.add_column("Relevant",   justify="right", style="bold green")
        cty_table.add_column("Rel %",      justify="right", style="dim")
        for country, tot in country_total.most_common():
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

    if fh:
        fh.close()
        print(f"Stats written → {output_path}")


def _print_open_analysis_stats(open_analyses: list, output_path: Path | None = None) -> None:
    from collections import Counter
    from rich.console import Console
    from rich.table import Table
    from rich import box

    if output_path:
        fh = open(output_path, "a")
        console = Console(file=fh, highlight=False)
    else:
        fh = None
        console = Console()

    total = len(open_analyses)
    if not total:
        console.print("[yellow]No open analyses to report.[/yellow]")
        return

    with_issues = [oa for oa in open_analyses if oa.has_issues]
    all_obs = [o for oa in open_analyses for o in oa.observations]

    console.rule("[bold]Open-Ended Failure Analysis (Pass 1)[/bold]")
    ov = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    ov.add_column(style="dim")
    ov.add_column(justify="right", style="bold")
    ov.add_column(justify="right", style="dim")
    ov.add_row("Culturally-relevant convs analysed", str(total), "")
    ov.add_row("With issues", str(len(with_issues)), f"{len(with_issues)/total*100:.1f}%")
    ov.add_row("No issues", str(total - len(with_issues)), f"{(total-len(with_issues))/total*100:.1f}%")
    ov.add_row("Total observations", str(len(all_obs)), "")
    if with_issues:
        avg_obs = len(all_obs) / len(with_issues)
        ov.add_row("Avg observations / conv with issues", f"{avg_obs:.1f}", "")
    console.print(ov)

    # Severity breakdown across all observations
    sev_counter: Counter = Counter()
    for o in all_obs:
        sev_counter[o.severity] += 1

    if sev_counter:
        console.rule("[bold]Observations by Severity[/bold]")
        sev_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        sev_table.add_column("Severity", style="bold")
        sev_table.add_column("Count", justify="right")
        sev_table.add_column("%", justify="right", style="dim")
        sev_colours = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}
        for sev in ["critical", "high", "medium", "low"]:
            cnt = sev_counter.get(sev, 0)
            if cnt:
                colour = sev_colours[sev]
                sev_table.add_row(
                    f"[{colour}]{sev}[/{colour}]",
                    str(cnt),
                    f"{cnt/len(all_obs)*100:.1f}%",
                )
        console.print(sev_table)

    # Conv-level overall severity
    conv_sev: Counter = Counter()
    for oa in with_issues:
        conv_sev[oa.overall_severity or "unspecified"] += 1

    if conv_sev:
        console.rule("[bold]Conversations by Overall Severity[/bold]")
        cs_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        cs_table.add_column("Overall Severity", style="bold")
        cs_table.add_column("Conversations", justify="right")
        cs_table.add_column("%", justify="right", style="dim")
        sev_colours = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green", "unspecified": "dim"}
        for sev in ["critical", "high", "medium", "low", "unspecified"]:
            cnt = conv_sev.get(sev, 0)
            if cnt:
                colour = sev_colours.get(sev, "white")
                cs_table.add_row(
                    f"[{colour}]{sev}[/{colour}]",
                    str(cnt),
                    f"{cnt/len(with_issues)*100:.1f}%",
                )
        console.print(cs_table)

    # Country breakdown
    country_total: Counter = Counter()
    country_issues: Counter = Counter()
    for oa in open_analyses:
        c = oa.metadata.get("country") or "Unknown"
        country_total[c] += 1
        if oa.has_issues:
            country_issues[c] += 1

    if len(country_total) > 1:
        console.rule("[bold]Countries (Open Analysis)[/bold]")
        cty_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        cty_table.add_column("Country", style="blue")
        cty_table.add_column("Analysed", justify="right")
        cty_table.add_column("With Issues", justify="right", style="bold red")
        cty_table.add_column("Issue %", justify="right", style="dim")
        for country, tot in country_total.most_common():
            iss = country_issues[country]
            cty_table.add_row(country, str(tot), str(iss), f"{iss/tot*100:.1f}%")
        console.print(cty_table)

    if fh:
        fh.close()


def _print_synthesis_stats(report, output_path: Path | None = None) -> None:
    from collections import Counter
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.text import Text

    if output_path:
        fh = open(output_path, "a")
        console = Console(file=fh, highlight=False)
    else:
        fh = None
        console = Console()

    console.rule("[bold]Emergent Failure Taxonomy (Pass 2 Synthesis)[/bold]")

    ov = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    ov.add_column(style="dim")
    ov.add_column(justify="right", style="bold")
    ov.add_row("Conversations in synthesis", str(report.total_conversations))
    ov.add_row("With issues", str(report.total_with_issues))
    ov.add_row("Total observations", str(report.total_observations))
    ov.add_row("Emergent patterns identified", str(len(report.patterns)))
    if report.uncategorised_observations:
        ov.add_row("Uncategorised observations", str(len(report.uncategorised_observations)))
    console.print(ov)

    if report.synthesis_summary:
        console.rule("[bold]Synthesis Summary[/bold]")
        console.print(report.synthesis_summary)

    if report.patterns:
        console.rule("[bold]Emergent Patterns[/bold]")
        pat_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        pat_table.add_column("Rank",        justify="right", style="dim", width=5)
        pat_table.add_column("Pattern",     style="cyan", no_wrap=False)
        pat_table.add_column("Freq",        justify="right", style="bold green")
        pat_table.add_column("Severity",    justify="left")
        sev_colours = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}
        for rank, p in enumerate(sorted(report.patterns, key=lambda x: x.frequency, reverse=True), 1):
            sev_parts = []
            for sev in ["critical", "high", "medium", "low"]:
                cnt = p.severity_distribution.get(sev, 0)
                if cnt:
                    colour = sev_colours[sev]
                    sev_parts.append(f"[{colour}]{sev}:{cnt}[/{colour}]")
            sev_str = "  ".join(sev_parts) if sev_parts else "—"
            pat_table.add_row(str(rank), p.name, str(p.frequency), sev_str)
        console.print(pat_table)

        # Detail block per pattern
        console.rule("[bold]Pattern Descriptions[/bold]")
        for rank, p in enumerate(sorted(report.patterns, key=lambda x: x.frequency, reverse=True), 1):
            console.print(f"\n[bold cyan]{rank}. {p.name}[/bold cyan]  (freq={p.frequency})")
            console.print(f"   [dim]{p.description}[/dim]")
            if p.example_evidence:
                console.print("   [italic]Evidence:[/italic]")
                for ev in p.example_evidence[:3]:
                    console.print(f"     • {ev}")

    if report.uncategorised_observations:
        console.rule("[bold]Uncategorised Observations[/bold]")
        for obs in report.uncategorised_observations:
            console.print(f"  • {obs}")

    if fh:
        fh.close()


def cmd_stats(args: argparse.Namespace) -> None:
    from src.models import Annotation, FailureAnalysis, OpenAnalysis, SynthesisReport

    annotations = []
    with open(args.input) as f:
        for line in f:
            annotations.append(Annotation.model_validate_json(line))

    failures = []
    if args.failures:
        with open(args.failures) as f:
            for line in f:
                failures.append(FailureAnalysis.model_validate_json(line))

    open_analyses = []
    if args.open_analyses:
        with open(args.open_analyses) as f:
            for line in f:
                open_analyses.append(OpenAnalysis.model_validate_json(line))

    synthesis = None
    if args.synthesis:
        raw = Path(args.synthesis).read_text()
        synthesis = SynthesisReport.model_validate_json(raw)

    output_path = Path(args.output) if args.output else None
    _print_stats(annotations, output_path=output_path)
    if failures:
        _print_failure_stats(failures, output_path=output_path)
    if open_analyses:
        _print_open_analysis_stats(open_analyses, output_path=output_path)
    if synthesis:
        _print_synthesis_stats(synthesis, output_path=output_path)


# ---------------------------------------------------------------------------
# analyse command
# ---------------------------------------------------------------------------

def cmd_analyse(args: argparse.Namespace) -> None:
    from src.annotator import Annotator
    from src.models import Annotation

    # ── Load conversations (with stable synthetic IDs) ───────────────────────
    # _load_conversations_from_jsonl assigns conv_0000, conv_0001, … by position,
    # matching exactly what cmd_annotate stored in the annotations file.
    conv_by_id = {c.conversation_id: c for c in _load_conversations_from_jsonl(args.conversations)}

    # ── Load annotations ─────────────────────────────────────────────────────
    annotations: list[Annotation] = []
    with open(args.annotations) as f:
        for line in f:
            annotations.append(Annotation.model_validate_json(line))

    relevant = [a for a in annotations if a.is_culturally_relevant]
    print(f"Loaded {len(annotations)} annotations, {len(relevant)} culturally-relevant.")

    # ── Match annotations to conversation text by ID ─────────────────────────
    conv_lookup = []
    for ann in relevant:
        conv = conv_by_id.get(ann.conversation_id)
        if conv:
            conv_lookup.append((conv, ann))
        else:
            logging.getLogger(__name__).warning(
                "Could not match annotation id=%r to a conversation — skipping.",
                ann.conversation_id,
            )

    print(f"Matched {len(conv_lookup)} relevant conversations to source text.")

    # ── Run failure analysis ─────────────────────────────────────────────────
    annotator = Annotator(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        requests_per_minute=args.requests_per_minute,
    )
    output_path = Path(args.output)
    conversations, anns = zip(*conv_lookup) if conv_lookup else ([], [])

    results = annotator.analyse_failures_batch(
        list(conversations),
        list(anns),
        output_path=output_path,
        on_error="skip",
    )

    failures_found = [r for r in results if r.has_failures]
    print(f"\nAnalysed {len(results)} conversations → {output_path}")
    print(f"Failures found: {len(failures_found)}/{len(results)} ({len(failures_found)/len(results)*100:.1f}%)" if results else "")
    _print_failure_stats(results, output_path=None)


def _print_failure_stats(results: list, output_path: Path | None = None) -> None:
    from collections import Counter
    from rich.console import Console
    from rich.table import Table
    from rich import box

    if output_path:
        fh = open(output_path, "a")
        console = Console(file=fh, highlight=False)
    else:
        fh = None
        console = Console()
    total = len(results)
    if not total:
        console.print("[yellow]No failure analyses to report.[/yellow]")
        return

    has_failures = [r for r in results if r.has_failures]
    console.rule("[bold]Failure Analysis Summary[/bold]")

    overview = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    overview.add_column(style="dim")
    overview.add_column(justify="right", style="bold")
    overview.add_column(justify="right", style="dim")
    overview.add_row("Conversations analysed", str(total), "")
    overview.add_row("With failures", str(len(has_failures)), f"{len(has_failures)/total*100:.1f}%")
    overview.add_row("No failures", str(total - len(has_failures)), f"{(total-len(has_failures))/total*100:.1f}%")
    console.print(overview)

    # Severity breakdown
    sev_counter: Counter = Counter()
    type_counter: Counter = Counter()
    type_sev_sums: dict = {}
    for r in results:
        for f in r.failures:
            sev_counter[f.severity] += 1
            type_counter[f.failure_type] += 1
            sev_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            type_sev_sums[f.failure_type] = max(
                type_sev_sums.get(f.failure_type, 0),
                sev_order.get(f.severity, 0),
            )

    if sev_counter:
        console.rule("[bold]Failures by Severity[/bold]")
        sev_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        sev_table.add_column("Severity", style="bold")
        sev_table.add_column("Count", justify="right")
        sev_table.add_column("%", justify="right", style="dim")
        total_failures = sum(sev_counter.values())
        sev_order_display = ["critical", "high", "medium", "low"]
        sev_colours = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}
        for sev in sev_order_display:
            cnt = sev_counter.get(sev, 0)
            if cnt:
                sev_table.add_row(
                    f"[{sev_colours[sev]}]{sev}[/{sev_colours[sev]}]",
                    str(cnt),
                    f"{cnt/total_failures*100:.1f}%",
                )
        console.print(sev_table)

    if type_counter:
        console.rule("[bold]Failure Types[/bold]")
        type_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        type_table.add_column("Rank", justify="right", style="dim", width=5)
        type_table.add_column("Failure Type", style="cyan")
        type_table.add_column("Count", justify="right", style="bold")
        type_table.add_column("%", justify="right", style="dim")
        sev_rev = {1: "low", 2: "medium", 3: "high", 4: "critical"}
        sev_colours = {"critical": "red", "high": "orange3", "medium": "yellow", "low": "green"}
        for rank, (ftype, cnt) in enumerate(type_counter.most_common(), 1):
            worst_sev = sev_rev.get(type_sev_sums.get(ftype, 0), "low")
            colour = sev_colours[worst_sev]
            type_table.add_row(
                str(rank),
                ftype,
                str(cnt),
                f"{cnt/total_failures*100:.1f}%",
            )
        console.print(type_table)

    # Country breakdown
    country_total: Counter = Counter()
    country_failures: Counter = Counter()
    for r in results:
        c = r.metadata.get("country") or "Unknown"
        country_total[c] += 1
        if r.has_failures:
            country_failures[c] += 1

    if len(country_total) > 1:
        console.rule("[bold]Countries[/bold]")
        cty_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        cty_table.add_column("Country", style="blue")
        cty_table.add_column("Analysed", justify="right")
        cty_table.add_column("With Failures", justify="right", style="bold red")
        cty_table.add_column("Failure %", justify="right", style="dim")
        for country, tot in country_total.most_common():
            fail = country_failures[country]
            cty_table.add_row(country, str(tot), str(fail), f"{fail/tot*100:.1f}%")
        console.print(cty_table)

    if fh:
        fh.close()


# ---------------------------------------------------------------------------
# open-analyse command  (Option B, Pass 1)
# ---------------------------------------------------------------------------

def cmd_open_analyse(args: argparse.Namespace) -> None:
    from src.annotator import Annotator
    from src.models import Annotation

    conv_by_id = {c.conversation_id: c for c in _load_conversations_from_jsonl(args.conversations)}

    annotations: list[Annotation] = []
    with open(args.annotations) as f:
        for line in f:
            annotations.append(Annotation.model_validate_json(line))

    relevant = [a for a in annotations if a.is_culturally_relevant]
    print(f"Loaded {len(annotations)} annotations, {len(relevant)} culturally-relevant.")

    conv_lookup = []
    for ann in relevant:
        conv = conv_by_id.get(ann.conversation_id)
        if conv:
            conv_lookup.append((conv, ann))
        else:
            logging.getLogger(__name__).warning(
                "Could not match annotation id=%r to a conversation — skipping.",
                ann.conversation_id,
            )
    print(f"Matched {len(conv_lookup)} conversations to source text.")

    annotator = Annotator(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        requests_per_minute=args.requests_per_minute,
    )
    output_path = Path(args.output)
    conversations, anns = zip(*conv_lookup) if conv_lookup else ([], [])

    results = annotator.open_analyse_batch(
        list(conversations), list(anns), output_path=output_path, on_error="skip",
    )

    with_issues = [r for r in results if r.has_issues]
    print(f"\nOpen-analysed {len(results)} conversations → {output_path}")
    print(f"With issues: {len(with_issues)}/{len(results)} ({len(with_issues)/len(results)*100:.1f}%)" if results else "")


# ---------------------------------------------------------------------------
# synthesise command  (Option B, Pass 2)
# ---------------------------------------------------------------------------

def cmd_synthesise(args: argparse.Namespace) -> None:
    from src.annotator import Annotator
    from src.models import OpenAnalysis
    from rich.console import Console
    from rich.table import Table
    from rich import box

    open_analyses: list[OpenAnalysis] = []
    with open(args.input) as f:
        for line in f:
            open_analyses.append(OpenAnalysis.model_validate_json(line))

    with_issues = [oa for oa in open_analyses if oa.has_issues]
    total_obs = sum(len(oa.observations) for oa in with_issues)
    print(f"Loaded {len(open_analyses)} open analyses, {len(with_issues)} with issues, {total_obs} total observations.")

    annotator = Annotator(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        requests_per_minute=args.requests_per_minute,
    )
    output_path = Path(args.output)
    report = annotator.synthesise(open_analyses, output_path=output_path)
    print(f"\nSynthesis complete → {output_path}")

    console = Console()
    console.rule("[bold]Emergent Failure Patterns[/bold]")
    console.print(f"\n[dim]Conversations: {report.total_conversations} | "
                  f"With issues: {report.total_with_issues} | "
                  f"Observations: {report.total_observations}[/dim]\n")

    if report.patterns:
        tbl = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        tbl.add_column("Rank", justify="right", style="dim", width=5)
        tbl.add_column("Pattern", style="cyan")
        tbl.add_column("Freq", justify="right", style="bold")
        tbl.add_column("Severities", style="dim")
        tbl.add_column("Description")
        for rank, p in enumerate(report.patterns, 1):
            sev = p.severity_distribution
            sev_str = "  ".join(
                f"[{'red' if k=='critical' else 'orange3' if k=='high' else 'yellow' if k=='medium' else 'green'}]{k[0].upper()}:{sev.get(k,0)}[/]"
                for k in ["critical","high","medium","low"] if sev.get(k,0)
            )
            tbl.add_row(str(rank), p.name, str(p.frequency), sev_str, p.description[:80])
        console.print(tbl)

    console.rule("[bold]Synthesis Summary[/bold]")
    console.print(report.synthesis_summary)

    if report.uncategorised_observations:
        console.rule(f"[bold]Uncategorised ({len(report.uncategorised_observations)})[/bold]")
        for obs in report.uncategorised_observations:
            console.print(f"  • {obs[:120]}")


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
    p_sta.add_argument("--failures", "-f", default=None, help="Failure analysis JSONL file (output of `analyse`)")
    p_sta.add_argument("--open-analyses", default=None, help="Open-ended analysis JSONL file (output of `open-analyse`)")
    p_sta.add_argument("--synthesis", default=None, help="Synthesis JSON file (output of `synthesise`)")
    p_sta.add_argument("--output", "-o", default=None, help="Write stats to a text file instead of (or in addition to) the terminal")

    # -- analyse --
    p_ana = sub.add_parser("analyse", parents=[verbose_parser], help="Analyse culturally-relevant conversations for AI assistant failure modes")
    p_ana.add_argument("--annotations", "-a", required=True, help="Annotation JSONL file (output of `annotate`)")
    p_ana.add_argument("--conversations", "-c", required=True, help="Source conversation JSONL file (output of `sample`)")
    p_ana.add_argument("--output", "-o", required=True, help="Output JSONL file for failure analyses")
    p_ana.add_argument("--backend", default="anthropic", choices=["anthropic", "openai", "vllm"])
    p_ana.add_argument("--model", default="claude-sonnet-4-6")
    p_ana.add_argument("--base-url", default="http://localhost:8000/v1")
    p_ana.add_argument("--max-tokens", type=int, default=2048)
    p_ana.add_argument("--requests-per-minute", type=int, default=5)

    # -- open-analyse --  (Option B, Pass 1)
    p_oa = sub.add_parser("open-analyse", parents=[verbose_parser],
                          help="Open-ended cultural failure analysis — no predefined failure types (Pass 1 of 2)")
    p_oa.add_argument("--annotations", "-a", required=True, help="Annotation JSONL file (output of `annotate`)")
    p_oa.add_argument("--conversations", "-c", required=True, help="Source conversation JSONL file (output of `sample`)")
    p_oa.add_argument("--output", "-o", required=True, help="Output JSONL file for open analyses")
    p_oa.add_argument("--backend", default="anthropic", choices=["anthropic", "openai", "vllm"])
    p_oa.add_argument("--model", default="claude-sonnet-4-6")
    p_oa.add_argument("--base-url", default="http://localhost:8000/v1")
    p_oa.add_argument("--max-tokens", type=int, default=4096, help="Open analyses need more tokens than annotation (default 4096)")
    p_oa.add_argument("--requests-per-minute", type=int, default=5)

    # -- synthesise --  (Option B, Pass 2)
    p_syn = sub.add_parser("synthesise", parents=[verbose_parser],
                           help="Synthesise emergent failure patterns from open analyses (Pass 2 of 2)")
    p_syn.add_argument("--input", "-i", required=True, help="Open-analysis JSONL file (output of `open-analyse`)")
    p_syn.add_argument("--output", "-o", required=True, help="Output JSON file for the synthesis report")
    p_syn.add_argument("--backend", default="anthropic", choices=["anthropic", "openai", "vllm"])
    p_syn.add_argument("--model", default="claude-sonnet-4-6")
    p_syn.add_argument("--base-url", default="http://localhost:8000/v1")
    p_syn.add_argument("--max-tokens", type=int, default=8192, help="Synthesis needs more tokens (default 8192)")
    p_syn.add_argument("--requests-per-minute", type=int, default=5)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))

    dispatch = {
        "annotate": cmd_annotate,
        "sample": cmd_sample,
        "stats": cmd_stats,
        "analyse": cmd_analyse,
        "open-analyse": cmd_open_analyse,
        "synthesise": cmd_synthesise,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
