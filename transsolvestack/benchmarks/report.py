"""Benchmark report writer."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from transsolvestack.benchmarks.validation import BenchmarkDryRunReport


def write_dry_run_report(report: BenchmarkDryRunReport, path: str | Path) -> Path:
    """Write a compact Markdown report for a dry-run validation."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(report)
    lines = [
        f"# Benchmark Dry Run: {report.workload_id}",
        "",
        f"- candidate_set_id: `{report.candidate_set_id}`",
        f"- backend: `{report.backend}`",
        f"- budget_mode: `{report.budget_mode}`",
        f"- num_families: `{report.num_families}`",
        f"- num_contexts: `{report.num_contexts}`",
        f"- num_candidates: `{report.num_candidates}`",
        f"- num_systems: `{report.num_systems}`",
        f"- max_problem_n: `{report.max_problem_n}`",
        f"- max_effective_nnz: `{report.max_effective_nnz}`",
        f"- max_candidate_iter: `{report.max_candidate_iter}`",
        "",
        "## Warnings",
        "",
    ]
    if report.warnings:
        lines.extend(f"- {warning}" for warning in report.warnings)
    else:
        lines.append("- none")
    lines.extend(["", "## Raw", "", "```text"])
    lines.extend(f"{key}: {value}" for key, value in data.items())
    lines.extend(["```", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
