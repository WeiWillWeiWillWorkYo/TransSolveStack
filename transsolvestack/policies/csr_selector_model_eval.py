"""Offline quality gate for CSR selector model artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_SELECTOR_MODEL_EVAL_SCHEMA_VERSION = "phase1_csr_selector_model_eval_v1"
EVALUATION_ID = "csr_selector_model_quality_gate_v1"


@dataclass(frozen=True)
class CsrSelectorModelEvalRow:
    schema_version: str
    evaluation_id: str
    model_id: str
    model_family: str
    model_kind: str
    comparison_role: str
    model_trained: bool
    runtime_selector_changed: bool
    num_eval_predictions: int
    num_eval_oracle_requests: int
    num_profiled_success_predictions: int
    eval_oracle_top1_accuracy: float
    eval_profiled_success_selection_rate: float
    eval_non_success_selection_count: int
    eval_mean_regret_ms: float | None
    eval_max_regret_ms: float | None
    beats_baseline: bool
    runtime_eligible: bool
    gate_failures: tuple[str, ...]


@dataclass(frozen=True)
class CsrSelectorModelEvaluationSummary:
    status: str
    schema_version: str
    evaluation_id: str
    baseline_model_id: str
    challenger_model_id: str
    best_offline_model_id: str
    runtime_selected_model_id: str | None
    runtime_selector_changed: bool
    num_models: int
    num_eval_predictions: int
    num_eval_oracle_requests: int
    min_required_eval_oracle_requests: int
    min_runtime_oracle_top1_accuracy: float
    min_runtime_profiled_success_rate: float
    challenger_beats_baseline: bool
    challenger_runtime_eligible: bool
    challenger_gate_failures: tuple[str, ...]
    recommendation: str


@dataclass(frozen=True)
class CsrSelectorModelEvaluationExport:
    rows: tuple[CsrSelectorModelEvalRow, ...]
    summary: CsrSelectorModelEvaluationSummary
    schema: dict[str, Any]


def evaluate_csr_selector_models_from_files(
    baseline_summary_path: str | Path,
    baseline_predictions_path: str | Path,
    ranker_summary_path: str | Path,
    ranker_predictions_path: str | Path,
    *,
    min_eval_oracle_requests: int = 10,
    min_runtime_oracle_top1_accuracy: float = 0.5,
    min_runtime_profiled_success_rate: float = 2.0 / 3.0,
) -> CsrSelectorModelEvaluationExport:
    baseline_summary = json.loads(Path(baseline_summary_path).read_text(encoding="utf-8"))
    baseline_predictions = tuple(read_jsonl(baseline_predictions_path))
    ranker_summary = json.loads(Path(ranker_summary_path).read_text(encoding="utf-8"))
    ranker_predictions = tuple(read_jsonl(ranker_predictions_path))
    return evaluate_csr_selector_models(
        baseline_summary,
        baseline_predictions,
        ranker_summary,
        ranker_predictions,
        min_eval_oracle_requests=min_eval_oracle_requests,
        min_runtime_oracle_top1_accuracy=min_runtime_oracle_top1_accuracy,
        min_runtime_profiled_success_rate=min_runtime_profiled_success_rate,
    )


def evaluate_csr_selector_models(
    baseline_summary: dict[str, Any],
    baseline_predictions: Iterable[dict[str, Any]],
    ranker_summary: dict[str, Any],
    ranker_predictions: Iterable[dict[str, Any]],
    *,
    min_eval_oracle_requests: int = 10,
    min_runtime_oracle_top1_accuracy: float = 0.5,
    min_runtime_profiled_success_rate: float = 2.0 / 3.0,
) -> CsrSelectorModelEvaluationExport:
    baseline_metrics = _baseline_metrics(baseline_summary, tuple(baseline_predictions))
    challenger_metrics = _ranker_metrics(ranker_summary, tuple(ranker_predictions))
    challenger_failures = _challenger_gate_failures(
        baseline_metrics,
        challenger_metrics,
        min_eval_oracle_requests=min_eval_oracle_requests,
        min_runtime_oracle_top1_accuracy=min_runtime_oracle_top1_accuracy,
        min_runtime_profiled_success_rate=min_runtime_profiled_success_rate,
    )
    challenger_beats_baseline = _beats_baseline(challenger_metrics, baseline_metrics)
    challenger_runtime_eligible = not challenger_failures
    baseline_row = _row(
        baseline_metrics,
        comparison_role="baseline",
        beats_baseline=False,
        runtime_eligible=False,
        gate_failures=("comparison_baseline_not_runtime_candidate",),
    )
    challenger_row = _row(
        challenger_metrics,
        comparison_role="challenger",
        beats_baseline=challenger_beats_baseline,
        runtime_eligible=challenger_runtime_eligible,
        gate_failures=challenger_failures,
    )
    best = _best_model((baseline_metrics, challenger_metrics))
    status = (
        "passed"
        if _metrics_valid(baseline_metrics)
        and _metrics_valid(challenger_metrics)
        and baseline_metrics["num_eval_predictions"] == challenger_metrics["num_eval_predictions"]
        and baseline_metrics["num_eval_oracle_requests"]
        == challenger_metrics["num_eval_oracle_requests"]
        else "failed"
    )
    summary = CsrSelectorModelEvaluationSummary(
        status=status,
        schema_version=CSR_SELECTOR_MODEL_EVAL_SCHEMA_VERSION,
        evaluation_id=EVALUATION_ID,
        baseline_model_id=str(baseline_metrics["model_id"]),
        challenger_model_id=str(challenger_metrics["model_id"]),
        best_offline_model_id=str(best["model_id"]),
        runtime_selected_model_id=(
            str(challenger_metrics["model_id"]) if challenger_runtime_eligible else None
        ),
        runtime_selector_changed=False,
        num_models=2,
        num_eval_predictions=int(challenger_metrics["num_eval_predictions"]),
        num_eval_oracle_requests=int(challenger_metrics["num_eval_oracle_requests"]),
        min_required_eval_oracle_requests=min_eval_oracle_requests,
        min_runtime_oracle_top1_accuracy=min_runtime_oracle_top1_accuracy,
        min_runtime_profiled_success_rate=min_runtime_profiled_success_rate,
        challenger_beats_baseline=challenger_beats_baseline,
        challenger_runtime_eligible=challenger_runtime_eligible,
        challenger_gate_failures=challenger_failures,
        recommendation=(
            "keep_runtime_artifact_backed_and_expand_benchmark_coverage"
            if challenger_failures
            else "eligible_for_runtime_gating_experiment"
        ),
    )
    return CsrSelectorModelEvaluationExport(
        rows=(baseline_row, challenger_row),
        summary=summary,
        schema=_schema(),
    )


def write_csr_selector_model_eval_rows(
    rows: Iterable[CsrSelectorModelEvalRow],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_selector_model_eval_summary(
    summary: CsrSelectorModelEvaluationSummary,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_selector_model_eval_schema(schema: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_selector_model_eval_report(
    export: CsrSelectorModelEvaluationExport,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = export.summary
    lines = [
        "# CSR Selector Model Evaluation",
        "",
        f"- status: `{summary.status}`",
        f"- schema_version: `{summary.schema_version}`",
        f"- evaluation_id: `{summary.evaluation_id}`",
        f"- baseline_model_id: `{summary.baseline_model_id}`",
        f"- challenger_model_id: `{summary.challenger_model_id}`",
        f"- best_offline_model_id: `{summary.best_offline_model_id}`",
        f"- runtime_selected_model_id: `{summary.runtime_selected_model_id or ''}`",
        f"- runtime_selector_changed: `{summary.runtime_selector_changed}`",
        f"- num_eval_predictions: `{summary.num_eval_predictions}`",
        f"- num_eval_oracle_requests: `{summary.num_eval_oracle_requests}`",
        f"- min_required_eval_oracle_requests: `{summary.min_required_eval_oracle_requests}`",
        f"- min_runtime_oracle_top1_accuracy: `{summary.min_runtime_oracle_top1_accuracy:.6g}`",
        f"- min_runtime_profiled_success_rate: `{summary.min_runtime_profiled_success_rate:.6g}`",
        f"- challenger_beats_baseline: `{summary.challenger_beats_baseline}`",
        f"- challenger_runtime_eligible: `{summary.challenger_runtime_eligible}`",
        f"- challenger_gate_failures: `{list(summary.challenger_gate_failures)}`",
        f"- recommendation: `{summary.recommendation}`",
        "",
        "| role | model | oracle_top1 | success_rate | non_success | runtime_eligible | failures |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in export.rows:
        lines.append(
            "| "
            f"{row.comparison_role} | "
            f"{row.model_id} | "
            f"{row.eval_oracle_top1_accuracy:.6g} | "
            f"{row.eval_profiled_success_selection_rate:.6g} | "
            f"{row.eval_non_success_selection_count} | "
            f"{row.runtime_eligible} | "
            f"{', '.join(row.gate_failures)} |"
        )
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _baseline_metrics(
    summary: dict[str, Any],
    predictions: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    oracle_count = sum(1 for row in predictions if row.get("oracle_candidate_id") is not None)
    success_count = int(summary["num_eval_profiled_success_predictions"])
    return {
        "model_id": str(summary["baseline_id"]),
        "model_family": "deterministic_candidate_prior",
        "model_kind": "non_model_baseline",
        "model_trained": False,
        "runtime_selector_changed": False,
        "num_eval_predictions": int(summary["num_eval_predictions"]),
        "num_eval_oracle_requests": oracle_count,
        "num_profiled_success_predictions": success_count,
        "eval_oracle_top1_accuracy": float(summary["eval_oracle_top1_accuracy"]),
        "eval_profiled_success_selection_rate": float(summary["eval_profiled_success_rate"]),
        "eval_non_success_selection_count": int(summary["num_eval_predictions"]) - success_count,
        "eval_mean_regret_ms": _optional_float(summary.get("eval_mean_regret_ms")),
        "eval_max_regret_ms": _optional_float(summary.get("eval_max_regret_ms")),
    }


def _ranker_metrics(
    summary: dict[str, Any],
    predictions: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    eval_predictions = tuple(row for row in predictions if row["split"] == "eval")
    oracle_count = sum(1 for row in eval_predictions if row.get("oracle_candidate_id") is not None)
    success_count = sum(1 for row in eval_predictions if row["selected_target_status"] == "success")
    return {
        "model_id": str(summary["model_id"]),
        "model_family": str(summary["model_family"]),
        "model_kind": "trainable_pairwise_ranker",
        "model_trained": bool(summary["model_trained"]),
        "runtime_selector_changed": bool(summary["runtime_selector_changed"]),
        "num_eval_predictions": int(summary["num_eval_requests"]),
        "num_eval_oracle_requests": oracle_count,
        "num_profiled_success_predictions": success_count,
        "eval_oracle_top1_accuracy": float(summary["eval_oracle_top1_accuracy"]),
        "eval_profiled_success_selection_rate": float(
            summary["eval_profiled_success_selection_rate"]
        ),
        "eval_non_success_selection_count": int(summary["eval_non_success_selection_count"]),
        "eval_mean_regret_ms": _optional_float(summary.get("eval_mean_regret_ms")),
        "eval_max_regret_ms": _optional_float(summary.get("eval_max_regret_ms")),
    }


def _challenger_gate_failures(
    baseline: dict[str, Any],
    challenger: dict[str, Any],
    *,
    min_eval_oracle_requests: int,
    min_runtime_oracle_top1_accuracy: float,
    min_runtime_profiled_success_rate: float,
) -> tuple[str, ...]:
    failures: list[str] = []
    if not challenger["model_trained"]:
        failures.append("model_not_trained")
    if challenger["runtime_selector_changed"]:
        failures.append("runtime_selector_changed")
    if int(challenger["num_eval_oracle_requests"]) < min_eval_oracle_requests:
        failures.append(
            "insufficient_eval_oracle_requests:"
            f"{challenger['num_eval_oracle_requests']}<{min_eval_oracle_requests}"
        )
    if challenger["eval_oracle_top1_accuracy"] < min_runtime_oracle_top1_accuracy:
        failures.append("below_min_oracle_top1")
    if challenger["eval_profiled_success_selection_rate"] < min_runtime_profiled_success_rate:
        failures.append("below_min_profiled_success_rate")
    if challenger["eval_oracle_top1_accuracy"] < baseline["eval_oracle_top1_accuracy"]:
        failures.append("below_baseline_oracle_top1")
    if (
        challenger["eval_profiled_success_selection_rate"]
        < baseline["eval_profiled_success_selection_rate"]
    ):
        failures.append("below_baseline_profiled_success_rate")
    if int(challenger["eval_non_success_selection_count"]) > 0:
        failures.append("non_success_eval_selections")
    return tuple(failures)


def _beats_baseline(challenger: dict[str, Any], baseline: dict[str, Any]) -> bool:
    challenger_key = _ranking_key(challenger)
    baseline_key = _ranking_key(baseline)
    return challenger_key > baseline_key


def _best_model(rows: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    return max(rows, key=_ranking_key)


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, int, float]:
    max_regret = row["eval_max_regret_ms"]
    regret_penalty = 0.0 if max_regret is None else -float(max_regret)
    return (
        float(row["eval_oracle_top1_accuracy"]),
        float(row["eval_profiled_success_selection_rate"]),
        -int(row["eval_non_success_selection_count"]),
        regret_penalty,
    )


def _metrics_valid(row: dict[str, Any]) -> bool:
    for key in (
        "eval_oracle_top1_accuracy",
        "eval_profiled_success_selection_rate",
    ):
        value = float(row[key])
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            return False
    if int(row["num_eval_predictions"]) <= 0:
        return False
    if int(row["num_eval_oracle_requests"]) <= 0:
        return False
    if int(row["eval_non_success_selection_count"]) < 0:
        return False
    return True


def _row(
    metrics: dict[str, Any],
    *,
    comparison_role: str,
    beats_baseline: bool,
    runtime_eligible: bool,
    gate_failures: tuple[str, ...],
) -> CsrSelectorModelEvalRow:
    return CsrSelectorModelEvalRow(
        schema_version=CSR_SELECTOR_MODEL_EVAL_SCHEMA_VERSION,
        evaluation_id=EVALUATION_ID,
        model_id=str(metrics["model_id"]),
        model_family=str(metrics["model_family"]),
        model_kind=str(metrics["model_kind"]),
        comparison_role=comparison_role,
        model_trained=bool(metrics["model_trained"]),
        runtime_selector_changed=bool(metrics["runtime_selector_changed"]),
        num_eval_predictions=int(metrics["num_eval_predictions"]),
        num_eval_oracle_requests=int(metrics["num_eval_oracle_requests"]),
        num_profiled_success_predictions=int(metrics["num_profiled_success_predictions"]),
        eval_oracle_top1_accuracy=float(metrics["eval_oracle_top1_accuracy"]),
        eval_profiled_success_selection_rate=float(
            metrics["eval_profiled_success_selection_rate"]
        ),
        eval_non_success_selection_count=int(metrics["eval_non_success_selection_count"]),
        eval_mean_regret_ms=metrics["eval_mean_regret_ms"],
        eval_max_regret_ms=metrics["eval_max_regret_ms"],
        beats_baseline=beats_baseline,
        runtime_eligible=runtime_eligible,
        gate_failures=gate_failures,
    )


def _schema() -> dict[str, Any]:
    return {
        "schema_version": CSR_SELECTOR_MODEL_EVAL_SCHEMA_VERSION,
        "evaluation_id": EVALUATION_ID,
        "task": "compare_offline_csr_selector_models_before_runtime_integration",
        "runtime_selector_changed": False,
        "models": {
            "baseline": "candidate_prior_success_median_v1",
            "challenger": "pairwise_linear_ranker_v1",
        },
        "quality_gate": {
            "min_eval_oracle_requests_default": 10,
            "min_runtime_oracle_top1_accuracy_default": 0.5,
            "min_runtime_profiled_success_rate_default": 2.0 / 3.0,
            "requires_no_non_success_eval_selections": True,
            "requires_not_below_baseline": True,
        },
        "integration_boundary": {
            "status": "offline_quality_gate_only",
            "next_step": "expand_real_csr_benchmark_rows_before_transformer_or_runtime_learned_selector",
        },
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None
