"""Plan coverage expansion for guarded learned-policy CSR promotion."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


CSR_GUARDED_PROMOTION_COVERAGE_SCHEMA_VERSION = (
    "phase1_csr_guarded_promotion_coverage_plan_v1"
)


@dataclass(frozen=True)
class CsrGuardedPromotionCoverageScenario:
    schema_version: str
    plan_id: str
    scenario_id: str
    scenario_kind: str
    matrix_id: str
    context_id: str
    matrix_source_path: str | None
    selected_candidate_id: str
    selected_target_status: str | None
    selected_evaluation_status: str | None
    selected_confidence: float
    selected_score_margin: float | None
    selected_candidate_is_exact_profiled_success: bool
    artifact_candidate_id: str | None
    artifact_solver: str | None
    artifact_preconditioner: str | None
    artifact_precision: str | None
    artifact_candidate_count: int
    planned_runtime_candidate_id: str | None
    planned_runtime_solver: str | None
    planned_runtime_preconditioner: str | None
    planned_runtime_solver_parameters: dict[str, Any]
    fallback_candidate_ids: tuple[str, ...]
    requires_fixture_quality_gate: bool
    requires_fixture_prediction: bool
    requires_runtime_exception_injection: bool
    requires_gpu_execution: bool
    expected_guard_status: str
    expected_runtime_selection_source: str
    expected_final_result_status: str
    planned_status: str
    safety_assertions: tuple[str, ...]


@dataclass(frozen=True)
class CsrGuardedPromotionCoverageSummary:
    status: str
    schema_version: str
    plan_id: str
    source_selector_rows_path: str
    source_predictions_path: str
    source_quality_gate_summary_path: str
    source_csr_paths: tuple[str, ...]
    runtime_selector_changed: bool
    executes_gpu: bool
    model_id: str | None
    quality_gate_runtime_eligible: bool
    quality_gate_failures: tuple[str, ...]
    num_predictions: int
    profiled_success_predictions: int
    non_success_predictions: int
    exact_profiled_success_matrices: int
    planned_scenarios: int
    planned_gpu_final_solves: int
    planned_guard_only_scenarios: int
    current_blocked_scenarios: int
    fixture_promotion_scenarios: int
    fixture_promotions_changing_candidate: int
    non_success_block_scenarios: int
    runtime_fallback_scenarios: int
    by_scenario_kind: dict[str, int]
    by_selected_target_status: dict[str, int]
    by_runtime_solver: dict[str, int]
    selected_matrix_count: int
    max_current_blocked: int
    max_fixture_promotions: int
    max_non_success_blocks: int
    max_runtime_fallbacks: int
    next_step: str


@dataclass(frozen=True)
class CsrGuardedPromotionCoveragePlan:
    rows: tuple[CsrGuardedPromotionCoverageScenario, ...]
    summary: CsrGuardedPromotionCoverageSummary
    schema: dict[str, Any]


def build_csr_guarded_promotion_coverage_plan_from_files(
    selector_rows_path: str | Path = "runs/phase1_csr_transformer_ready/combined_csr_selector_rows.jsonl",
    predictions_path: str | Path = "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl",
    quality_gate_summary_path: str | Path = "runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    csr_paths: Iterable[str | Path] = (
        "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
        "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    ),
    *,
    plan_id: str = "phase1_csr_guarded_promotion_coverage_m57",
    max_current_blocked: int = 4,
    max_fixture_promotions: int = 4,
    max_non_success_blocks: int = 4,
    max_runtime_fallbacks: int = 3,
) -> CsrGuardedPromotionCoveragePlan:
    selector_rows = tuple(read_jsonl(selector_rows_path))
    predictions = tuple(read_jsonl(predictions_path))
    quality_gate_summary = json.loads(Path(quality_gate_summary_path).read_text(encoding="utf-8"))
    csr_source_map = _csr_source_map(csr_paths)
    return build_csr_guarded_promotion_coverage_plan(
        selector_rows,
        predictions,
        quality_gate_summary,
        csr_source_map=csr_source_map,
        source_selector_rows_path=str(selector_rows_path),
        source_predictions_path=str(predictions_path),
        source_quality_gate_summary_path=str(quality_gate_summary_path),
        source_csr_paths=tuple(str(path) for path in csr_paths),
        plan_id=plan_id,
        max_current_blocked=max_current_blocked,
        max_fixture_promotions=max_fixture_promotions,
        max_non_success_blocks=max_non_success_blocks,
        max_runtime_fallbacks=max_runtime_fallbacks,
    )


def build_csr_guarded_promotion_coverage_plan(
    selector_rows: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    quality_gate_summary: dict[str, Any],
    *,
    csr_source_map: dict[str, str],
    source_selector_rows_path: str,
    source_predictions_path: str,
    source_quality_gate_summary_path: str,
    source_csr_paths: tuple[str, ...],
    plan_id: str = "phase1_csr_guarded_promotion_coverage_m57",
    max_current_blocked: int = 4,
    max_fixture_promotions: int = 4,
    max_non_success_blocks: int = 4,
    max_runtime_fallbacks: int = 3,
) -> CsrGuardedPromotionCoveragePlan:
    selector = tuple(selector_rows)
    prediction_rows = tuple(predictions)
    success_by_matrix = _success_profiles_by_matrix(selector)
    success_by_key = {
        (row["matrix_id"], row["context_id"], row["candidate_id"]): row
        for rows in success_by_matrix.values()
        for row in rows
    }

    current_blocked = _current_blocked_scenarios(
        prediction_rows,
        success_by_matrix=success_by_matrix,
        success_by_key=success_by_key,
        csr_source_map=csr_source_map,
        plan_id=plan_id,
        limit=max_current_blocked,
    )
    fixture_promotions = _fixture_promotion_scenarios(
        prediction_rows,
        success_by_matrix=success_by_matrix,
        success_by_key=success_by_key,
        csr_source_map=csr_source_map,
        plan_id=plan_id,
        limit=max_fixture_promotions,
    )
    non_success_blocks = _non_success_block_scenarios(
        prediction_rows,
        success_by_matrix=success_by_matrix,
        csr_source_map=csr_source_map,
        plan_id=plan_id,
        limit=max_non_success_blocks,
    )
    runtime_fallbacks = _runtime_fallback_scenarios(
        success_by_matrix,
        csr_source_map=csr_source_map,
        plan_id=plan_id,
        limit=max_runtime_fallbacks,
    )
    rows = (
        *current_blocked,
        *fixture_promotions,
        *non_success_blocks,
        *runtime_fallbacks,
    )
    summary = _summary(
        rows,
        prediction_rows,
        success_by_matrix,
        quality_gate_summary,
        source_selector_rows_path=source_selector_rows_path,
        source_predictions_path=source_predictions_path,
        source_quality_gate_summary_path=source_quality_gate_summary_path,
        source_csr_paths=source_csr_paths,
        plan_id=plan_id,
        max_current_blocked=max_current_blocked,
        max_fixture_promotions=max_fixture_promotions,
        max_non_success_blocks=max_non_success_blocks,
        max_runtime_fallbacks=max_runtime_fallbacks,
    )
    return CsrGuardedPromotionCoveragePlan(
        rows=rows,
        summary=summary,
        schema=_schema(),
    )


def write_csr_guarded_promotion_coverage_rows(
    rows: Iterable[CsrGuardedPromotionCoverageScenario],
    path: str | Path,
) -> Path:
    return write_jsonl((asdict(row) for row in rows), path)


def write_csr_guarded_promotion_coverage_summary(
    summary: CsrGuardedPromotionCoverageSummary,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_guarded_promotion_coverage_schema(
    schema: dict[str, Any],
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csr_guarded_promotion_coverage_report(
    plan: CsrGuardedPromotionCoveragePlan,
    path: str | Path,
) -> Path:
    summary = plan.summary
    lines = [
        "# CSR Guarded Promotion Coverage Plan",
        "",
        f"- status: `{summary.status}`",
        f"- schema_version: `{summary.schema_version}`",
        f"- scenarios: `{summary.planned_scenarios}`",
        f"- planned_gpu_final_solves: `{summary.planned_gpu_final_solves}`",
        f"- planned_guard_only_scenarios: `{summary.planned_guard_only_scenarios}`",
        f"- current_blocked_scenarios: `{summary.current_blocked_scenarios}`",
        f"- fixture_promotion_scenarios: `{summary.fixture_promotion_scenarios}`",
        f"- non_success_block_scenarios: `{summary.non_success_block_scenarios}`",
        f"- runtime_fallback_scenarios: `{summary.runtime_fallback_scenarios}`",
        f"- quality_gate_runtime_eligible: `{summary.quality_gate_runtime_eligible}`",
        f"- runtime_selector_changed: `{summary.runtime_selector_changed}`",
        f"- executes_gpu: `{summary.executes_gpu}`",
        "",
        "| scenario | kind | matrix | selected | runtime | gpu | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in plan.rows:
        lines.append(
            "| "
            f"{row.scenario_id} | "
            f"{row.scenario_kind} | "
            f"{row.matrix_id} | "
            f"{row.selected_candidate_id} | "
            f"{row.planned_runtime_candidate_id or ''} | "
            f"{row.requires_gpu_execution} | "
            f"{row.planned_status} |"
        )
    lines.append("")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _current_blocked_scenarios(
    predictions: tuple[dict[str, Any], ...],
    *,
    success_by_matrix: dict[str, tuple[dict[str, Any], ...]],
    success_by_key: dict[tuple[str, str, str], dict[str, Any]],
    csr_source_map: dict[str, str],
    plan_id: str,
    limit: int,
) -> tuple[CsrGuardedPromotionCoverageScenario, ...]:
    candidates = [
        (prediction, *_prediction_confidence(prediction))
        for prediction in predictions
        if prediction.get("selected_target_status") == "success"
        and _prediction_key(prediction) in success_by_key
        and str(prediction["matrix_id"]) in csr_source_map
    ]
    candidates.sort(key=lambda item: (-item[1], str(item[0]["matrix_id"])))
    return tuple(
        _scenario_from_prediction(
            prediction,
            success_by_matrix=success_by_matrix,
            success_by_key=success_by_key,
            csr_source_map=csr_source_map,
            plan_id=plan_id,
            scenario_id=f"m57_current_blocked_{index:02d}",
            scenario_kind="current_ranker_quality_gate_block",
            selected_confidence=confidence,
            selected_score_margin=margin,
            expected_guard_status="blocked_quality_gate",
            expected_runtime_selection_source="artifact",
            expected_final_result_status="success",
            requires_fixture_quality_gate=False,
            requires_fixture_prediction=False,
            requires_runtime_exception_injection=False,
            requires_gpu_execution=True,
            planned_status="queued_current_gate_block_gpu_smoke",
            safety_assertions=(
                "real_quality_gate_blocks_current_ranker",
                "artifact_candidate_remains_runtime_source",
                "fallback_chain_uses_exact_profiled_success_rows",
            ),
        )
        for index, (prediction, confidence, margin) in enumerate(candidates[:limit], start=1)
    )


def _fixture_promotion_scenarios(
    predictions: tuple[dict[str, Any], ...],
    *,
    success_by_matrix: dict[str, tuple[dict[str, Any], ...]],
    success_by_key: dict[tuple[str, str, str], dict[str, Any]],
    csr_source_map: dict[str, str],
    plan_id: str,
    limit: int,
) -> tuple[CsrGuardedPromotionCoverageScenario, ...]:
    items = []
    for prediction in predictions:
        key = _prediction_key(prediction)
        if (
            prediction.get("selected_target_status") == "success"
            and key in success_by_key
            and str(prediction["matrix_id"]) in csr_source_map
        ):
            artifact = _best_success_profile(
                success_by_matrix[str(prediction["matrix_id"])],
                context_id=str(prediction["context_id"]),
            )
            confidence, margin = _prediction_confidence(prediction)
            changes_candidate = artifact is not None and artifact["candidate_id"] != prediction["selected_candidate_id"]
            items.append((prediction, confidence, margin, changes_candidate))
    items.sort(key=lambda item: (not item[3], -item[1], str(item[0]["matrix_id"])))
    return tuple(
        _scenario_from_prediction(
            prediction,
            success_by_matrix=success_by_matrix,
            success_by_key=success_by_key,
            csr_source_map=csr_source_map,
            plan_id=plan_id,
            scenario_id=f"m57_fixture_promotion_{index:02d}",
            scenario_kind="fixture_profiled_success_promotion",
            selected_confidence=confidence,
            selected_score_margin=margin,
            expected_guard_status="promoted",
            expected_runtime_selection_source="learned",
            expected_final_result_status="success",
            requires_fixture_quality_gate=True,
            requires_fixture_prediction=True,
            requires_runtime_exception_injection=False,
            requires_gpu_execution=True,
            planned_status="queued_fixture_promotion_gpu_smoke",
            safety_assertions=(
                "fixture_quality_gate_must_be_runtime_eligible",
                "learned_candidate_must_be_exact_profiled_success",
                "fallback_chain_uses_exact_profiled_success_rows",
            ),
        )
        for index, (prediction, confidence, margin, _) in enumerate(items[:limit], start=1)
    )


def _non_success_block_scenarios(
    predictions: tuple[dict[str, Any], ...],
    *,
    success_by_matrix: dict[str, tuple[dict[str, Any], ...]],
    csr_source_map: dict[str, str],
    plan_id: str,
    limit: int,
) -> tuple[CsrGuardedPromotionCoverageScenario, ...]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        status = str(prediction.get("selected_target_status"))
        if status != "success" and str(prediction["matrix_id"]) in csr_source_map:
            grouped[status].append(prediction)
    for status_rows in grouped.values():
        status_rows.sort(key=lambda row: str(row["matrix_id"]))
    ordered = []
    status_order = ("screened_out", "not_profiled", "not_applicable")
    while len(ordered) < limit and any(grouped.values()):
        for status in status_order:
            if grouped[status] and len(ordered) < limit:
                ordered.append(grouped[status].pop(0))
    rows = []
    for index, prediction in enumerate(ordered, start=1):
        confidence, margin = _prediction_confidence(prediction)
        rows.append(
            _scenario_from_prediction(
                prediction,
                success_by_matrix=success_by_matrix,
                success_by_key={},
                csr_source_map=csr_source_map,
                plan_id=plan_id,
                scenario_id=f"m57_non_success_block_{index:02d}",
                scenario_kind="fixture_non_success_candidate_block",
                selected_confidence=confidence,
                selected_score_margin=margin,
                expected_guard_status="blocked_non_success_candidate",
                expected_runtime_selection_source="artifact",
                expected_final_result_status="guard_plan_only",
                requires_fixture_quality_gate=True,
                requires_fixture_prediction=True,
                requires_runtime_exception_injection=False,
                requires_gpu_execution=False,
                planned_status="queued_guard_only_no_exact_profiled_fallback",
                safety_assertions=(
                    "fixture_quality_gate_reaches_candidate_validation",
                    "non_success_learned_candidate_must_not_promote",
                    "gpu_execution_requires_exact_profiled_fallback_before_enablement",
                ),
            )
        )
    return tuple(rows)


def _runtime_fallback_scenarios(
    success_by_matrix: dict[str, tuple[dict[str, Any], ...]],
    *,
    csr_source_map: dict[str, str],
    plan_id: str,
    limit: int,
) -> tuple[CsrGuardedPromotionCoverageScenario, ...]:
    candidates = [
        (matrix_id, rows)
        for matrix_id, rows in success_by_matrix.items()
        if len(rows) >= 2 and matrix_id in csr_source_map
    ]
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    scenarios: list[CsrGuardedPromotionCoverageScenario] = []
    for index, (matrix_id, rows) in enumerate(candidates[:limit], start=1):
        context_id = str(rows[0]["context_id"])
        artifact = _best_success_profile(rows, context_id=context_id)
        assert artifact is not None
        fallback_ids = tuple(
            row["candidate_id"]
            for row in sorted(rows, key=_profile_sort_key)
            if row["candidate_id"] != artifact["candidate_id"]
        )
        scenarios.append(
            CsrGuardedPromotionCoverageScenario(
                schema_version=CSR_GUARDED_PROMOTION_COVERAGE_SCHEMA_VERSION,
                plan_id=plan_id,
                scenario_id=f"m57_runtime_fallback_{index:02d}",
                scenario_kind="runtime_exception_fallback",
                matrix_id=matrix_id,
                context_id=context_id,
                matrix_source_path=csr_source_map.get(matrix_id),
                selected_candidate_id="injected_invalid_primary",
                selected_target_status=None,
                selected_evaluation_status=None,
                selected_confidence=0.0,
                selected_score_margin=None,
                selected_candidate_is_exact_profiled_success=False,
                artifact_candidate_id=artifact["candidate_id"],
                artifact_solver=artifact["solver"],
                artifact_preconditioner=artifact["preconditioner"],
                artifact_precision=artifact["precision"],
                artifact_candidate_count=len(rows),
                planned_runtime_candidate_id=artifact["candidate_id"],
                planned_runtime_solver=artifact["solver"],
                planned_runtime_preconditioner=artifact["preconditioner"],
                planned_runtime_solver_parameters=dict(artifact.get("solver_parameters", {})),
                fallback_candidate_ids=fallback_ids,
                requires_fixture_quality_gate=False,
                requires_fixture_prediction=False,
                requires_runtime_exception_injection=True,
                requires_gpu_execution=True,
                expected_guard_status="not_applicable_runtime_guard_only",
                expected_runtime_selection_source="runtime_guard_fallback",
                expected_final_result_status="fallback_success",
                planned_status="queued_runtime_exception_fallback_gpu_smoke",
                safety_assertions=(
                    "first_attempt_must_record_exception",
                    "fallback_candidate_must_be_exact_profiled_success",
                    "final_result_must_be_taichi_gpu_success",
                ),
            )
        )
    return tuple(scenarios)


def _scenario_from_prediction(
    prediction: dict[str, Any],
    *,
    success_by_matrix: dict[str, tuple[dict[str, Any], ...]],
    success_by_key: dict[tuple[str, str, str], dict[str, Any]],
    csr_source_map: dict[str, str],
    plan_id: str,
    scenario_id: str,
    scenario_kind: str,
    selected_confidence: float,
    selected_score_margin: float | None,
    expected_guard_status: str,
    expected_runtime_selection_source: str,
    expected_final_result_status: str,
    requires_fixture_quality_gate: bool,
    requires_fixture_prediction: bool,
    requires_runtime_exception_injection: bool,
    requires_gpu_execution: bool,
    planned_status: str,
    safety_assertions: tuple[str, ...],
) -> CsrGuardedPromotionCoverageScenario:
    matrix_id = str(prediction["matrix_id"])
    context_id = str(prediction["context_id"])
    selected_id = str(prediction["selected_candidate_id"])
    success_rows = success_by_matrix.get(matrix_id, ())
    artifact = _best_success_profile(success_rows, context_id=context_id)
    selected_profile = success_by_key.get((matrix_id, context_id, selected_id))
    runtime_profile = selected_profile if expected_runtime_selection_source == "learned" else artifact
    fallback_ids = ()
    if runtime_profile is not None:
        fallback_ids = tuple(
            row["candidate_id"]
            for row in sorted(success_rows, key=_profile_sort_key)
            if row["candidate_id"] != runtime_profile["candidate_id"]
        )
    return CsrGuardedPromotionCoverageScenario(
        schema_version=CSR_GUARDED_PROMOTION_COVERAGE_SCHEMA_VERSION,
        plan_id=plan_id,
        scenario_id=scenario_id,
        scenario_kind=scenario_kind,
        matrix_id=matrix_id,
        context_id=context_id,
        matrix_source_path=csr_source_map.get(matrix_id),
        selected_candidate_id=selected_id,
        selected_target_status=(
            None
            if prediction.get("selected_target_status") is None
            else str(prediction.get("selected_target_status"))
        ),
        selected_evaluation_status=(
            None
            if prediction.get("evaluation_status") is None
            else str(prediction.get("evaluation_status"))
        ),
        selected_confidence=selected_confidence,
        selected_score_margin=selected_score_margin,
        selected_candidate_is_exact_profiled_success=selected_profile is not None,
        artifact_candidate_id=None if artifact is None else str(artifact["candidate_id"]),
        artifact_solver=None if artifact is None else str(artifact["solver"]),
        artifact_preconditioner=None if artifact is None else str(artifact["preconditioner"]),
        artifact_precision=None if artifact is None else str(artifact["precision"]),
        artifact_candidate_count=len(success_rows),
        planned_runtime_candidate_id=(
            None if runtime_profile is None else str(runtime_profile["candidate_id"])
        ),
        planned_runtime_solver=None if runtime_profile is None else str(runtime_profile["solver"]),
        planned_runtime_preconditioner=(
            None if runtime_profile is None else str(runtime_profile["preconditioner"])
        ),
        planned_runtime_solver_parameters=(
            {} if runtime_profile is None else dict(runtime_profile.get("solver_parameters", {}))
        ),
        fallback_candidate_ids=fallback_ids,
        requires_fixture_quality_gate=requires_fixture_quality_gate,
        requires_fixture_prediction=requires_fixture_prediction,
        requires_runtime_exception_injection=requires_runtime_exception_injection,
        requires_gpu_execution=requires_gpu_execution,
        expected_guard_status=expected_guard_status,
        expected_runtime_selection_source=expected_runtime_selection_source,
        expected_final_result_status=expected_final_result_status,
        planned_status=planned_status,
        safety_assertions=safety_assertions,
    )


def _summary(
    rows: tuple[CsrGuardedPromotionCoverageScenario, ...],
    predictions: tuple[dict[str, Any], ...],
    success_by_matrix: dict[str, tuple[dict[str, Any], ...]],
    quality_gate_summary: dict[str, Any],
    *,
    source_selector_rows_path: str,
    source_predictions_path: str,
    source_quality_gate_summary_path: str,
    source_csr_paths: tuple[str, ...],
    plan_id: str,
    max_current_blocked: int,
    max_fixture_promotions: int,
    max_non_success_blocks: int,
    max_runtime_fallbacks: int,
) -> CsrGuardedPromotionCoverageSummary:
    by_kind = Counter(row.scenario_kind for row in rows)
    by_status = Counter(
        row.selected_target_status or "none" for row in rows if row.selected_target_status is not None
    )
    by_solver = Counter(
        row.planned_runtime_solver
        for row in rows
        if row.planned_runtime_solver is not None and row.requires_gpu_execution
    )
    profiled_predictions = sum(
        1 for row in predictions if row.get("selected_target_status") == "success"
    )
    non_success_predictions = len(predictions) - profiled_predictions
    fixture_promotions = by_kind["fixture_profiled_success_promotion"]
    fixture_changes = sum(
        1
        for row in rows
        if row.scenario_kind == "fixture_profiled_success_promotion"
        and row.artifact_candidate_id != row.selected_candidate_id
    )
    status = (
        "passed"
        if rows
        and by_kind["current_ranker_quality_gate_block"] >= 1
        and fixture_promotions >= 1
        and fixture_changes >= 1
        and by_kind["fixture_non_success_candidate_block"] >= 3
        and by_kind["runtime_exception_fallback"] >= 1
        and all(row.matrix_source_path for row in rows)
        and all(
            row.artifact_candidate_id is not None
            for row in rows
            if row.requires_gpu_execution
        )
        else "failed"
    )
    return CsrGuardedPromotionCoverageSummary(
        status=status,
        schema_version=CSR_GUARDED_PROMOTION_COVERAGE_SCHEMA_VERSION,
        plan_id=plan_id,
        source_selector_rows_path=source_selector_rows_path,
        source_predictions_path=source_predictions_path,
        source_quality_gate_summary_path=source_quality_gate_summary_path,
        source_csr_paths=source_csr_paths,
        runtime_selector_changed=False,
        executes_gpu=False,
        model_id=(
            None if not predictions else str(predictions[0].get("model_id"))
        ),
        quality_gate_runtime_eligible=bool(
            quality_gate_summary.get("challenger_runtime_eligible")
        ),
        quality_gate_failures=tuple(
            str(item) for item in quality_gate_summary.get("challenger_gate_failures", ())
        ),
        num_predictions=len(predictions),
        profiled_success_predictions=profiled_predictions,
        non_success_predictions=non_success_predictions,
        exact_profiled_success_matrices=len(success_by_matrix),
        planned_scenarios=len(rows),
        planned_gpu_final_solves=sum(1 for row in rows if row.requires_gpu_execution),
        planned_guard_only_scenarios=sum(1 for row in rows if not row.requires_gpu_execution),
        current_blocked_scenarios=by_kind["current_ranker_quality_gate_block"],
        fixture_promotion_scenarios=fixture_promotions,
        fixture_promotions_changing_candidate=fixture_changes,
        non_success_block_scenarios=by_kind["fixture_non_success_candidate_block"],
        runtime_fallback_scenarios=by_kind["runtime_exception_fallback"],
        by_scenario_kind=dict(by_kind),
        by_selected_target_status=dict(by_status),
        by_runtime_solver={str(key): value for key, value in by_solver.items()},
        selected_matrix_count=len({row.matrix_id for row in rows}),
        max_current_blocked=max_current_blocked,
        max_fixture_promotions=max_fixture_promotions,
        max_non_success_blocks=max_non_success_blocks,
        max_runtime_fallbacks=max_runtime_fallbacks,
        next_step="execute_selected_m57_scenarios_as_light_gpu_guarded_runtime_coverage",
    )


def _schema() -> dict[str, Any]:
    return {
        "schema_version": CSR_GUARDED_PROMOTION_COVERAGE_SCHEMA_VERSION,
        "task": "plan_guarded_learned_policy_runtime_coverage_expansion",
        "integration_boundary": {
            "status": "plan_only_no_gpu_execution",
            "runtime_selector_changed": False,
            "executes_gpu": False,
        },
        "scenario_kinds": {
            "current_ranker_quality_gate_block": "real current quality gate must block current learned ranker",
            "fixture_profiled_success_promotion": "fixture eligible gate proves promotion to exact profiled-success candidate",
            "fixture_non_success_candidate_block": "fixture eligible gate reaches candidate validation and blocks non-success learned candidate",
            "runtime_exception_fallback": "runtime guard retries exact profiled-success fallback after injected exception",
        },
        "safety_rules": [
            "current learned model remains artifact-backed until quality gate is runtime eligible",
            "fixture promotions must reference exact matrix/context profiled-success candidates",
            "non-success learned candidates are guard-only unless exact profiled fallback coverage exists",
            "runtime fallback scenarios must end on exact profiled-success artifact candidates",
        ],
    }


def _csr_source_map(csr_paths: Iterable[str | Path]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for path in csr_paths:
        source = str(path)
        file_path = Path(path)
        if not file_path.exists():
            continue
        for row in read_jsonl(file_path):
            sources.setdefault(str(row["matrix_id"]), source)
    return sources


def _success_profiles_by_matrix(
    selector_rows: tuple[dict[str, Any], ...]
) -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selector_rows:
        if str(row.get("target_status")) == "success" and float(row.get("target_success_rate", 0.0)) >= 1.0:
            grouped[str(row["matrix_id"])].append(row)
    return {
        matrix_id: tuple(sorted(rows, key=_profile_sort_key))
        for matrix_id, rows in grouped.items()
    }


def _best_success_profile(
    rows: tuple[dict[str, Any], ...],
    *,
    context_id: str,
) -> dict[str, Any] | None:
    matching = tuple(row for row in rows if str(row["context_id"]) == context_id)
    if not matching:
        return None
    return sorted(matching, key=_profile_sort_key)[0]


def _profile_sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
    solve_time = row.get("target_median_solve_time_ms")
    if solve_time is None:
        solve_time = row.get("target_solve_time_ms")
    iqr = row.get("target_solve_time_iqr_ms")
    return (
        float("inf") if solve_time is None else float(solve_time),
        0.0 if iqr is None else float(iqr),
        str(row["candidate_id"]),
    )


def _prediction_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["matrix_id"]),
        str(row["context_id"]),
        str(row["selected_candidate_id"]),
    )


def _prediction_confidence(row: dict[str, Any]) -> tuple[float, float | None]:
    scores = {str(key): float(value) for key, value in row.get("scores", {}).items()}
    ranked = tuple(str(item) for item in row.get("ranked_candidate_ids", ()))
    if not ranked:
        return 0.0, None
    if len(ranked) == 1:
        return 1.0, math.inf
    top = ranked[0]
    second = ranked[1]
    if top not in scores or second not in scores:
        return 0.0, None
    margin = scores[top] - scores[second]
    if not math.isfinite(margin):
        return 0.0, margin
    confidence = 1.0 / (1.0 + math.exp(-max(min(margin, 60.0), -60.0)))
    return confidence, margin
