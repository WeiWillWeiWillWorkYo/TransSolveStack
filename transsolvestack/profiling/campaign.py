"""Phase campaign artifact aggregation and readiness reporting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


@dataclass(frozen=True)
class CampaignStageReport:
    stage_id: str
    artifact_kind: str
    artifact_dir: str
    status: str
    row_count: int
    stale_paths: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CampaignReport:
    campaign_id: str
    status: str
    stages: tuple[CampaignStageReport, ...]

    @property
    def num_stages(self) -> int:
        return len(self.stages)


def build_phase1_campaign_report(root: str | Path = "runs") -> CampaignReport:
    runs = Path(root)
    queue_batch_stages = _queue_batch_stages(runs)
    stages = (
        _stage(
            stage_id="benchmark",
            artifact_dir=runs / "phase1_smoke_taichi",
            row_file="candidate_performance.jsonl",
        ),
        _stage(
            stage_id="expanded_solver_benchmark",
            artifact_dir=runs / "phase1_smoke_expanded_solvers",
            row_file="candidate_performance.jsonl",
        ),
        _stage(
            stage_id="regression",
            artifact_dir=runs / "phase1_numerical_regression",
            row_file="numerical_regression.jsonl",
        ),
        _stage(
            stage_id="sequence",
            artifact_dir=runs / "phase1_sequence_taichi",
            row_file="sequence_steps.jsonl",
        ),
        _stage(
            stage_id="policy_selection",
            artifact_dir=runs / "phase1_policy_selection",
            row_file="selected_policy_plans.jsonl",
        ),
        _stage(
            stage_id="policy_solve",
            artifact_dir=runs / "phase1_policy_solve",
            row_file="policy_solve_results.jsonl",
        ),
        _stage(
            stage_id="dataset_plan",
            artifact_dir=runs / "phase1_dataset_plan",
            row_file="download_plan.jsonl",
        ),
        _stage(
            stage_id="matrix_market_probe",
            artifact_dir=runs / "phase1_matrix_market_probe",
            row_file="matrix_metadata.jsonl",
        ),
        _stage(
            stage_id="suitesparse_selection",
            artifact_dir=runs / "phase1_suitesparse_selection",
            row_file="selected_matrices.jsonl",
        ),
        _stage(
            stage_id="suitesparse_header_probe",
            artifact_dir=runs / "phase1_suitesparse_header_probe",
            row_file="archive_header_probe.jsonl",
        ),
        _stage(
            stage_id="suitesparse_csr_import",
            artifact_dir=runs / "phase1_suitesparse_csr_import",
            row_file="csr_matrices.jsonl",
        ),
        _stage(
            stage_id="taichi_csr_matvec",
            artifact_dir=runs / "phase1_taichi_csr_matvec",
            row_file="csr_matvec_results.jsonl",
        ),
        _stage(
            stage_id="taichi_csr_primitives",
            artifact_dir=runs / "phase1_taichi_csr_primitives",
            row_file="csr_primitives_results.jsonl",
        ),
        _stage(
            stage_id="taichi_csr_solve",
            artifact_dir=runs / "phase1_taichi_csr_solve",
            row_file="csr_solve_results.jsonl",
        ),
        _stage(
            stage_id="taichi_csr_bicgstab",
            artifact_dir=runs / "phase1_taichi_csr_bicgstab",
            row_file="csr_bicgstab_results.jsonl",
        ),
        _stage(
            stage_id="taichi_csr_gmres",
            artifact_dir=runs / "phase1_taichi_csr_gmres",
            row_file="csr_gmres_results.jsonl",
        ),
        _stage(
            stage_id="taichi_csr_richardson",
            artifact_dir=runs / "phase1_taichi_csr_richardson",
            row_file="csr_richardson_results.jsonl",
        ),
        _stage(
            stage_id="taichi_csr_chebyshev",
            artifact_dir=runs / "phase1_taichi_csr_chebyshev",
            row_file="csr_chebyshev_results.jsonl",
        ),
        _stage(
            stage_id="taichi_csr_symmetric_equilibration",
            artifact_dir=runs / "phase1_taichi_csr_symmetric_equilibration",
            row_file="csr_symmetric_equilibration_results.jsonl",
        ),
        _stage(
            stage_id="taichi_csr_row_column_equilibration",
            artifact_dir=runs / "phase1_taichi_csr_row_column_equilibration",
            row_file="csr_row_column_equilibration_results.jsonl",
        ),
        _stage(
            stage_id="taichi_csr_ilu0",
            artifact_dir=runs / "phase1_taichi_csr_ilu0",
            row_file="csr_ilu0_results.jsonl",
        ),
        _stage(
            stage_id="csr_selector_readiness",
            artifact_dir=runs / "phase1_csr_selector_readiness",
            row_file="csr_selector_rows.jsonl",
        ),
        _stage(
            stage_id="csr_selector_policy",
            artifact_dir=runs / "phase1_csr_selector_policy",
            row_file="csr_selected_policy_plans.jsonl",
        ),
        _stage(
            stage_id="csr_auto_solve",
            artifact_dir=runs / "phase1_csr_auto_solve",
            row_file="csr_auto_solve_results.jsonl",
        ),
        _stage(
            stage_id="csr_learning_readiness",
            artifact_dir=runs / "phase1_csr_learning_readiness",
            row_file="csr_learning_rows.jsonl",
        ),
        _stage(
            stage_id="csr_model_contract",
            artifact_dir=runs / "phase1_csr_model_contract",
            row_file="csr_model_requests.jsonl",
        ),
        _stage(
            stage_id="csr_training_tensors",
            artifact_dir=runs / "phase1_csr_training_tensors",
            row_file="csr_training_request_index.jsonl",
        ),
        _stage(
            stage_id="csr_linear_ranker",
            artifact_dir=runs / "phase1_csr_linear_ranker",
            row_file="csr_linear_ranker_predictions.jsonl",
        ),
        _stage(
            stage_id="csr_selector_model_eval",
            artifact_dir=runs / "phase1_csr_selector_model_eval",
            row_file="csr_selector_model_eval_rows.jsonl",
        ),
        _stage(
            stage_id="csr_benchmark_expansion_plan",
            artifact_dir=runs / "phase1_csr_benchmark_expansion_plan",
            row_file="csr_benchmark_candidate_queue.jsonl",
        ),
        _stage(
            stage_id="csr_full_dataset_queue",
            artifact_dir=runs / "phase1_csr_full_dataset_queue",
            row_file="csr_full_dataset_queue_jobs.jsonl",
        ),
        *queue_batch_stages,
        _stage(
            stage_id="csr_queue_training_pool",
            artifact_dir=runs / "phase1_csr_queue_training_pool",
            row_file="csr_queue_training_pool_selector_rows.jsonl",
        ),
        _stage(
            stage_id="csr_queue_batch_training_bundle",
            artifact_dir=runs / "phase1_csr_queue_batch_training_bundle",
            row_file="csr_transformer_request_index.jsonl",
        ),
        _stage(
            stage_id="csr_queue_batch_reference_ranker",
            artifact_dir=runs / "phase1_csr_queue_batch_reference_ranker",
            row_file="csr_queue_batch_reference_ranker_predictions.jsonl",
        ),
        _stage(
            stage_id="csr_queue_batch_model_replay",
            artifact_dir=runs / "phase1_csr_queue_batch_model_replay",
            row_file="csr_queue_batch_model_replay_comparison.jsonl",
        ),
        _stage(
            stage_id="csr_queue_candidate_coverage",
            artifact_dir=runs / "phase1_csr_queue_candidate_coverage",
            row_file="csr_queue_candidate_coverage_gap_rows.jsonl",
        ),
        _stage(
            stage_id="csr_gmres_restart_coverage",
            artifact_dir=runs / "phase1_csr_gmres_restart_coverage",
            row_file="csr_micro_campaign_results.jsonl",
        ),
        _stage(
            stage_id="csr_blocked_gap_probe",
            artifact_dir=runs / "phase1_csr_blocked_gap_probe",
            row_file="csr_micro_campaign_results.jsonl",
        ),
        _stage(
            stage_id="csr_blocked_gap_positive_search",
            artifact_dir=runs / "phase1_csr_blocked_gap_positive_search",
            row_file="csr_blocked_gap_positive_results.jsonl",
        ),
        _stage(
            stage_id="csr_blocked_gap_training_integration",
            artifact_dir=runs / "phase1_csr_blocked_gap_training_integration",
            row_file="combined_csr_selector_rows.jsonl",
        ),
        _stage(
            stage_id="csr_blocked_gap_augmented_ranker",
            artifact_dir=runs / "phase1_csr_blocked_gap_augmented_ranker",
            row_file="csr_blocked_gap_augmented_ranker_predictions.jsonl",
        ),
        _stage(
            stage_id="csr_blocked_gap_guarded_replay",
            artifact_dir=runs / "phase1_csr_blocked_gap_guarded_replay",
            row_file="csr_blocked_gap_guarded_replay_rows.jsonl",
        ),
        _stage(
            stage_id="csr_transformer_handoff_bundle",
            artifact_dir=runs / "phase1_csr_transformer_handoff_bundle",
            row_file="csr_transformer_handoff_bundle_rows.jsonl",
        ),
        _stage(
            stage_id="csr_transformer_training_package",
            artifact_dir=runs / "phase1_csr_transformer_training_package",
            row_file="csr_transformer_training_package_rows.jsonl",
        ),
        _stage(
            stage_id="csr_transformer_package_consumer_dry_run",
            artifact_dir=runs / "phase1_csr_transformer_package_consumer_dry_run",
            row_file="csr_transformer_package_consumer_dry_run_rows.jsonl",
        ),
        _stage(
            stage_id="csr_micro_campaign",
            artifact_dir=runs / "phase1_csr_micro_campaign",
            row_file="csr_micro_selector_rows.jsonl",
        ),
        _stage(
            stage_id="csr_transformer_ready",
            artifact_dir=runs / "phase1_csr_transformer_ready",
            row_file="csr_transformer_request_index.jsonl",
        ),
        _stage(
            stage_id="csr_transformer_ranker",
            artifact_dir=runs / "phase1_csr_transformer_ranker",
            row_file="csr_transformer_ranker_predictions.jsonl",
        ),
        _stage(
            stage_id="csr_external_model_adapter",
            artifact_dir=runs / "phase1_csr_external_model_adapter",
            row_file="csr_external_model_adapter_rows.jsonl",
        ),
        _stage(
            stage_id="csr_transformer_model_replay",
            artifact_dir=runs / "phase1_csr_transformer_model_replay",
            row_file="csr_transformer_model_replay_predictions.jsonl",
        ),
        _stage(
            stage_id="csr_transformer_quality_gate",
            artifact_dir=runs / "phase1_csr_transformer_quality_gate",
            row_file="csr_transformer_quality_gate_rows.jsonl",
        ),
        _stage(
            stage_id="csr_policy_model_artifact",
            artifact_dir=runs / "phase1_csr_policy_model_artifact",
            row_file="csr_policy_model_artifact_rows.jsonl",
        ),
        _stage(
            stage_id="csr_policy_model_acceptance",
            artifact_dir=runs / "phase1_csr_policy_model_acceptance",
            row_file="csr_policy_model_acceptance_rows.jsonl",
        ),
        _stage(
            stage_id="csr_policy_model_submission",
            artifact_dir=runs / "phase1_csr_policy_model_submission",
            row_file="csr_policy_model_submission_rows.jsonl",
        ),
        _stage(
            stage_id="csr_external_model_intake",
            artifact_dir=runs / "phase1_csr_external_model_intake",
            row_file="csr_external_model_intake_rows.jsonl",
        ),
        _stage(
            stage_id="csr_transformer_training_entrypoint",
            artifact_dir=runs / "phase1_csr_transformer_training_entrypoint",
            row_file="csr_transformer_training_entrypoint_rows.jsonl",
        ),
        _stage(
            stage_id="csr_transformer_reference_training_export",
            artifact_dir=runs / "phase1_csr_transformer_reference_training_export",
            row_file="csr_transformer_reference_training_export_rows.jsonl",
        ),
        _stage(
            stage_id="csr_learned_runtime_guard",
            artifact_dir=runs / "phase1_csr_learned_guard",
            row_file="csr_learned_guard_rows.jsonl",
        ),
        _stage(
            stage_id="csr_guarded_auto_solve",
            artifact_dir=runs / "phase1_csr_guarded_auto_solve",
            row_file="csr_guarded_auto_solve_results.jsonl",
        ),
        _stage(
            stage_id="csr_guarded_promotion_readiness",
            artifact_dir=runs / "phase1_csr_guarded_promotion_readiness",
            row_file="csr_guarded_promotion_readiness_results.jsonl",
        ),
        _stage(
            stage_id="csr_guarded_promotion_coverage_plan",
            artifact_dir=runs / "phase1_csr_guarded_promotion_coverage_plan",
            row_file="csr_guarded_promotion_coverage_scenarios.jsonl",
        ),
        _stage(
            stage_id="csr_guarded_promotion_coverage_exec",
            artifact_dir=runs / "phase1_csr_guarded_promotion_coverage_exec",
            row_file="csr_guarded_promotion_coverage_exec_results.jsonl",
        ),
        _stage(
            stage_id="csr_non_success_fallback_probe",
            artifact_dir=runs / "phase1_csr_non_success_fallback_probe",
            row_file="csr_non_success_fallback_probe_results.jsonl",
        ),
        _stage(
            stage_id="csr_guarded_non_success_fallback_integration",
            artifact_dir=runs / "phase1_csr_guarded_non_success_fallback_integration",
            row_file="csr_guarded_non_success_fallback_integration_results.jsonl",
        ),
        _stage(
            stage_id="csr_ilu0_guarded_integration",
            artifact_dir=runs / "phase1_csr_ilu0_guarded_integration",
            row_file="csr_ilu0_guarded_integration_results.jsonl",
        ),
        _stage(
            stage_id="csr_ilu0_coverage_expansion",
            artifact_dir=runs / "phase1_csr_ilu0_coverage_expansion",
            row_file="csr_ilu0_coverage_results.jsonl",
        ),
        _stage(
            stage_id="csr_unresolved_fallback_coverage_plan",
            artifact_dir=runs / "phase1_csr_unresolved_fallback_coverage_plan",
            row_file="csr_unresolved_fallback_candidate_plan.jsonl",
        ),
        _stage(
            stage_id="csr_unresolved_fallback_cpu_screen",
            artifact_dir=runs / "phase1_csr_unresolved_fallback_cpu_screen",
            row_file="csr_unresolved_fallback_cpu_screen_results.jsonl",
        ),
        _stage(
            stage_id="csr_unresolved_matrix_diagnostics",
            artifact_dir=runs / "phase1_csr_unresolved_matrix_diagnostics",
            row_file="csr_unresolved_matrix_diagnostics.jsonl",
        ),
        _stage(
            stage_id="public_release_hygiene",
            artifact_dir=runs / "phase1_public_release_hygiene",
            row_file="public_release_hygiene_rows.jsonl",
        ),
        _stage(
            stage_id="transformer_readiness",
            artifact_dir=runs / "phase1_transformer_readiness",
            row_file="policy_training_rows.jsonl",
        ),
        _stage(
            stage_id="public_api_smoke",
            artifact_dir=runs / "phase1_public_api_smoke",
            row_file="public_api_smoke.json",
        ),
        _stage(
            stage_id="csr_public_api_smoke",
            artifact_dir=runs / "phase1_csr_public_api",
            row_file="csr_public_api_smoke.json",
        ),
        _stage(
            stage_id="solver_functional",
            artifact_dir=runs / "phase1_solver_functional",
            row_file="solver_functional_results.jsonl",
        ),
    )
    status = "passed" if all(stage.status == "passed" for stage in stages) else "failed"
    return CampaignReport(
        campaign_id="phase1_readiness",
        status=status,
        stages=stages,
    )


def write_campaign_artifacts(report: CampaignReport, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "campaign_summary.json"
    report_path = output / "campaign_report.md"
    summary_path.write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_campaign_report(report, report_path)
    return {"summary": summary_path, "report": report_path}


def _stage(
    stage_id: str,
    artifact_dir: Path,
    row_file: str,
) -> CampaignStageReport:
    manifest_path = artifact_dir / "artifact_manifest.json"
    try:
        manifest = read_manifest(manifest_path)
        stale = verify_manifest_hashes(manifest)
        row_count = _row_count(artifact_dir / row_file)
    except Exception as exc:
        return CampaignStageReport(
            stage_id=stage_id,
            artifact_kind="unknown",
            artifact_dir=str(artifact_dir),
            status="failed",
            row_count=0,
            stale_paths=(),
            metadata={"error": str(exc)},
        )
    return CampaignStageReport(
        stage_id=stage_id,
        artifact_kind=manifest.artifact_kind,
        artifact_dir=str(artifact_dir),
        status="passed" if not stale else "failed",
        row_count=row_count,
        stale_paths=stale,
        metadata=dict(manifest.metadata),
    )


def _queue_batch_stages(runs: Path) -> tuple[CampaignStageReport, ...]:
    stages: list[CampaignStageReport] = []
    for path in sorted(runs.glob("phase1_csr_queue_batch_*"), key=lambda item: item.name):
        if not path.is_dir():
            continue
        summary_path = path / "csr_queue_batch_execution_summary.json"
        manifest_path = path / "artifact_manifest.json"
        if not summary_path.exists() or not manifest_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if summary.get("status") != "passed":
            continue
        batch_id = str(summary.get("batch_id") or path.name.rsplit("_", 1)[-1])
        stages.append(
            _stage(
                stage_id=f"csr_queue_batch_{batch_id.removeprefix('batch_')}",
                artifact_dir=path,
                row_file="csr_micro_campaign_results.jsonl",
            )
        )
    return tuple(stages)


def _row_count(path: Path) -> int:
    if path.suffix == ".json":
        return 1
    return len(read_jsonl(path))


def _write_campaign_report(report: CampaignReport, path: Path) -> Path:
    lines = [
        "# Phase 1 Campaign Report",
        "",
        f"- campaign: `{report.campaign_id}`",
        f"- status: `{report.status}`",
        f"- stages: `{report.num_stages}`",
        "",
        "| stage | status | artifact_kind | rows | artifact_dir |",
        "|---|---|---|---:|---|",
    ]
    for stage in report.stages:
        lines.append(
            "| "
            f"{stage.stage_id} | "
            f"{stage.status} | "
            f"{stage.artifact_kind} | "
            f"{stage.row_count} | "
            f"{stage.artifact_dir} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
