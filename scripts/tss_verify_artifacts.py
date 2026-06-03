"""Verify benchmark/regression artifacts are current and numerically valid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.artifacts import read_candidate_performance
from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.csr_blocked_gap_augmented_ranker_verifier import (
    verify_csr_blocked_gap_augmented_ranker as _verify_csr_blocked_gap_augmented_ranker,
)
from transsolvestack.profiling.csr_blocked_gap_guarded_replay_verifier import (
    verify_csr_blocked_gap_guarded_replay as _verify_csr_blocked_gap_guarded_replay,
)
from transsolvestack.profiling.csr_blocked_gap_training_integration_verifier import (
    verify_csr_blocked_gap_training_integration as _verify_csr_blocked_gap_training_integration,
)
from transsolvestack.profiling.csr_transformer_handoff_bundle_verifier import (
    verify_csr_transformer_handoff_bundle as _verify_csr_transformer_handoff_bundle,
)
from transsolvestack.profiling.csr_transformer_training_package_verifier import (
    verify_csr_transformer_training_package as _verify_csr_transformer_training_package,
)
from transsolvestack.profiling.csr_transformer_package_consumer_dry_run_verifier import (
    verify_csr_transformer_package_consumer_dry_run as _verify_csr_transformer_package_consumer_dry_run,
)
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def _verify_repeated_row(row: dict, *, expected_repeats: int, label: str) -> None:
    if int(row.get("measurement_repeats", 0)) != expected_repeats:
        raise SystemExit(f"{label} repeat count mismatch")
    if int(row.get("success_count", 0)) != expected_repeats:
        raise SystemExit(f"{label} success count mismatch")
    if float(row.get("success_rate", 0.0)) != 1.0:
        raise SystemExit(f"{label} success rate mismatch")
    if len(row.get("repeat_records", ())) != expected_repeats:
        raise SystemExit(f"{label} repeat records missing")
    if len(row.get("solve_time_samples_ms", ())) != expected_repeats:
        raise SystemExit(f"{label} solve time samples missing")
    median = float(row.get("median_solve_time_ms", math.inf))
    if not math.isfinite(median) or median <= 0.0:
        raise SystemExit(f"{label} median solve time invalid")
    if abs(float(row["solve_time_ms"]) - median) > 1.0e-9:
        raise SystemExit(f"{label} solve_time_ms is not median")
    if float(row.get("solve_time_iqr_ms", -1.0)) < 0.0:
        raise SystemExit(f"{label} solve time IQR invalid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default="runs/phase1_smoke_taichi")
    parser.add_argument(
        "--expanded-benchmark-dir",
        default="runs/phase1_smoke_expanded_solvers",
    )
    parser.add_argument("--regression-dir", default="runs/phase1_numerical_regression")
    parser.add_argument("--sequence-dir", default="runs/phase1_sequence_taichi")
    parser.add_argument("--policy-dir", default="runs/phase1_policy_selection")
    parser.add_argument("--policy-solve-dir", default="runs/phase1_policy_solve")
    parser.add_argument("--dataset-plan-dir", default="runs/phase1_dataset_plan")
    parser.add_argument("--matrix-probe-dir", default="runs/phase1_matrix_market_probe")
    parser.add_argument(
        "--suitesparse-selection-dir",
        default="runs/phase1_suitesparse_selection",
    )
    parser.add_argument(
        "--suitesparse-header-probe-dir",
        default="runs/phase1_suitesparse_header_probe",
    )
    parser.add_argument(
        "--suitesparse-csr-import-dir",
        default="runs/phase1_suitesparse_csr_import",
    )
    parser.add_argument(
        "--taichi-csr-matvec-dir",
        default="runs/phase1_taichi_csr_matvec",
    )
    parser.add_argument(
        "--taichi-csr-primitives-dir",
        default="runs/phase1_taichi_csr_primitives",
    )
    parser.add_argument(
        "--taichi-csr-solve-dir",
        default="runs/phase1_taichi_csr_solve",
    )
    parser.add_argument(
        "--taichi-csr-bicgstab-dir",
        default="runs/phase1_taichi_csr_bicgstab",
    )
    parser.add_argument(
        "--taichi-csr-gmres-dir",
        default="runs/phase1_taichi_csr_gmres",
    )
    parser.add_argument(
        "--taichi-csr-richardson-dir",
        default="runs/phase1_taichi_csr_richardson",
    )
    parser.add_argument(
        "--taichi-csr-chebyshev-dir",
        default="runs/phase1_taichi_csr_chebyshev",
    )
    parser.add_argument(
        "--taichi-csr-symmetric-equilibration-dir",
        default="runs/phase1_taichi_csr_symmetric_equilibration",
    )
    parser.add_argument(
        "--taichi-csr-row-column-equilibration-dir",
        default="runs/phase1_taichi_csr_row_column_equilibration",
    )
    parser.add_argument(
        "--taichi-csr-ilu0-dir",
        default="runs/phase1_taichi_csr_ilu0",
    )
    parser.add_argument(
        "--csr-selector-dir",
        default="runs/phase1_csr_selector_readiness",
    )
    parser.add_argument(
        "--csr-selector-policy-dir",
        default="runs/phase1_csr_selector_policy",
    )
    parser.add_argument(
        "--csr-auto-solve-dir",
        default="runs/phase1_csr_auto_solve",
    )
    parser.add_argument(
        "--csr-learning-dir",
        default="runs/phase1_csr_learning_readiness",
    )
    parser.add_argument(
        "--csr-model-contract-dir",
        default="runs/phase1_csr_model_contract",
    )
    parser.add_argument(
        "--csr-training-tensor-dir",
        default="runs/phase1_csr_training_tensors",
    )
    parser.add_argument(
        "--csr-linear-ranker-dir",
        default="runs/phase1_csr_linear_ranker",
    )
    parser.add_argument(
        "--csr-selector-model-eval-dir",
        default="runs/phase1_csr_selector_model_eval",
    )
    parser.add_argument(
        "--csr-benchmark-expansion-dir",
        default="runs/phase1_csr_benchmark_expansion_plan",
    )
    parser.add_argument(
        "--csr-full-dataset-queue-dir",
        default="runs/phase1_csr_full_dataset_queue",
    )
    parser.add_argument(
        "--csr-queue-batch-00001-dir",
        default="runs/phase1_csr_queue_batch_00001",
    )
    parser.add_argument(
        "--csr-queue-batch-00002-dir",
        default="runs/phase1_csr_queue_batch_00002",
    )
    parser.add_argument(
        "--csr-queue-batches-root",
        default="runs",
    )
    parser.add_argument(
        "--csr-queue-training-pool-dir",
        default="runs/phase1_csr_queue_training_pool",
    )
    parser.add_argument(
        "--csr-queue-batch-training-bundle-dir",
        default="runs/phase1_csr_queue_batch_training_bundle",
    )
    parser.add_argument(
        "--csr-queue-batch-reference-ranker-dir",
        default="runs/phase1_csr_queue_batch_reference_ranker",
    )
    parser.add_argument(
        "--csr-queue-batch-model-replay-dir",
        default="runs/phase1_csr_queue_batch_model_replay",
    )
    parser.add_argument(
        "--csr-queue-candidate-coverage-dir",
        default="runs/phase1_csr_queue_candidate_coverage",
    )
    parser.add_argument(
        "--csr-gmres-restart-coverage-dir",
        default="runs/phase1_csr_gmres_restart_coverage",
    )
    parser.add_argument(
        "--csr-blocked-gap-probe-dir",
        default="runs/phase1_csr_blocked_gap_probe",
    )
    parser.add_argument(
        "--csr-blocked-gap-positive-search-dir",
        default="runs/phase1_csr_blocked_gap_positive_search",
    )
    parser.add_argument(
        "--csr-blocked-gap-training-integration-dir",
        default="runs/phase1_csr_blocked_gap_training_integration",
    )
    parser.add_argument(
        "--csr-blocked-gap-augmented-ranker-dir",
        default="runs/phase1_csr_blocked_gap_augmented_ranker",
    )
    parser.add_argument(
        "--csr-blocked-gap-guarded-replay-dir",
        default="runs/phase1_csr_blocked_gap_guarded_replay",
    )
    parser.add_argument(
        "--csr-transformer-handoff-bundle-dir",
        default="runs/phase1_csr_transformer_handoff_bundle",
    )
    parser.add_argument(
        "--csr-transformer-training-package-dir",
        default="runs/phase1_csr_transformer_training_package",
    )
    parser.add_argument(
        "--csr-transformer-package-consumer-dry-run-dir",
        default="runs/phase1_csr_transformer_package_consumer_dry_run",
    )
    parser.add_argument(
        "--csr-micro-campaign-dir",
        default="runs/phase1_csr_micro_campaign",
    )
    parser.add_argument(
        "--csr-transformer-ready-dir",
        default="runs/phase1_csr_transformer_ready",
    )
    parser.add_argument(
        "--csr-transformer-ranker-dir",
        default="runs/phase1_csr_transformer_ranker",
    )
    parser.add_argument(
        "--csr-external-model-adapter-dir",
        default="runs/phase1_csr_external_model_adapter",
    )
    parser.add_argument(
        "--csr-transformer-model-replay-dir",
        default="runs/phase1_csr_transformer_model_replay",
    )
    parser.add_argument(
        "--csr-transformer-quality-gate-dir",
        default="runs/phase1_csr_transformer_quality_gate",
    )
    parser.add_argument(
        "--csr-policy-model-artifact-dir",
        default="runs/phase1_csr_policy_model_artifact",
    )
    parser.add_argument(
        "--csr-policy-model-acceptance-dir",
        default="runs/phase1_csr_policy_model_acceptance",
    )
    parser.add_argument(
        "--csr-policy-model-submission-dir",
        default="runs/phase1_csr_policy_model_submission",
    )
    parser.add_argument(
        "--csr-external-model-intake-dir",
        default="runs/phase1_csr_external_model_intake",
    )
    parser.add_argument(
        "--csr-transformer-training-entrypoint-dir",
        default="runs/phase1_csr_transformer_training_entrypoint",
    )
    parser.add_argument(
        "--csr-transformer-reference-training-export-dir",
        default="runs/phase1_csr_transformer_reference_training_export",
    )
    parser.add_argument(
        "--csr-learned-guard-dir",
        default="runs/phase1_csr_learned_guard",
    )
    parser.add_argument(
        "--csr-guarded-auto-solve-dir",
        default="runs/phase1_csr_guarded_auto_solve",
    )
    parser.add_argument(
        "--csr-guarded-promotion-readiness-dir",
        default="runs/phase1_csr_guarded_promotion_readiness",
    )
    parser.add_argument(
        "--csr-guarded-promotion-coverage-dir",
        default="runs/phase1_csr_guarded_promotion_coverage_plan",
    )
    parser.add_argument(
        "--csr-guarded-promotion-coverage-exec-dir",
        default="runs/phase1_csr_guarded_promotion_coverage_exec",
    )
    parser.add_argument(
        "--csr-non-success-fallback-probe-dir",
        default="runs/phase1_csr_non_success_fallback_probe",
    )
    parser.add_argument(
        "--csr-guarded-non-success-fallback-integration-dir",
        default="runs/phase1_csr_guarded_non_success_fallback_integration",
    )
    parser.add_argument(
        "--csr-ilu0-guarded-integration-dir",
        default="runs/phase1_csr_ilu0_guarded_integration",
    )
    parser.add_argument(
        "--csr-ilu0-coverage-expansion-dir",
        default="runs/phase1_csr_ilu0_coverage_expansion",
    )
    parser.add_argument(
        "--csr-unresolved-fallback-coverage-plan-dir",
        default="runs/phase1_csr_unresolved_fallback_coverage_plan",
    )
    parser.add_argument(
        "--csr-unresolved-fallback-cpu-screen-dir",
        default="runs/phase1_csr_unresolved_fallback_cpu_screen",
    )
    parser.add_argument(
        "--csr-unresolved-matrix-diagnostics-dir",
        default="runs/phase1_csr_unresolved_matrix_diagnostics",
    )
    parser.add_argument(
        "--public-release-hygiene-dir",
        default="runs/phase1_public_release_hygiene",
    )
    parser.add_argument(
        "--transformer-dir",
        default="runs/phase1_transformer_readiness",
    )
    parser.add_argument("--public-api-dir", default="runs/phase1_public_api_smoke")
    parser.add_argument("--csr-public-api-dir", default="runs/phase1_csr_public_api")
    parser.add_argument("--solver-functional-dir", default="runs/phase1_solver_functional")
    parser.add_argument("--campaign-dir", default="runs/phase1_campaign")
    args = parser.parse_args()

    _verify_benchmark(Path(args.benchmark_dir))
    _verify_expanded_benchmark(Path(args.expanded_benchmark_dir))
    _verify_regression(Path(args.regression_dir))
    _verify_sequence(Path(args.sequence_dir))
    _verify_policy_selection(Path(args.policy_dir))
    _verify_policy_solve(Path(args.policy_solve_dir))
    _verify_dataset_plan(Path(args.dataset_plan_dir))
    _verify_matrix_market_probe(Path(args.matrix_probe_dir))
    _verify_suitesparse_selection(Path(args.suitesparse_selection_dir))
    _verify_suitesparse_header_probe(Path(args.suitesparse_header_probe_dir))
    _verify_suitesparse_csr_import(Path(args.suitesparse_csr_import_dir))
    _verify_taichi_csr_matvec(Path(args.taichi_csr_matvec_dir))
    _verify_taichi_csr_primitives(Path(args.taichi_csr_primitives_dir))
    _verify_taichi_csr_solve(Path(args.taichi_csr_solve_dir))
    _verify_taichi_csr_bicgstab(Path(args.taichi_csr_bicgstab_dir))
    _verify_taichi_csr_gmres(Path(args.taichi_csr_gmres_dir))
    _verify_taichi_csr_richardson(Path(args.taichi_csr_richardson_dir))
    _verify_taichi_csr_chebyshev(Path(args.taichi_csr_chebyshev_dir))
    _verify_taichi_csr_symmetric_equilibration(
        Path(args.taichi_csr_symmetric_equilibration_dir)
    )
    _verify_taichi_csr_row_column_equilibration(
        Path(args.taichi_csr_row_column_equilibration_dir)
    )
    _verify_taichi_csr_ilu0(Path(args.taichi_csr_ilu0_dir))
    _verify_csr_selector_readiness(Path(args.csr_selector_dir))
    _verify_csr_selector_policy(Path(args.csr_selector_policy_dir))
    _verify_csr_auto_solve(Path(args.csr_auto_solve_dir))
    _verify_csr_learning_readiness(Path(args.csr_learning_dir))
    _verify_csr_model_contract(Path(args.csr_model_contract_dir))
    _verify_csr_training_tensors(Path(args.csr_training_tensor_dir))
    _verify_csr_linear_ranker(Path(args.csr_linear_ranker_dir))
    _verify_csr_selector_model_eval(Path(args.csr_selector_model_eval_dir))
    _verify_csr_benchmark_expansion_plan(Path(args.csr_benchmark_expansion_dir))
    _verify_csr_full_dataset_queue(Path(args.csr_full_dataset_queue_dir))
    _verify_completed_csr_queue_batches(Path(args.csr_queue_batches_root))
    _verify_csr_queue_training_pool(Path(args.csr_queue_training_pool_dir))
    _verify_csr_queue_batch_training_bundle(
        Path(args.csr_queue_batch_training_bundle_dir)
    )
    _verify_csr_queue_batch_reference_ranker(
        Path(args.csr_queue_batch_reference_ranker_dir)
    )
    _verify_csr_queue_batch_model_replay(Path(args.csr_queue_batch_model_replay_dir))
    _verify_csr_queue_candidate_coverage(
        Path(args.csr_queue_candidate_coverage_dir)
    )
    _verify_csr_gmres_restart_coverage(Path(args.csr_gmres_restart_coverage_dir))
    _verify_csr_blocked_gap_probe(Path(args.csr_blocked_gap_probe_dir))
    _verify_csr_blocked_gap_positive_search(
        Path(args.csr_blocked_gap_positive_search_dir)
    )
    _verify_csr_blocked_gap_training_integration(
        Path(args.csr_blocked_gap_training_integration_dir)
    )
    _verify_csr_blocked_gap_augmented_ranker(
        Path(args.csr_blocked_gap_augmented_ranker_dir)
    )
    _verify_csr_blocked_gap_guarded_replay(
        Path(args.csr_blocked_gap_guarded_replay_dir)
    )
    _verify_csr_transformer_handoff_bundle(
        Path(args.csr_transformer_handoff_bundle_dir)
    )
    _verify_csr_transformer_training_package(
        Path(args.csr_transformer_training_package_dir)
    )
    _verify_csr_transformer_package_consumer_dry_run(
        Path(args.csr_transformer_package_consumer_dry_run_dir)
    )
    _verify_csr_micro_campaign(Path(args.csr_micro_campaign_dir))
    _verify_csr_transformer_ready(Path(args.csr_transformer_ready_dir))
    _verify_csr_transformer_ranker(Path(args.csr_transformer_ranker_dir))
    _verify_csr_external_model_adapter(Path(args.csr_external_model_adapter_dir))
    _verify_csr_transformer_model_replay(Path(args.csr_transformer_model_replay_dir))
    _verify_csr_transformer_quality_gate(Path(args.csr_transformer_quality_gate_dir))
    _verify_csr_policy_model_artifact(Path(args.csr_policy_model_artifact_dir))
    _verify_csr_policy_model_acceptance(Path(args.csr_policy_model_acceptance_dir))
    _verify_csr_policy_model_submission(Path(args.csr_policy_model_submission_dir))
    _verify_csr_external_model_intake(Path(args.csr_external_model_intake_dir))
    _verify_csr_transformer_training_entrypoint(
        Path(args.csr_transformer_training_entrypoint_dir)
    )
    _verify_csr_transformer_reference_training_export(
        Path(args.csr_transformer_reference_training_export_dir)
    )
    _verify_csr_learned_runtime_guard(Path(args.csr_learned_guard_dir))
    _verify_csr_guarded_auto_solve(Path(args.csr_guarded_auto_solve_dir))
    _verify_csr_guarded_promotion_readiness(
        Path(args.csr_guarded_promotion_readiness_dir)
    )
    _verify_csr_guarded_promotion_coverage(
        Path(args.csr_guarded_promotion_coverage_dir)
    )
    _verify_csr_guarded_promotion_coverage_exec(
        Path(args.csr_guarded_promotion_coverage_exec_dir)
    )
    _verify_csr_non_success_fallback_probe(
        Path(args.csr_non_success_fallback_probe_dir)
    )
    _verify_csr_guarded_non_success_fallback_integration(
        Path(args.csr_guarded_non_success_fallback_integration_dir)
    )
    _verify_csr_ilu0_guarded_integration(
        Path(args.csr_ilu0_guarded_integration_dir)
    )
    _verify_csr_ilu0_coverage_expansion(
        Path(args.csr_ilu0_coverage_expansion_dir)
    )
    _verify_csr_unresolved_fallback_coverage_plan(
        Path(args.csr_unresolved_fallback_coverage_plan_dir)
    )
    _verify_csr_unresolved_fallback_cpu_screen(
        Path(args.csr_unresolved_fallback_cpu_screen_dir)
    )
    _verify_csr_unresolved_matrix_diagnostics(
        Path(args.csr_unresolved_matrix_diagnostics_dir)
    )
    _verify_public_release_hygiene(Path(args.public_release_hygiene_dir))
    _verify_transformer_readiness(Path(args.transformer_dir))
    _verify_public_api_smoke(Path(args.public_api_dir))
    _verify_csr_public_api_smoke(Path(args.csr_public_api_dir))
    _verify_solver_functional(Path(args.solver_functional_dir))
    _verify_campaign(Path(args.campaign_dir))
    print("artifact verification: passed")


def _verify_benchmark(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale benchmark artifact manifest: {stale}")
    rows = read_candidate_performance(path / "candidate_performance.jsonl")
    if len(rows) != 21:
        raise SystemExit(f"expected 21 benchmark rows, got {len(rows)}")
    if any(row.status != "success" for row in rows):
        raise SystemExit("benchmark contains non-success rows")
    if any(row.final_residual_norm is None or row.final_residual_norm > 1.0e-6 for row in rows):
        raise SystemExit("benchmark residual check failed")
    if any(row.metadata.get("relative_error_to_true", 1.0) >= 5.0e-3 for row in rows):
        raise SystemExit("benchmark relative error check failed")
    if any(row.measurement_repeats != 3 for row in rows):
        raise SystemExit("benchmark repeat count check failed")
    evaluation_path = path / "candidate_evaluation.jsonl"
    report_path = path / "benchmark_report.md"
    if not evaluation_path.exists() or not report_path.exists():
        raise SystemExit("benchmark evaluation artifacts are missing")
    evaluation_rows = read_jsonl(evaluation_path)
    if len(evaluation_rows) != 21:
        raise SystemExit(f"expected 21 evaluation rows, got {len(evaluation_rows)}")
    if sum(1 for row in evaluation_rows if row["is_oracle"]) != 7:
        raise SystemExit("expected exactly seven per-system oracle rows")
    if any(row["regret_vs_oracle"] is None for row in evaluation_rows):
        raise SystemExit("evaluation row missing regret")


def _verify_expanded_benchmark(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale expanded benchmark artifact manifest: {stale}")
    rows = read_candidate_performance(path / "candidate_performance.jsonl")
    if len(rows) != 35:
        raise SystemExit(f"expected 35 expanded benchmark rows, got {len(rows)}")
    if {row.candidate_id for row in rows} != {
        "taichi_pcg_jacobi_f32",
        "taichi_cg_none_f32",
        "taichi_bicgstab_jacobi_f32",
        "taichi_richardson_jacobi_f32",
        "taichi_chebyshev_jacobi_f32",
    }:
        raise SystemExit("expanded benchmark candidate set mismatch")
    failed = [row for row in rows if row.status != "success"]
    if len(failed) != 7:
        raise SystemExit(f"expected 7 expanded benchmark failures, got {len(failed)}")
    if any(row.candidate_id != "taichi_richardson_jacobi_f32" for row in failed):
        raise SystemExit("expanded benchmark has non-Richardson failures")
    successful = [row for row in rows if row.status == "success"]
    if any(
        row.final_residual_norm is None or row.final_residual_norm > 1.0e-6
        for row in successful
    ):
        raise SystemExit("expanded benchmark success residual check failed")
    if any(
        row.metadata.get("relative_error_to_true", 1.0) >= 5.0e-3
        for row in successful
    ):
        raise SystemExit("expanded benchmark success relative error check failed")
    evaluation_rows = read_jsonl(path / "candidate_evaluation.jsonl")
    if len(evaluation_rows) != 35:
        raise SystemExit(f"expected 35 expanded evaluation rows, got {len(evaluation_rows)}")
    if sum(1 for row in evaluation_rows if row["is_oracle"]) != 7:
        raise SystemExit("expanded benchmark expected seven oracle rows")


def _verify_regression(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale regression artifact manifest: {stale}")
    records_path = path / "numerical_regression.jsonl"
    rows = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 21:
        raise SystemExit(f"expected 21 regression rows, got {len(rows)}")
    if any(not row["passed"] for row in rows):
        raise SystemExit("regression contains failed rows")


def _verify_sequence(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale sequence artifact manifest: {stale}")
    steps = read_jsonl(path / "sequence_steps.jsonl")
    summary = read_jsonl(path / "sequence_summary.jsonl")
    if len(steps) != 8:
        raise SystemExit(f"expected 8 sequence step rows, got {len(steps)}")
    if len(summary) != 2:
        raise SystemExit(f"expected 2 sequence summary rows, got {len(summary)}")
    if any(row["status"] != "success" for row in steps):
        raise SystemExit("sequence contains non-success step rows")
    if any(row["final_residual_norm"] > 1.0e-6 for row in steps):
        raise SystemExit("sequence residual check failed")


def _verify_policy_selection(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale policy selection artifact manifest: {stale}")
    rows = read_jsonl(path / "selected_policy_plans.jsonl")
    if len(rows) != 7:
        raise SystemExit(f"expected 7 selected policy plans, got {len(rows)}")
    if any(row["backend"] != "taichi_gpu" for row in rows):
        raise SystemExit("policy selection contains non-Taichi backend")
    if any(row["reason"] != "profiled_success" for row in rows):
        raise SystemExit("policy selection contains fallback rows")
    if any(not row["audit"]["is_oracle"] for row in rows):
        raise SystemExit("policy selection did not choose per-system oracle rows")
    if any(len(row["fallback_candidate_ids"]) != 2 for row in rows):
        raise SystemExit("policy selection fallback chain check failed")


def _verify_policy_solve(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale policy solve artifact manifest: {stale}")
    rows = read_jsonl(path / "policy_solve_results.jsonl")
    if len(rows) != 7:
        raise SystemExit(f"expected 7 policy solve rows, got {len(rows)}")
    if any(row["status"] != "success" for row in rows):
        raise SystemExit("policy solve contains non-success rows")
    if any(row["selection_reason"] != "profiled_success" for row in rows):
        raise SystemExit("policy solve contains non-profiled selections")
    if any(row["final_residual_norm"] > 1.0e-6 for row in rows):
        raise SystemExit("policy solve residual check failed")
    if any(row["relative_error_to_true"] >= 5.0e-3 for row in rows):
        raise SystemExit("policy solve relative error check failed")
    if any(len(row["fallback_candidate_ids"]) != 2 for row in rows):
        raise SystemExit("policy solve fallback chain check failed")


def _verify_dataset_plan(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale dataset plan artifact manifest: {stale}")
    rows = read_jsonl(path / "download_plan.jsonl")
    if len(rows) != 3:
        raise SystemExit(f"expected 3 dataset plan rows, got {len(rows)}")
    if sum(float(row["estimated_download_gb"]) for row in rows) > 2.0:
        raise SystemExit("dataset plan exceeds coexistence download budget")
    if any(not row["url"].startswith("https://") for row in rows):
        raise SystemExit("dataset plan contains non-https URL")
    if any(row["format"] not in {"matrix_market_tar_gz", "matrix_market_gz"} for row in rows):
        raise SystemExit("dataset plan contains unsupported format")


def _verify_matrix_market_probe(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale matrix market probe artifact manifest: {stale}")
    rows = read_jsonl(path / "matrix_metadata.jsonl")
    if len(rows) != 1:
        raise SystemExit(f"expected 1 matrix metadata row, got {len(rows)}")
    row = rows[0]
    if row["storage_format"] != "coordinate":
        raise SystemExit("matrix metadata probe did not parse coordinate format")
    if row["n_rows"] != 3 or row["n_cols"] != 3 or row["nnz"] != 5:
        raise SystemExit("matrix metadata probe shape check failed")
    if row["symmetry"] != "symmetric":
        raise SystemExit("matrix metadata probe symmetry check failed")


def _verify_suitesparse_selection(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale SuiteSparse selection artifact manifest: {stale}")
    selected = read_jsonl(path / "selected_matrices.jsonl")
    systems = read_jsonl(path / "selected_systems.jsonl")
    summary = json.loads((path / "selection_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "suitesparse_matrix_subset_selection":
        raise SystemExit("unexpected SuiteSparse selection artifact kind")
    if summary["status"] != "ready":
        raise SystemExit("SuiteSparse selection is not ready")
    if len(selected) != 64 or len(systems) != 64:
        raise SystemExit("SuiteSparse selection expected 64 selected matrices/systems")
    if summary["selected_matrices"] != 64:
        raise SystemExit("SuiteSparse selection summary count mismatch")
    if summary["candidates_considered"] != 2904:
        raise SystemExit("SuiteSparse selection source count mismatch")
    if summary["candidates_after_filter"] <= summary["selected_matrices"]:
        raise SystemExit("SuiteSparse selection did not preserve a larger candidate pool")
    if any(row["source"] != "suitesparse" for row in selected):
        raise SystemExit("SuiteSparse selection contains non-SuiteSparse source")
    if any(row["n_rows"] != row["n_cols"] for row in selected):
        raise SystemExit("SuiteSparse selection contains non-square matrix")
    if any(not row["download_present"] for row in selected if "download_present" in row):
        raise SystemExit("SuiteSparse selection contains missing archive")
    if any(row["operator"]["kind"] != "assembled_sparse" for row in systems):
        raise SystemExit("SuiteSparse selection systems are not assembled_sparse")
    if any(row["operator"]["device_resident"] for row in systems):
        raise SystemExit("SuiteSparse selection systems should be metadata-only")


def _verify_suitesparse_header_probe(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale SuiteSparse header probe artifact manifest: {stale}")
    rows = read_jsonl(path / "archive_header_probe.jsonl")
    summary = json.loads((path / "archive_header_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "suitesparse_archive_header_probe":
        raise SystemExit("unexpected SuiteSparse header probe artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("SuiteSparse header probe is not passed")
    if len(rows) != 64:
        raise SystemExit("SuiteSparse header probe expected 64 rows")
    if summary["selected_matrices"] != 64 or summary["probed_archives"] != 64:
        raise SystemExit("SuiteSparse header probe summary count mismatch")
    if summary["success_count"] != 64 or summary["failure_count"] != 0:
        raise SystemExit("SuiteSparse header probe contains failed archive reads")
    if summary["shape_mismatch_count"] != 0:
        raise SystemExit("SuiteSparse header probe found shape mismatches")
    if any(row["status"] != "success" for row in rows):
        raise SystemExit("SuiteSparse header probe has non-success rows")
    if any(row["shape_matches"] is not True for row in rows):
        raise SystemExit("SuiteSparse header probe row shape check failed")
    if any(row["storage_format"] != "coordinate" for row in rows):
        raise SystemExit("SuiteSparse header probe expected coordinate matrices")


def _verify_suitesparse_csr_import(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale SuiteSparse CSR import artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_matrices.jsonl")
    summary = json.loads((path / "csr_import_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "suitesparse_csr_import_boundary":
        raise SystemExit("unexpected SuiteSparse CSR import artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("SuiteSparse CSR import is not passed")
    if summary["attempted_imports"] != 12 or summary["imported_matrices"] != 12:
        raise SystemExit("SuiteSparse CSR import expected 12 imported matrices")
    if summary["failed_imports"] != 0:
        raise SystemExit("SuiteSparse CSR import contains failed imports")
    if len(rows) != 12:
        raise SystemExit("SuiteSparse CSR import expected 12 CSR rows")
    if sum(int(row["csr_nnz"]) for row in rows) != summary["total_csr_nnz"]:
        raise SystemExit("SuiteSparse CSR import total nnz mismatch")
    for row in rows:
        row_ptr = row["row_ptr"]
        col_ind = row["col_ind"]
        values = row["values"]
        n_rows = int(row["n_rows"])
        n_cols = int(row["n_cols"])
        csr_nnz = int(row["csr_nnz"])
        if row["status"] != "success":
            raise SystemExit("SuiteSparse CSR import has non-success row")
        if len(row_ptr) != n_rows + 1:
            raise SystemExit("SuiteSparse CSR row_ptr length check failed")
        if row_ptr[0] != 0 or row_ptr[-1] != csr_nnz:
            raise SystemExit("SuiteSparse CSR row_ptr boundary check failed")
        if any(row_ptr[i] > row_ptr[i + 1] for i in range(n_rows)):
            raise SystemExit("SuiteSparse CSR row_ptr monotonicity check failed")
        if len(col_ind) != csr_nnz or len(values) != csr_nnz:
            raise SystemExit("SuiteSparse CSR array length check failed")
        if any(col < 0 or col >= n_cols for col in col_ind):
            raise SystemExit("SuiteSparse CSR column bounds check failed")
        if any(not math.isfinite(float(value)) for value in values):
            raise SystemExit("SuiteSparse CSR finite value check failed")


def _verify_taichi_csr_matvec(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale Taichi CSR matvec artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_matvec_results.jsonl")
    summary = json.loads((path / "csr_matvec_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "taichi_csr_matvec_smoke":
        raise SystemExit("unexpected Taichi CSR matvec artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("Taichi CSR matvec smoke is not passed")
    if summary["num_matrices"] != 4 or len(rows) != 4:
        raise SystemExit("Taichi CSR matvec expected 4 rows")
    if summary["num_success"] != 4 or summary["num_failed"] != 0:
        raise SystemExit("Taichi CSR matvec contains failed rows")
    if sum(int(row["csr_nnz"]) for row in rows) != summary["total_csr_nnz"]:
        raise SystemExit("Taichi CSR matvec total nnz mismatch")
    for row in rows:
        if row["status"] != "success":
            raise SystemExit("Taichi CSR matvec has non-success row")
        if row["backend"] != "taichi_gpu":
            raise SystemExit("Taichi CSR matvec did not use taichi_gpu")
        if float(row["relative_l2_error"]) > 1.0e-5:
            raise SystemExit("Taichi CSR matvec relative error check failed")
        if float(row["max_abs_error"]) > float(row["tolerance_abs"]):
            raise SystemExit("Taichi CSR matvec absolute error check failed")


def _verify_taichi_csr_primitives(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale Taichi CSR primitive artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_primitives_results.jsonl")
    summary = json.loads((path / "csr_primitives_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "taichi_csr_primitives_smoke":
        raise SystemExit("unexpected Taichi CSR primitive artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("Taichi CSR primitive smoke is not passed")
    if summary["num_matrices"] != 4 or len(rows) != 4:
        raise SystemExit("Taichi CSR primitive smoke expected 4 rows")
    if summary["num_success"] != 4 or summary["num_failed"] != 0:
        raise SystemExit("Taichi CSR primitive smoke contains failed rows")
    if sum(int(row["csr_nnz"]) for row in rows) != summary["total_csr_nnz"]:
        raise SystemExit("Taichi CSR primitive smoke total nnz mismatch")
    if float(summary["max_matvec_relative_l2_error"]) > 1.0e-5:
        raise SystemExit("Taichi CSR primitive summary matvec error check failed")
    if float(summary["max_residual_relative_norm"]) > 1.0e-5:
        raise SystemExit("Taichi CSR primitive summary residual check failed")
    if float(summary["max_dot_relative_error"]) > 1.0e-4:
        raise SystemExit("Taichi CSR primitive summary dot check failed")
    if float(summary["max_norm_relative_error"]) > 1.0e-4:
        raise SystemExit("Taichi CSR primitive summary norm check failed")
    for row in rows:
        if row["status"] != "success":
            raise SystemExit("Taichi CSR primitive smoke has non-success row")
        if row["backend"] != "taichi_gpu":
            raise SystemExit("Taichi CSR primitive smoke did not use taichi_gpu")
        if float(row["matvec_relative_l2_error"]) > 1.0e-5:
            raise SystemExit("Taichi CSR primitive row matvec error check failed")
        if float(row["matvec_max_abs_error"]) > float(row["matvec_tolerance_abs"]):
            raise SystemExit("Taichi CSR primitive row absolute error check failed")
        if float(row["residual_relative_norm"]) > 1.0e-5:
            raise SystemExit("Taichi CSR primitive row residual check failed")
        if float(row["residual_max_abs"]) > float(row["matvec_tolerance_abs"]):
            raise SystemExit("Taichi CSR primitive row residual abs check failed")
        if float(row["dot_relative_error"]) > 1.0e-4:
            raise SystemExit("Taichi CSR primitive row dot check failed")
        if float(row["norm_relative_error"]) > 1.0e-4:
            raise SystemExit("Taichi CSR primitive row norm check failed")


def _verify_taichi_csr_solve(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale Taichi CSR solve artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_solve_results.jsonl")
    summary = json.loads((path / "csr_solve_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "taichi_csr_solve_smoke":
        raise SystemExit("unexpected Taichi CSR solve artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("Taichi CSR solve smoke is not passed")
    if summary["num_selected_matrices"] != 2 or summary["num_solves"] != 4:
        raise SystemExit("Taichi CSR solve expected 2 matrices and 4 solves")
    if summary["precision"] != "float64":
        raise SystemExit("Taichi CSR solve expected float64 precision")
    if summary["measurement_repeats"] != 3:
        raise SystemExit("Taichi CSR solve expected 3 measurement repeats")
    if summary["num_success"] != 4 or summary["num_failed"] != 0:
        raise SystemExit("Taichi CSR solve contains failed rows")
    if {row["solver"] for row in rows} != {"cg", "pcg"}:
        raise SystemExit("Taichi CSR solve solver set mismatch")
    if {row["preconditioner"] for row in rows} != {"none", "jacobi"}:
        raise SystemExit("Taichi CSR solve preconditioner set mismatch")
    if sum(int(row["csr_nnz"]) for row in rows) != summary["total_csr_nnz"]:
        raise SystemExit("Taichi CSR solve total nnz mismatch")
    if float(summary["max_final_relative_residual"]) > float(summary["tolerance_rel"]):
        raise SystemExit("Taichi CSR solve summary trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("Taichi CSR solve summary CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("Taichi CSR solve summary solution error check failed")
    skipped_reasons = {row["reason"] for row in summary["skipped_candidates"]}
    if "cpu_reference_cg_screen_failed" not in skipped_reasons:
        raise SystemExit("Taichi CSR solve did not record CPU-screened skip")
    for row in rows:
        if row["status"] != "success":
            raise SystemExit("Taichi CSR solve has non-success row")
        _verify_repeated_row(row, expected_repeats=3, label="Taichi CSR solve")
        if row["backend"] != "taichi_gpu":
            raise SystemExit("Taichi CSR solve did not use taichi_gpu")
        if row["precision"] != summary["precision"]:
            raise SystemExit("Taichi CSR solve row precision mismatch")
        if row["symmetry"] != "symmetric":
            raise SystemExit("Taichi CSR solve used a non-symmetric matrix")
        if float(row["final_relative_residual"]) > float(summary["tolerance_rel"]):
            raise SystemExit("Taichi CSR solve row trace residual check failed")
        if float(row["cpu_recomputed_relative_residual"]) > 1.0e-4:
            raise SystemExit("Taichi CSR solve row CPU residual check failed")
        if float(row["solution_relative_error"]) > 5.0e-3:
            raise SystemExit("Taichi CSR solve row solution error check failed")
        history = row["residual_history"]
        if len(history) < 2 or float(history[-1]) > float(history[0]):
            raise SystemExit("Taichi CSR solve residual history check failed")


def _verify_taichi_csr_bicgstab(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale Taichi CSR BiCGSTAB artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_bicgstab_results.jsonl")
    summary = json.loads((path / "csr_bicgstab_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "taichi_csr_bicgstab_smoke":
        raise SystemExit("unexpected Taichi CSR BiCGSTAB artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("Taichi CSR BiCGSTAB smoke is not passed")
    if summary["num_selected_matrices"] != 2 or summary["num_solves"] != 4:
        raise SystemExit("Taichi CSR BiCGSTAB expected 2 matrices and 4 solves")
    if summary["precision"] != "float64":
        raise SystemExit("Taichi CSR BiCGSTAB expected float64 precision")
    if summary["measurement_repeats"] != 3:
        raise SystemExit("Taichi CSR BiCGSTAB expected 3 measurement repeats")
    if summary["num_success"] != 4 or summary["num_failed"] != 0:
        raise SystemExit("Taichi CSR BiCGSTAB contains failed rows")
    if {row["solver"] for row in rows} != {"bicgstab"}:
        raise SystemExit("Taichi CSR BiCGSTAB solver set mismatch")
    if {row["preconditioner"] for row in rows} != {"none", "jacobi"}:
        raise SystemExit("Taichi CSR BiCGSTAB preconditioner set mismatch")
    if any(row["symmetry"] == "symmetric" for row in rows):
        raise SystemExit("Taichi CSR BiCGSTAB smoke should use nonsymmetric rows")
    if sum(int(row["csr_nnz"]) for row in rows) != summary["total_csr_nnz"]:
        raise SystemExit("Taichi CSR BiCGSTAB total nnz mismatch")
    if float(summary["max_final_relative_residual"]) > float(summary["tolerance_rel"]):
        raise SystemExit("Taichi CSR BiCGSTAB summary trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("Taichi CSR BiCGSTAB summary CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("Taichi CSR BiCGSTAB summary solution error check failed")
    for row in rows:
        if row["status"] != "success":
            raise SystemExit("Taichi CSR BiCGSTAB has non-success row")
        _verify_repeated_row(row, expected_repeats=3, label="Taichi CSR BiCGSTAB")
        if row["backend"] != "taichi_gpu":
            raise SystemExit("Taichi CSR BiCGSTAB did not use taichi_gpu")
        if row["precision"] != summary["precision"]:
            raise SystemExit("Taichi CSR BiCGSTAB row precision mismatch")
        if row["failure_reasons"]:
            raise SystemExit("Taichi CSR BiCGSTAB row recorded failure reasons")
        if float(row["final_relative_residual"]) > float(summary["tolerance_rel"]):
            raise SystemExit("Taichi CSR BiCGSTAB row trace residual check failed")
        if float(row["cpu_recomputed_relative_residual"]) > 1.0e-4:
            raise SystemExit("Taichi CSR BiCGSTAB row CPU residual check failed")
        if float(row["solution_relative_error"]) > 5.0e-3:
            raise SystemExit("Taichi CSR BiCGSTAB row solution error check failed")
        if row["cpu_screen"]["success"] is not True:
            raise SystemExit("Taichi CSR BiCGSTAB row CPU screen did not pass")
        history = row["residual_history"]
        if len(history) < 2 or float(history[-1]) > float(history[0]):
            raise SystemExit("Taichi CSR BiCGSTAB residual history check failed")


def _verify_taichi_csr_gmres(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale Taichi CSR GMRES artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_gmres_results.jsonl")
    summary = json.loads((path / "csr_gmres_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "taichi_csr_gmres_smoke":
        raise SystemExit("unexpected Taichi CSR GMRES artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("Taichi CSR GMRES smoke is not passed")
    if summary["num_selected_matrices"] != 2 or summary["num_solves"] != 5:
        raise SystemExit("Taichi CSR GMRES expected 2 matrices and 5 solves")
    if summary["precision"] != "float64":
        raise SystemExit("Taichi CSR GMRES expected float64 precision")
    if tuple(summary["restarts"]) != (8, 16, 32):
        raise SystemExit("Taichi CSR GMRES expected restart sweep 8,16,32")
    if summary["num_restarts"] != 3:
        raise SystemExit("Taichi CSR GMRES expected 3 restart values")
    if summary["measurement_repeats"] != 3:
        raise SystemExit("Taichi CSR GMRES expected 3 measurement repeats")
    if summary["num_success"] != 5 or summary["num_failed"] != 0:
        raise SystemExit("Taichi CSR GMRES contains failed rows")
    if {row["solver"] for row in rows} != {"gmres"}:
        raise SystemExit("Taichi CSR GMRES solver set mismatch")
    if {row["preconditioner"] for row in rows} != {"jacobi"}:
        raise SystemExit("Taichi CSR GMRES preconditioner set mismatch")
    if not any(
        row["matrix_id"] == "suitesparse:HB/curtis54" and int(row["restart"]) == 16
        for row in rows
    ):
        raise SystemExit("Taichi CSR GMRES missing curtis54 restart=16 row")
    if not any(
        row["matrix_id"] == "suitesparse:HB/curtis54" and int(row["restart"]) == 32
        for row in rows
    ):
        raise SystemExit("Taichi CSR GMRES missing curtis54 restart=32 row")
    if not any(
        row["matrix_id"] == "suitesparse:HB/curtis54" and int(row["restart"]) == 8
        for row in summary["screened_out_restart_configs"]
    ):
        raise SystemExit("Taichi CSR GMRES did not record screened curtis54 restart=8")
    if any(row["symmetry"] == "symmetric" for row in rows):
        raise SystemExit("Taichi CSR GMRES smoke should use nonsymmetric rows")
    if sum(int(row["csr_nnz"]) for row in rows) != summary["total_csr_nnz"]:
        raise SystemExit("Taichi CSR GMRES total nnz mismatch")
    if float(summary["max_final_relative_residual"]) > float(summary["tolerance_rel"]):
        raise SystemExit("Taichi CSR GMRES summary trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("Taichi CSR GMRES summary CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("Taichi CSR GMRES summary solution error check failed")
    for row in rows:
        if row["status"] != "success":
            raise SystemExit("Taichi CSR GMRES has non-success row")
        _verify_repeated_row(row, expected_repeats=3, label="Taichi CSR GMRES")
        if row["backend"] != "taichi_gpu":
            raise SystemExit("Taichi CSR GMRES did not use taichi_gpu")
        if row["precision"] != summary["precision"]:
            raise SystemExit("Taichi CSR GMRES row precision mismatch")
        if row["failure_reasons"]:
            raise SystemExit("Taichi CSR GMRES row recorded failure reasons")
        if int(row["restart"]) not in {8, 16, 32}:
            raise SystemExit("Taichi CSR GMRES restart mismatch")
        if row["orthogonalization_backend"] != "taichi_gpu_modified_gram_schmidt":
            raise SystemExit("Taichi CSR GMRES orthogonalization backend mismatch")
        if row["least_squares_backend"] != "numpy_lstsq_dense_hessenberg":
            raise SystemExit("Taichi CSR GMRES least-squares backend mismatch")
        if float(row["final_relative_residual"]) > float(summary["tolerance_rel"]):
            raise SystemExit("Taichi CSR GMRES row trace residual check failed")
        if float(row["cpu_recomputed_relative_residual"]) > 1.0e-4:
            raise SystemExit("Taichi CSR GMRES row CPU residual check failed")
        if float(row["solution_relative_error"]) > 5.0e-3:
            raise SystemExit("Taichi CSR GMRES row solution error check failed")
        if row["cpu_screen"]["success"] is not True:
            raise SystemExit("Taichi CSR GMRES row CPU screen did not pass")
        history = row["residual_history"]
        if len(history) < 2 or float(history[-1]) > float(history[0]):
            raise SystemExit("Taichi CSR GMRES residual history check failed")


def _verify_taichi_csr_richardson(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale Taichi CSR Richardson artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_richardson_results.jsonl")
    summary = json.loads((path / "csr_richardson_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "taichi_csr_richardson_smoke":
        raise SystemExit("unexpected Taichi CSR Richardson artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("Taichi CSR Richardson smoke is not passed")
    if summary["num_selected_matrices"] != 1 or summary["num_solves"] != 1:
        raise SystemExit("Taichi CSR Richardson expected 1 matrix and 1 solve")
    if summary["precision"] != "float64":
        raise SystemExit("Taichi CSR Richardson expected float64 precision")
    if summary["measurement_repeats"] != 3:
        raise SystemExit("Taichi CSR Richardson expected 3 measurement repeats")
    if summary["num_success"] != 1 or summary["num_failed"] != 0:
        raise SystemExit("Taichi CSR Richardson contains failed rows")
    if {row["solver"] for row in rows} != {"richardson"}:
        raise SystemExit("Taichi CSR Richardson solver set mismatch")
    if {row["preconditioner"] for row in rows} != {"jacobi"}:
        raise SystemExit("Taichi CSR Richardson preconditioner set mismatch")
    if any(row["symmetry"] != "symmetric" for row in rows):
        raise SystemExit("Taichi CSR Richardson smoke should use symmetric rows")
    if sum(int(row["csr_nnz"]) for row in rows) != summary["total_csr_nnz"]:
        raise SystemExit("Taichi CSR Richardson total nnz mismatch")
    if float(summary["max_final_relative_residual"]) > float(summary["tolerance_rel"]):
        raise SystemExit("Taichi CSR Richardson summary trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("Taichi CSR Richardson summary CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("Taichi CSR Richardson summary solution error check failed")
    for row in rows:
        if row["status"] != "success":
            raise SystemExit("Taichi CSR Richardson has non-success row")
        _verify_repeated_row(row, expected_repeats=3, label="Taichi CSR Richardson")
        if row["backend"] != "taichi_gpu":
            raise SystemExit("Taichi CSR Richardson did not use taichi_gpu")
        if row["precision"] != summary["precision"]:
            raise SystemExit("Taichi CSR Richardson row precision mismatch")
        if row["failure_reasons"]:
            raise SystemExit("Taichi CSR Richardson row recorded failure reasons")
        if float(row["omega"]) != float(summary["omega"]):
            raise SystemExit("Taichi CSR Richardson omega mismatch")
        if float(row["final_relative_residual"]) > float(summary["tolerance_rel"]):
            raise SystemExit("Taichi CSR Richardson row trace residual check failed")
        if float(row["cpu_recomputed_relative_residual"]) > 1.0e-4:
            raise SystemExit("Taichi CSR Richardson row CPU residual check failed")
        if float(row["solution_relative_error"]) > 5.0e-3:
            raise SystemExit("Taichi CSR Richardson row solution error check failed")
        if row["cpu_screen"]["success"] is not True:
            raise SystemExit("Taichi CSR Richardson row CPU screen did not pass")
        history = row["residual_history"]
        if len(history) < 2 or float(history[-1]) > float(history[0]):
            raise SystemExit("Taichi CSR Richardson residual history check failed")


def _verify_taichi_csr_chebyshev(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale Taichi CSR Chebyshev artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_chebyshev_results.jsonl")
    summary = json.loads((path / "csr_chebyshev_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "taichi_csr_chebyshev_smoke":
        raise SystemExit("unexpected Taichi CSR Chebyshev artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("Taichi CSR Chebyshev smoke is not passed")
    if summary["num_selected_matrices"] != 1 or summary["num_solves"] != 1:
        raise SystemExit("Taichi CSR Chebyshev expected 1 matrix and 1 solve")
    if summary["precision"] != "float64":
        raise SystemExit("Taichi CSR Chebyshev expected float64 precision")
    if summary["measurement_repeats"] != 3:
        raise SystemExit("Taichi CSR Chebyshev expected 3 measurement repeats")
    if summary["num_success"] != 1 or summary["num_failed"] != 0:
        raise SystemExit("Taichi CSR Chebyshev contains failed rows")
    if {row["solver"] for row in rows} != {"chebyshev"}:
        raise SystemExit("Taichi CSR Chebyshev solver set mismatch")
    if {row["preconditioner"] for row in rows} != {"jacobi"}:
        raise SystemExit("Taichi CSR Chebyshev preconditioner set mismatch")
    if any(row["symmetry"] != "symmetric" for row in rows):
        raise SystemExit("Taichi CSR Chebyshev smoke should use symmetric rows")
    if sum(int(row["csr_nnz"]) for row in rows) != summary["total_csr_nnz"]:
        raise SystemExit("Taichi CSR Chebyshev total nnz mismatch")
    if float(summary["max_final_relative_residual"]) > float(summary["tolerance_rel"]):
        raise SystemExit("Taichi CSR Chebyshev summary trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("Taichi CSR Chebyshev summary CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("Taichi CSR Chebyshev summary solution error check failed")
    skipped_reasons = {row["reason"] for row in summary["skipped_candidates"]}
    if "cpu_reference_chebyshev_screen_failed" not in skipped_reasons:
        raise SystemExit("Taichi CSR Chebyshev did not record CPU-screened skip")
    for row in rows:
        if row["status"] != "success":
            raise SystemExit("Taichi CSR Chebyshev has non-success row")
        _verify_repeated_row(row, expected_repeats=3, label="Taichi CSR Chebyshev")
        if row["backend"] != "taichi_gpu":
            raise SystemExit("Taichi CSR Chebyshev did not use taichi_gpu")
        if row["precision"] != summary["precision"]:
            raise SystemExit("Taichi CSR Chebyshev row precision mismatch")
        if row["failure_reasons"]:
            raise SystemExit("Taichi CSR Chebyshev row recorded failure reasons")
        lower = float(row["lambda_min"])
        upper = float(row["lambda_max"])
        if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
            raise SystemExit("Taichi CSR Chebyshev invalid spectral bounds")
        if row["spectral_bounds_source"] != "dense_eigvalsh_jacobi_preconditioned_padded":
            raise SystemExit("Taichi CSR Chebyshev spectral source mismatch")
        if float(row["final_relative_residual"]) > float(summary["tolerance_rel"]):
            raise SystemExit("Taichi CSR Chebyshev row trace residual check failed")
        if float(row["cpu_recomputed_relative_residual"]) > 1.0e-4:
            raise SystemExit("Taichi CSR Chebyshev row CPU residual check failed")
        if float(row["solution_relative_error"]) > 5.0e-3:
            raise SystemExit("Taichi CSR Chebyshev row solution error check failed")
        if row["cpu_screen"]["success"] is not True:
            raise SystemExit("Taichi CSR Chebyshev row CPU screen did not pass")
        history = row["residual_history"]
        if len(history) < 2 or float(history[-1]) > float(history[0]):
            raise SystemExit("Taichi CSR Chebyshev residual history check failed")


def _verify_taichi_csr_symmetric_equilibration(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(
            f"stale Taichi CSR symmetric equilibration artifact manifest: {stale}"
        )
    rows = read_jsonl(path / "csr_symmetric_equilibration_results.jsonl")
    summary = json.loads(
        (path / "csr_symmetric_equilibration_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_symmetric_equilibration_schema.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "taichi_csr_symmetric_equilibration_smoke":
        raise SystemExit("unexpected Taichi CSR symmetric equilibration artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("Taichi CSR symmetric equilibration smoke is not passed")
    if summary["schema_version"] != "phase1_taichi_csr_symmetric_equilibration_v1":
        raise SystemExit("Taichi CSR symmetric equilibration schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("Taichi CSR symmetric equilibration schema/summary mismatch")
    if schema["integration_boundary"]["executes_gpu"] is not True:
        raise SystemExit("Taichi CSR symmetric equilibration should execute GPU")
    if summary["executes_gpu"] is not True:
        raise SystemExit("Taichi CSR symmetric equilibration GPU flag mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("Taichi CSR symmetric equilibration changed selector")
    if len(rows) != 1 or summary["candidate_rows"] != 1:
        raise SystemExit("Taichi CSR symmetric equilibration row count mismatch")
    if summary["gpu_executed_rows"] != 1:
        raise SystemExit("Taichi CSR symmetric equilibration did not execute GPU row")
    if summary["candidate_promoted"] is not False:
        raise SystemExit("Taichi CSR symmetric equilibration should not promote candidate")
    if summary["failed_numeric_gate_rows"] != 1:
        raise SystemExit("Taichi CSR symmetric equilibration gate failure mismatch")
    if summary["by_numeric_status"] != {"failed_numeric_gate": 1}:
        raise SystemExit("Taichi CSR symmetric equilibration status mismatch")
    row = rows[0]
    if row["matrix_id"] != "suitesparse:HB/bcsstk07":
        raise SystemExit("Taichi CSR symmetric equilibration matrix mismatch")
    if row["backend"] != "taichi_gpu" or row["gpu_executed"] is not True:
        raise SystemExit("Taichi CSR symmetric equilibration did not use taichi_gpu")
    if row["solver"] != "pcg":
        raise SystemExit("Taichi CSR symmetric equilibration solver mismatch")
    if row["preconditioner"] != "symmetric_equilibration":
        raise SystemExit("Taichi CSR symmetric equilibration preconditioner mismatch")
    if row["solver_status"] != "success":
        raise SystemExit("Taichi CSR symmetric equilibration solver did not converge")
    if row["numeric_status"] != "failed_numeric_gate":
        raise SystemExit("Taichi CSR symmetric equilibration numeric gate mismatch")
    if row["candidate_promoted"] is not False:
        raise SystemExit("Taichi CSR symmetric equilibration row promoted unexpectedly")
    if "solution_error_above_tolerance" not in row["failure_reasons"]:
        raise SystemExit("Taichi CSR symmetric equilibration missing solution-error gate")
    if float(row["final_relative_residual"]) > float(summary["tolerance_rel"]):
        raise SystemExit("Taichi CSR symmetric equilibration residual mismatch")
    if float(row["cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("Taichi CSR symmetric equilibration CPU residual mismatch")
    if float(row["solution_relative_error"]) <= 5.0e-3:
        raise SystemExit("Taichi CSR symmetric equilibration expected solution-error block")
    if row["trace_metadata"]["equilibration"] != "symmetric_diagonal":
        raise SystemExit("Taichi CSR symmetric equilibration metadata mismatch")
    if manifest.metadata["gpu_executed_rows"] != 1:
        raise SystemExit("Taichi CSR symmetric equilibration manifest GPU row mismatch")
    if manifest.metadata["candidate_promoted"] is not False:
        raise SystemExit("Taichi CSR symmetric equilibration manifest promotion mismatch")
    if manifest.metadata["executes_gpu"] is not True:
        raise SystemExit("Taichi CSR symmetric equilibration manifest GPU flag mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("Taichi CSR symmetric equilibration manifest runtime flag mismatch")


def _verify_taichi_csr_row_column_equilibration(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(
            f"stale Taichi CSR row/column equilibration artifact manifest: {stale}"
        )
    rows = read_jsonl(path / "csr_row_column_equilibration_results.jsonl")
    summary = json.loads(
        (path / "csr_row_column_equilibration_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_row_column_equilibration_schema.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "taichi_csr_row_column_equilibration_smoke":
        raise SystemExit("unexpected Taichi CSR row/column equilibration artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("Taichi CSR row/column equilibration smoke is not passed")
    if summary["schema_version"] != "phase1_taichi_csr_row_column_equilibration_v1":
        raise SystemExit("Taichi CSR row/column equilibration schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("Taichi CSR row/column equilibration schema/summary mismatch")
    if schema["integration_boundary"]["executes_gpu"] is not True:
        raise SystemExit("Taichi CSR row/column equilibration should execute GPU")
    if summary["executes_gpu"] is not True:
        raise SystemExit("Taichi CSR row/column equilibration GPU flag mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("Taichi CSR row/column equilibration changed selector")
    if len(rows) != summary["candidate_rows"] or len(rows) < 2:
        raise SystemExit("Taichi CSR row/column equilibration row count mismatch")
    if summary["gpu_executed_rows"] != len(rows):
        raise SystemExit("Taichi CSR row/column equilibration did not execute all GPU rows")
    if summary["numeric_success_rows"] < 1:
        raise SystemExit("Taichi CSR row/column equilibration expected one numeric success")
    if "suitesparse:Zitney/extr1b" not in {row["matrix_id"] for row in rows}:
        raise SystemExit("Taichi CSR row/column equilibration missing Zitney probe")
    for row in rows:
        if row["backend"] != "taichi_gpu" or row["gpu_executed"] is not True:
            raise SystemExit("Taichi CSR row/column equilibration did not use taichi_gpu")
        if row["preconditioner"] != "row_column_equilibration":
            raise SystemExit("Taichi CSR row/column equilibration preconditioner mismatch")
        if row["trace_metadata"]["equilibration"] != "row_column_l1":
            raise SystemExit("Taichi CSR row/column equilibration metadata mismatch")
        if row["runtime_selector_changed"] is not False:
            raise SystemExit("Taichi CSR row/column equilibration row changed selector")
        if not math.isfinite(float(row["final_relative_residual"])):
            raise SystemExit("Taichi CSR row/column equilibration nonfinite residual")
        if not math.isfinite(float(row["solution_relative_error"])):
            raise SystemExit("Taichi CSR row/column equilibration nonfinite solution error")
        if row["numeric_status"] == "success":
            if row["solver_status"] != "success":
                raise SystemExit("Taichi CSR row/column success row has failed solver")
            if float(row["final_relative_residual"]) > float(summary["tolerance_rel"]):
                raise SystemExit("Taichi CSR row/column success residual mismatch")
            if float(row["cpu_recomputed_relative_residual"]) > 1.0e-4:
                raise SystemExit("Taichi CSR row/column success CPU residual mismatch")
            if float(row["solution_relative_error"]) > 5.0e-3:
                raise SystemExit("Taichi CSR row/column success solution error mismatch")
    if manifest.metadata["gpu_executed_rows"] != len(rows):
        raise SystemExit("Taichi CSR row/column manifest GPU row mismatch")
    if manifest.metadata["numeric_success_rows"] != summary["numeric_success_rows"]:
        raise SystemExit("Taichi CSR row/column manifest success mismatch")
    if manifest.metadata["executes_gpu"] is not True:
        raise SystemExit("Taichi CSR row/column manifest GPU flag mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("Taichi CSR row/column manifest runtime flag mismatch")


def _verify_taichi_csr_ilu0(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale Taichi CSR ILU0 artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_ilu0_results.jsonl")
    summary = json.loads((path / "csr_ilu0_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_ilu0_schema.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "taichi_csr_ilu0_smoke":
        raise SystemExit("unexpected Taichi CSR ILU0 artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("Taichi CSR ILU0 smoke is not passed")
    if summary["schema_version"] != "phase1_taichi_csr_ilu0_v1":
        raise SystemExit("Taichi CSR ILU0 schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("Taichi CSR ILU0 schema/summary mismatch")
    if schema["integration_boundary"]["executes_gpu"] is not True:
        raise SystemExit("Taichi CSR ILU0 should execute GPU")
    if summary["executes_gpu"] is not True:
        raise SystemExit("Taichi CSR ILU0 GPU flag mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("Taichi CSR ILU0 changed selector")
    if len(rows) != summary["candidate_rows"] or len(rows) < 2:
        raise SystemExit("Taichi CSR ILU0 row count mismatch")
    if summary["gpu_executed_rows"] != len(rows):
        raise SystemExit("Taichi CSR ILU0 did not execute all GPU rows")
    if summary["numeric_success_rows"] < 1:
        raise SystemExit("Taichi CSR ILU0 expected one numeric success")
    if "suitesparse:Bai/cdde1" not in {row["matrix_id"] for row in rows}:
        raise SystemExit("Taichi CSR ILU0 missing Bai/cdde1 probe")
    for row in rows:
        if row["backend"] != "taichi_gpu" or row["gpu_executed"] is not True:
            raise SystemExit("Taichi CSR ILU0 did not use taichi_gpu")
        if row["solver"] != "bicgstab":
            raise SystemExit("Taichi CSR ILU0 solver mismatch")
        if row["preconditioner"] != "ilu0":
            raise SystemExit("Taichi CSR ILU0 preconditioner mismatch")
        if row["trace_metadata"]["ilu0_valid"] is not True:
            raise SystemExit("Taichi CSR ILU0 metadata did not record valid factors")
        if row["trace_metadata"]["ilu0_factorization"] != "doolittle_level_zero_csr_pattern":
            raise SystemExit("Taichi CSR ILU0 factorization metadata mismatch")
        if row["runtime_selector_changed"] is not False:
            raise SystemExit("Taichi CSR ILU0 row changed selector")
        if not math.isfinite(float(row["final_relative_residual"])):
            raise SystemExit("Taichi CSR ILU0 nonfinite residual")
        if not math.isfinite(float(row["solution_relative_error"])):
            raise SystemExit("Taichi CSR ILU0 nonfinite solution error")
        if row["numeric_status"] == "success":
            if row["solver_status"] != "success":
                raise SystemExit("Taichi CSR ILU0 success row has failed solver")
            if row["candidate_promoted"] is not True:
                raise SystemExit("Taichi CSR ILU0 success row was not promotable")
            if float(row["final_relative_residual"]) > float(summary["tolerance_rel"]):
                raise SystemExit("Taichi CSR ILU0 success residual mismatch")
            if float(row["cpu_recomputed_relative_residual"]) > 1.0e-4:
                raise SystemExit("Taichi CSR ILU0 success CPU residual mismatch")
            if float(row["solution_relative_error"]) > 5.0e-3:
                raise SystemExit("Taichi CSR ILU0 success solution error mismatch")
    if manifest.metadata["gpu_executed_rows"] != len(rows):
        raise SystemExit("Taichi CSR ILU0 manifest GPU row mismatch")
    if manifest.metadata["numeric_success_rows"] != summary["numeric_success_rows"]:
        raise SystemExit("Taichi CSR ILU0 manifest success mismatch")
    if manifest.metadata["executes_gpu"] is not True:
        raise SystemExit("Taichi CSR ILU0 manifest GPU flag mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("Taichi CSR ILU0 manifest runtime flag mismatch")


def _verify_csr_selector_readiness(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR selector readiness artifact manifest: {stale}")
    diagnostic_rows = read_jsonl(path / "csr_diagnostic_rows.jsonl")
    selector_rows = read_jsonl(path / "csr_selector_rows.jsonl")
    summary = json.loads((path / "csr_selector_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_selector_schema.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_selector_readiness_export":
        raise SystemExit("unexpected CSR selector readiness artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR selector readiness did not pass")
    if summary["schema_version"] != "phase1_csr_selector_features_v8":
        raise SystemExit("unexpected CSR selector summary schema version")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR selector schema/summary version mismatch")
    if schema["integration_boundary"]["status"] != "ready_for_real_csr_scaleout":
        raise SystemExit("CSR selector integration boundary not marked ready")
    if len(diagnostic_rows) != 12 or summary["num_diagnostic_rows"] != 12:
        raise SystemExit("CSR selector expected 12 diagnostic rows")
    if len(selector_rows) != 108 or summary["num_selector_rows"] != 108:
        raise SystemExit("CSR selector expected 108 selector rows")
    if summary["num_success_rows"] != 15 or summary["num_failed_rows"] != 15:
        raise SystemExit("CSR selector success/failed row count mismatch")
    if summary["num_applicability_rows"] != 78:
        raise SystemExit("CSR selector applicability row count mismatch")
    if summary["num_oracle_rows"] != 4:
        raise SystemExit("CSR selector expected 4 oracle rows")
    if summary["num_matrices_with_selector_rows"] != 12:
        raise SystemExit("CSR selector expected 12 matrices with selector rows")
    if float(summary["max_final_relative_residual"]) > 1.0e-5:
        raise SystemExit("CSR selector trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("CSR selector CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("CSR selector solution error check failed")
    if float(summary["min_success_rate"]) != 0.0:
        raise SystemExit("CSR selector expected screened-out rows")
    if not math.isinf(float(summary["max_failed_final_relative_residual"])):
        raise SystemExit("CSR selector expected nonfinite failed-row residual")
    if summary["failure_reason_counts"] != {
        "cpu_reference_bicgstab_screen_failed": 4,
        "cpu_reference_cg_screen_failed": 2,
        "cpu_reference_chebyshev_screen_failed": 1,
        "cpu_reference_gmres_screen_failed": 7,
        "cpu_reference_richardson_screen_failed": 1,
    }:
        raise SystemExit("CSR selector failure reason counts mismatch")
    if summary["applicability_status_counts"] != {
        "not_applicable": 34,
        "not_profiled": 44,
    }:
        raise SystemExit("CSR selector applicability status counts mismatch")
    if summary["applicability_reason_counts"] != {
        "above_smoke_size_limit": 2,
        "after_selection_limit": 42,
        "not_symmetric": 34,
    }:
        raise SystemExit("CSR selector applicability reason counts mismatch")
    if float(summary["max_solve_time_iqr_ms"]) < 0.0:
        raise SystemExit("CSR selector timing IQR check failed")
    if {row["candidate_id"] for row in selector_rows} != {
        "taichi_csr_cg_none_float64",
        "taichi_csr_pcg_jacobi_float64",
        "taichi_csr_bicgstab_none_float64",
        "taichi_csr_bicgstab_jacobi_float64",
        "taichi_csr_gmres_jacobi_restart8_float64",
        "taichi_csr_gmres_jacobi_restart16_float64",
        "taichi_csr_gmres_jacobi_restart32_float64",
        "taichi_csr_richardson_jacobi_float64",
        "taichi_csr_chebyshev_jacobi_float64",
    }:
        raise SystemExit("CSR selector candidate set mismatch")
    success_rows = tuple(row for row in selector_rows if row["target_status"] == "success")
    failed_rows = tuple(row for row in selector_rows if row["target_status"] == "screened_out")
    applicability_rows = tuple(
        row
        for row in selector_rows
        if row["target_status"] in {"not_applicable", "not_profiled"}
    )
    if (
        len(success_rows) != 15
        or len(failed_rows) != 15
        or len(applicability_rows) != 78
    ):
        raise SystemExit("CSR selector target status split mismatch")
    if {row["target_failure_reason"] for row in failed_rows} != {
        "cpu_reference_cg_screen_failed",
        "cpu_reference_bicgstab_screen_failed",
        "cpu_reference_gmres_screen_failed",
        "cpu_reference_richardson_screen_failed",
        "cpu_reference_chebyshev_screen_failed",
    }:
        raise SystemExit("CSR selector screened-out failure reason mismatch")
    if not any(
        row["matrix_id"] == "suitesparse:HB/curtis54"
        and row["solver"] == "gmres"
        and int(row["solver_parameters"].get("restart", 0)) == 8
        and float(row["target_final_relative_residual"]) > 1.0e-5
        for row in failed_rows
    ):
        raise SystemExit("CSR selector screened-out GMRES row mismatch")
    if not any(math.isinf(float(row["target_final_relative_residual"])) for row in failed_rows):
        raise SystemExit("CSR selector missing nonfinite Richardson failure row")
    if {row["target_status"] for row in applicability_rows} != {
        "not_applicable",
        "not_profiled",
    }:
        raise SystemExit("CSR selector applicability target status mismatch")
    if {row["target_applicability_reason"] for row in applicability_rows} != {
        "not_symmetric",
        "after_selection_limit",
        "above_smoke_size_limit",
    }:
        raise SystemExit("CSR selector applicability reason mismatch")
    if sum(1 for row in selector_rows if row["label_is_oracle"]) != 4:
        raise SystemExit("CSR selector oracle label count mismatch")
    if {row["solver"] for row in selector_rows} != {
        "cg",
        "pcg",
        "bicgstab",
        "gmres",
        "richardson",
        "chebyshev",
    }:
        raise SystemExit("CSR selector solver set mismatch")
    if not any(row["features"]["actual_symmetric"] for row in selector_rows):
        raise SystemExit("CSR selector missing symmetric benchmark rows")
    if not any(not row["features"]["actual_symmetric"] for row in selector_rows):
        raise SystemExit("CSR selector missing nonsymmetric benchmark rows")
    required_feature_keys = {
        "actual_symmetric",
        "recommended_precision",
        "solver",
        "preconditioner",
        "precision",
        "solver_parameters",
        "measurement_repeats",
        "success_rate",
        "median_solve_time_ms",
        "solve_time_iqr_ms",
        "screened_out_source",
        "failure_reason",
        "applicability_status",
        "applicability_reason",
    }
    for row in selector_rows:
        features = row["features"]
        if not required_feature_keys.issubset(features):
            raise SystemExit("CSR selector row missing required features")
        if features["recommended_precision"] not in {"float32", "float64"}:
            raise SystemExit("CSR selector expected a concrete precision recommendation")
        if features["precision"] != "float64":
            raise SystemExit("CSR selector benchmark rows must record float64 execution")
        if row["target_status"] == "success":
            if row["target_failure_reason"] is not None:
                raise SystemExit("CSR selector success row has failure reason")
            if row["target_applicability_status"] != "applicable":
                raise SystemExit("CSR selector success row applicability mismatch")
            if int(row["target_measurement_repeats"]) != 3:
                raise SystemExit("CSR selector success row repeat count mismatch")
            if float(row["target_success_rate"]) != 1.0:
                raise SystemExit("CSR selector success row success-rate mismatch")
            if float(row["target_median_solve_time_ms"]) <= 0.0:
                raise SystemExit("CSR selector success row median solve time invalid")
        elif row["target_status"] == "screened_out":
            if int(row["target_measurement_repeats"]) != 1:
                raise SystemExit("CSR selector screened row repeat count mismatch")
            if float(row["target_success_rate"]) != 0.0:
                raise SystemExit("CSR selector screened row success-rate mismatch")
            if row["target_median_solve_time_ms"] is not None:
                raise SystemExit("CSR selector screened row should not have GPU timing")
            if not row["target_failure_reason"]:
                raise SystemExit("CSR selector screened row missing failure reason")
            if row["target_applicability_status"] != "applicable":
                raise SystemExit("CSR selector screened row applicability mismatch")
            if not row["features"]["screened_out_source"]:
                raise SystemExit("CSR selector screened row missing source")
        else:
            if int(row["target_measurement_repeats"]) != 0:
                raise SystemExit("CSR selector applicability row repeat count mismatch")
            if float(row["target_success_rate"]) != 0.0:
                raise SystemExit("CSR selector applicability row success-rate mismatch")
            if row["target_median_solve_time_ms"] is not None:
                raise SystemExit("CSR selector applicability row should not have timing")
            if row["target_final_relative_residual"] is not None:
                raise SystemExit("CSR selector applicability row should not have residual")
            if row["target_failure_reason"] is not None:
                raise SystemExit("CSR selector applicability row should not have failure reason")
            if row["target_applicability_status"] not in {"not_applicable", "not_profiled"}:
                raise SystemExit("CSR selector applicability row status mismatch")
            if not row["target_applicability_reason"]:
                raise SystemExit("CSR selector applicability row missing reason")
            if not row["features"]["screened_out_source"]:
                raise SystemExit("CSR selector applicability row missing source")
        if row["solver"] == "chebyshev":
            params = row["solver_parameters"]
            if row["target_status"] in {"not_applicable", "not_profiled"}:
                continue
            if not (0.0 < float(params["lambda_min"]) < float(params["lambda_max"])):
                raise SystemExit("CSR selector Chebyshev row missing spectral bounds")
            if params["spectral_bounds_source"] != "dense_eigvalsh_jacobi_preconditioned_padded":
                raise SystemExit("CSR selector Chebyshev spectral source mismatch")
        if row["solver"] == "gmres" and int(row["solver_parameters"].get("restart", 0)) not in {8, 16, 32}:
            raise SystemExit("CSR selector GMRES row missing restart")
        if row["solver"] == "richardson" and "omega" not in row["solver_parameters"]:
            raise SystemExit("CSR selector Richardson row missing omega")


def _verify_csr_selector_policy(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR selector policy artifact manifest: {stale}")
    plans = read_jsonl(path / "csr_selected_policy_plans.jsonl")
    solve_checks = read_jsonl(path / "csr_selector_policy_solve_checks.jsonl")
    summary = json.loads((path / "csr_selector_policy_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_selector_policy_smoke":
        raise SystemExit("unexpected CSR selector policy artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR selector policy smoke did not pass")
    if summary["num_selected_plans"] != 4 or len(plans) != 4:
        raise SystemExit("CSR selector policy expected 4 selected plans")
    if summary["num_oracle_selected_plans"] != 4:
        raise SystemExit("CSR selector policy should select four oracle plans")
    if summary["num_solve_checks"] != 2 or len(solve_checks) != 2:
        raise SystemExit("CSR selector policy expected 2 solve checks")
    if summary["num_successful_solve_checks"] != 2:
        raise SystemExit("CSR selector policy solve checks did not all pass")
    if not set(summary["selected_solver_set"]).issubset(
        {"cg", "pcg", "bicgstab", "gmres", "richardson", "chebyshev"}
    ):
        raise SystemExit("CSR selector policy selected solver set mismatch")
    if float(summary["max_final_relative_residual"]) > 1.0e-5:
        raise SystemExit("CSR selector policy trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("CSR selector policy CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("CSR selector policy solution error check failed")
    for row in plans:
        if row["reason"] != "profiled_success":
            raise SystemExit("CSR selector policy selected non-profiled plan")
        if row["audit"]["selector"] != "csr_artifact":
            raise SystemExit("CSR selector policy audit selector mismatch")
        if row["audit"]["is_oracle"] is not True:
            raise SystemExit("CSR selector policy selected non-oracle plan")
        if row["plan"]["backend"] != "taichi_gpu":
            raise SystemExit("CSR selector policy selected non-Taichi backend")
        if row["plan"]["solver"]["name"] not in {
            "cg",
            "pcg",
            "bicgstab",
            "gmres",
            "richardson",
            "chebyshev",
        }:
            raise SystemExit("CSR selector policy unexpected selected solver")
        for fallback in row["plan"]["fallback_chain"]:
            if fallback["solver"]["name"] == "chebyshev":
                solver = fallback["solver"]
                if not (0.0 < float(solver["lambda_min"]) < float(solver["lambda_max"])):
                    raise SystemExit("CSR selector policy Chebyshev fallback lost bounds")
    for row in solve_checks:
        if row["status"] != "success":
            raise SystemExit("CSR selector policy has failed solve check")
        if row["trace"]["backend"] != "taichi_gpu":
            raise SystemExit("CSR selector policy solve check did not use taichi_gpu")
        if row["failure_reasons"]:
            raise SystemExit("CSR selector policy solve check recorded failure reasons")


def _verify_csr_auto_solve(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR auto-solve artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_auto_solve_results.jsonl")
    summary = json.loads((path / "csr_auto_solve_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_auto_solve_smoke":
        raise SystemExit("unexpected CSR auto-solve artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR auto-solve smoke did not pass")
    if summary["num_solves"] != 2 or len(rows) != 2:
        raise SystemExit("CSR auto-solve expected 2 solve rows")
    if summary["num_success"] != 2 or summary["num_failed"] != 0:
        raise SystemExit("CSR auto-solve contains failed solves")
    if not set(summary["selected_solver_set"]).issubset(
        {"cg", "pcg", "bicgstab", "gmres", "richardson", "chebyshev"}
    ):
        raise SystemExit("CSR auto-solve selected solver set mismatch")
    if summary["num_oracle_selected"] != 2:
        raise SystemExit("CSR auto-solve did not select two oracle candidates")
    if float(summary["max_final_relative_residual"]) > 1.0e-5:
        raise SystemExit("CSR auto-solve trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("CSR auto-solve CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("CSR auto-solve solution error check failed")
    for row in rows:
        if row["status"] != "success":
            raise SystemExit("CSR auto-solve has failed row")
        if row["selection_reason"] != "profiled_success":
            raise SystemExit("CSR auto-solve selected non-profiled row")
        if row["selected_is_oracle"] is not True:
            raise SystemExit("CSR auto-solve selected non-oracle row")
        if row["trace"]["backend"] != "taichi_gpu":
            raise SystemExit("CSR auto-solve did not use taichi_gpu")
        if row["trace"]["metadata"]["operator_backend"] != "taichi_csr":
            raise SystemExit("CSR auto-solve did not use taichi_csr")
        auto_metadata = row["trace"]["metadata"].get("auto_solve_csr")
        if not auto_metadata:
            raise SystemExit("CSR auto-solve trace missing auto_solve_csr metadata")
        if auto_metadata["selector"] != "csr_artifact":
            raise SystemExit("CSR auto-solve metadata selector mismatch")
        if auto_metadata["candidate_id"] != row["candidate_id"]:
            raise SystemExit("CSR auto-solve metadata candidate mismatch")
        if row["failure_reasons"]:
            raise SystemExit("CSR auto-solve row recorded failure reasons")


def _verify_csr_learning_readiness(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR learning readiness artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_learning_rows.jsonl")
    predictions = read_jsonl(path / "csr_baseline_predictions.jsonl")
    summary = json.loads((path / "csr_learning_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_learning_schema.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_learning_readiness_export":
        raise SystemExit("unexpected CSR learning readiness artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR learning readiness did not pass")
    if summary["schema_version"] != "phase1_csr_learning_features_v1":
        raise SystemExit("unexpected CSR learning schema version")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR learning schema/summary version mismatch")
    if schema["model_required"] is not False:
        raise SystemExit("CSR learning readiness should not require a model")
    if len(rows) != 108 or summary["num_rows"] != 108:
        raise SystemExit("CSR learning expected 108 rows")
    if summary["num_train_rows"] != 81 or summary["num_eval_rows"] != 27:
        raise SystemExit("CSR learning split row count mismatch")
    if summary["num_train_matrices"] != 9 or summary["num_eval_matrices"] != 3:
        raise SystemExit("CSR learning split matrix count mismatch")
    if summary["label_class_counts"] != {
        "not_applicable": 34,
        "not_profiled": 44,
        "screened_out": 15,
        "success_non_oracle": 11,
        "success_oracle": 4,
    }:
        raise SystemExit("CSR learning label class count mismatch")
    if len(predictions) != 3 or summary["num_eval_predictions"] != 3:
        raise SystemExit("CSR learning expected three eval predictions")
    if summary["num_eval_matrices_with_success"] != 2:
        raise SystemExit("CSR learning expected two eval matrices with success rows")
    if summary["num_eval_profiled_success_predictions"] != 2:
        raise SystemExit("CSR learning expected two profiled-success predictions")
    if abs(float(summary["eval_oracle_top1_accuracy"]) - (1.0 / 3.0)) > 1.0e-12:
        raise SystemExit("CSR learning baseline accuracy mismatch")
    if abs(float(summary["eval_profiled_success_rate"]) - (2.0 / 3.0)) > 1.0e-12:
        raise SystemExit("CSR learning baseline success-rate mismatch")
    if float(summary["eval_max_regret_ms"]) <= 0.0:
        raise SystemExit("CSR learning expected positive max regret")
    statuses = {row["status"] for row in predictions}
    if statuses != {
        "oracle_match",
        "profiled_success_non_oracle",
        "no_profiled_success_candidate",
    }:
        raise SystemExit("CSR learning prediction status set mismatch")
    if not all(row["schema_version"] == summary["schema_version"] for row in rows):
        raise SystemExit("CSR learning row schema mismatch")


def _verify_csr_model_contract(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR model contract artifact manifest: {stale}")
    requests = read_jsonl(path / "csr_model_requests.jsonl")
    targets = read_jsonl(path / "csr_model_targets.jsonl")
    predictions = read_jsonl(path / "csr_model_predictions.jsonl")
    summary = json.loads((path / "csr_model_contract_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_model_contract_schema.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_model_contract_export":
        raise SystemExit("unexpected CSR model contract artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR model contract did not pass")
    if summary["schema_version"] != "phase1_csr_model_contract_v1":
        raise SystemExit("unexpected CSR model contract schema version")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR model contract schema/summary mismatch")
    if schema["model_required"] is not False:
        raise SystemExit("CSR model contract should not require a model")
    if schema["runtime_integration"]["runtime_selector_changed"] is not False:
        raise SystemExit("CSR model contract must not change runtime selector")
    if summary["model_required"] is not False or summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR model contract runtime/model flags mismatch")
    if len(requests) != 12 or summary["num_requests"] != 12:
        raise SystemExit("CSR model contract expected 12 requests")
    if len(targets) != 12 or summary["num_targets"] != 12:
        raise SystemExit("CSR model contract expected 12 targets")
    if len(predictions) != 3 or summary["num_predictions"] != 3:
        raise SystemExit("CSR model contract expected 3 predictions")
    if summary["num_train_requests"] != 9 or summary["num_eval_requests"] != 3:
        raise SystemExit("CSR model contract split count mismatch")
    if summary["num_request_candidates"] != 108:
        raise SystemExit("CSR model contract candidate count mismatch")
    if summary["min_candidates_per_request"] != 9 or summary["max_candidates_per_request"] != 9:
        raise SystemExit("CSR model contract candidates/request mismatch")
    if summary["target_label_class_counts"] != {
        "not_applicable": 34,
        "not_profiled": 44,
        "screened_out": 15,
        "success_non_oracle": 11,
        "success_oracle": 4,
    }:
        raise SystemExit("CSR model contract label counts mismatch")
    if summary["num_predictions_with_selected_candidate"] != 2:
        raise SystemExit("CSR model contract selected prediction count mismatch")
    if summary["num_predictions_with_profiled_success_selection"] != 2:
        raise SystemExit("CSR model contract profiled prediction count mismatch")
    if abs(float(summary["eval_oracle_top1_accuracy"]) - (1.0 / 3.0)) > 1.0e-12:
        raise SystemExit("CSR model contract accuracy mismatch")
    if abs(float(summary["eval_profiled_selection_rate"]) - (2.0 / 3.0)) > 1.0e-12:
        raise SystemExit("CSR model contract profiled selection rate mismatch")
    if int(summary["validation_error_count"]) != 0:
        raise SystemExit("CSR model contract validation errors found")
    request_by_id = {row["request_id"]: row for row in requests}
    target_by_id = {row["request_id"]: row for row in targets}
    if set(request_by_id) != set(target_by_id):
        raise SystemExit("CSR model contract request/target ids mismatch")
    leakage_keys = {
        "measurement_repeats",
        "median_solve_time_ms",
        "solve_time_iqr_ms",
        "success_rate",
        "failure_reason",
        "screened_out_source",
        "skip_reason",
        "applicability_status",
        "applicability_reason",
    }
    for request in requests:
        if request["schema_version"] != summary["schema_version"]:
            raise SystemExit("CSR model contract request schema mismatch")
        if len(request["candidate_ids"]) != 9:
            raise SystemExit("CSR model contract request candidate count mismatch")
        if set(request["candidate_ids"]) != set(request["candidate_features"]):
            raise SystemExit("CSR model contract candidate feature ids mismatch")
        for features in request["candidate_features"].values():
            if leakage_keys & set(features):
                raise SystemExit("CSR model contract leaked target feature")
    for prediction in predictions:
        request = request_by_id[prediction["request_id"]]
        ranked = tuple(prediction["ranked_candidate_ids"])
        if set(ranked) != set(request["candidate_ids"]):
            raise SystemExit("CSR model contract prediction rank set mismatch")
        if len(ranked) != len(request["candidate_ids"]):
            raise SystemExit("CSR model contract prediction duplicate rank")
        if set(prediction["scores"]) != set(request["candidate_ids"]):
            raise SystemExit("CSR model contract prediction score set mismatch")
        selected = prediction["selected_candidate_id"]
        if selected is not None and ranked[0] != selected:
            raise SystemExit("CSR model contract selected candidate is not ranked first")
    if {row["evaluation_status"] for row in predictions} != {
        "oracle_match",
        "profiled_success_non_oracle",
        "no_profiled_success_candidate",
    }:
        raise SystemExit("CSR model contract prediction status mismatch")


def _verify_csr_training_tensors(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR training tensor artifact manifest: {stale}")
    arrays = json.loads((path / "csr_training_tensors.json").read_text(encoding="utf-8"))
    index_rows = read_jsonl(path / "csr_training_request_index.jsonl")
    summary = json.loads((path / "csr_training_tensor_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_training_tensor_schema.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_training_tensor_export":
        raise SystemExit("unexpected CSR training tensor artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR training tensor export did not pass")
    if summary["schema_version"] != "phase1_csr_training_tensors_v1":
        raise SystemExit("unexpected CSR training tensor schema version")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR training tensor schema/summary mismatch")
    if schema["model_required"] is not False or summary["model_required"] is not False:
        raise SystemExit("CSR training tensor should not require a model")
    if (
        schema["runtime_selector_changed"] is not False
        or summary["runtime_selector_changed"] is not False
    ):
        raise SystemExit("CSR training tensor must not change runtime selector")
    if summary["storage_format"] != "json_numeric_arrays_v1":
        raise SystemExit("CSR training tensor storage format mismatch")
    if summary["num_requests"] != 12 or len(index_rows) != 12:
        raise SystemExit("CSR training tensor expected 12 requests")
    if summary["num_train_requests"] != 9 or summary["num_eval_requests"] != 3:
        raise SystemExit("CSR training tensor split mismatch")
    if summary["num_global_candidates"] != 9:
        raise SystemExit("CSR training tensor expected 9 candidates")
    if summary["num_active_candidate_slots"] != 108:
        raise SystemExit("CSR training tensor active candidate count mismatch")
    if summary["num_oracle_targets"] != 4 or summary["num_requests_without_oracle"] != 8:
        raise SystemExit("CSR training tensor oracle count mismatch")
    if summary["has_missing_candidate_slots"] is not False:
        raise SystemExit("CSR training tensor should have no missing slots")
    if summary["label_class_counts"] != {
        "not_applicable": 34,
        "not_profiled": 44,
        "screened_out": 15,
        "success_non_oracle": 11,
        "success_oracle": 4,
    }:
        raise SystemExit("CSR training tensor label counts mismatch")
    if summary["target_status_counts"] != {
        "not_applicable": 34,
        "not_profiled": 44,
        "screened_out": 15,
        "success": 15,
    }:
        raise SystemExit("CSR training tensor status counts mismatch")
    if summary["matrix_feature_dim"] != 21 or summary["candidate_feature_dim"] != 16:
        raise SystemExit("CSR training tensor feature dimensions mismatch")
    if len(arrays["request_ids"]) != summary["num_requests"]:
        raise SystemExit("CSR training tensor request axis mismatch")
    if len(arrays["global_candidate_ids"]) != summary["num_global_candidates"]:
        raise SystemExit("CSR training tensor candidate axis mismatch")
    if len(arrays["matrix_feature_names"]) != summary["matrix_feature_dim"]:
        raise SystemExit("CSR training tensor matrix feature names mismatch")
    if len(arrays["candidate_feature_names"]) != summary["candidate_feature_dim"]:
        raise SystemExit("CSR training tensor candidate feature names mismatch")
    if len(arrays["matrix_features"]) != summary["num_requests"]:
        raise SystemExit("CSR training tensor matrix feature rows mismatch")
    if any(len(row) != summary["matrix_feature_dim"] for row in arrays["matrix_features"]):
        raise SystemExit("CSR training tensor matrix feature width mismatch")
    if len(arrays["candidate_features"]) != summary["num_requests"]:
        raise SystemExit("CSR training tensor candidate feature rows mismatch")
    if any(
        len(candidate_row) != summary["candidate_feature_dim"]
        for request_row in arrays["candidate_features"]
        for candidate_row in request_row
    ):
        raise SystemExit("CSR training tensor candidate feature width mismatch")
    for key in (
        "candidate_mask",
        "label_class_ids",
        "target_status_ids",
        "profiled_success_mask",
        "target_median_solve_time_ms",
        "target_regret_vs_oracle_ms",
    ):
        if len(arrays[key]) != summary["num_requests"]:
            raise SystemExit(f"CSR training tensor {key} row count mismatch")
        if any(len(row) != summary["num_global_candidates"] for row in arrays[key]):
            raise SystemExit(f"CSR training tensor {key} width mismatch")
    if sum(sum(row) for row in arrays["candidate_mask"]) != 108:
        raise SystemExit("CSR training tensor mask count mismatch")
    if sum(1 for value in arrays["oracle_index"] if int(value) >= 0) != 4:
        raise SystemExit("CSR training tensor oracle index count mismatch")
    if {row["split_id"] for row in index_rows} != {0, 1}:
        raise SystemExit("CSR training tensor split ids mismatch")


def _verify_csr_linear_ranker(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR linear ranker artifact manifest: {stale}")
    model = json.loads((path / "csr_linear_ranker_model.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(path / "csr_linear_ranker_predictions.jsonl")
    summary = json.loads((path / "csr_linear_ranker_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_linear_ranker_schema.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_linear_ranker_baseline":
        raise SystemExit("unexpected CSR linear ranker artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR linear ranker did not pass artifact checks")
    if summary["schema_version"] != "phase1_csr_linear_ranker_v1":
        raise SystemExit("unexpected CSR linear ranker schema version")
    if summary["model_family"] != "pairwise_linear_ranker_v1":
        raise SystemExit("unexpected CSR linear ranker model family")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR linear ranker schema/summary mismatch")
    if schema["model_family"] != summary["model_family"]:
        raise SystemExit("CSR linear ranker schema/model-family mismatch")
    if schema["model_required"] is not True:
        raise SystemExit("CSR linear ranker schema should require a model")
    if schema["runtime_selector_changed"] is not False:
        raise SystemExit("CSR linear ranker schema should not change runtime selector")
    if model["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR linear ranker model schema mismatch")
    if model["model_family"] != summary["model_family"]:
        raise SystemExit("CSR linear ranker model family mismatch")
    if model["model_id"] != summary["model_id"]:
        raise SystemExit("CSR linear ranker model id mismatch")
    if model["training"]["objective"] != "label_utility_pairwise_hinge_with_oracle_priority":
        raise SystemExit("CSR linear ranker training objective mismatch")
    if model["runtime_integration"]["runtime_selector_changed"] is not False:
        raise SystemExit("CSR linear ranker must not modify runtime selector")
    if summary["model_trained"] is not True:
        raise SystemExit("CSR linear ranker should record model_trained=true")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR linear ranker summary should not change runtime selector")
    if summary["num_requests"] != 12 or summary["num_predictions"] != 12:
        raise SystemExit("CSR linear ranker expected 12 predictions")
    if len(predictions) != summary["num_predictions"]:
        raise SystemExit("CSR linear ranker prediction count mismatch")
    if summary["num_train_requests"] != 9 or summary["num_eval_requests"] != 3:
        raise SystemExit("CSR linear ranker split count mismatch")
    if summary["num_train_oracle_requests"] != 2 or summary["num_eval_oracle_requests"] != 2:
        raise SystemExit("CSR linear ranker oracle split count mismatch")
    if summary["num_global_candidates"] != 9:
        raise SystemExit("CSR linear ranker expected 9 global candidates")
    if summary["num_features"] != 374:
        raise SystemExit("CSR linear ranker feature count mismatch")
    if len(model["feature_names"]) != summary["num_features"]:
        raise SystemExit("CSR linear ranker feature-name count mismatch")
    if len(model["weights"]) != summary["num_features"]:
        raise SystemExit("CSR linear ranker weight count mismatch")
    if len(model["normalization"]["mean"]) != summary["num_features"]:
        raise SystemExit("CSR linear ranker normalization mean count mismatch")
    if len(model["normalization"]["scale"]) != summary["num_features"]:
        raise SystemExit("CSR linear ranker normalization scale count mismatch")
    if set(model["weights"]) != set(model["feature_names"]):
        raise SystemExit("CSR linear ranker weight feature set mismatch")
    if any(not math.isfinite(float(value)) for value in model["weights"].values()):
        raise SystemExit("CSR linear ranker has nonfinite weight")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in model["normalization"]["scale"].values()
    ):
        raise SystemExit("CSR linear ranker has invalid normalization scale")
    if summary["num_epochs"] != 80:
        raise SystemExit("CSR linear ranker epoch count mismatch")
    if summary["num_pairwise_constraints"] != 110:
        raise SystemExit("CSR linear ranker pairwise constraint count mismatch")
    if summary["num_pairwise_updates"] != 148:
        raise SystemExit("CSR linear ranker pairwise update count mismatch")
    if not math.isfinite(float(summary["final_train_pairwise_loss"])):
        raise SystemExit("CSR linear ranker nonfinite train loss")
    if float(summary["train_oracle_top1_accuracy"]) != 0.5:
        raise SystemExit("CSR linear ranker train oracle accuracy mismatch")
    if float(summary["eval_oracle_top1_accuracy"]) != 0.0:
        raise SystemExit("CSR linear ranker eval oracle accuracy mismatch")
    if float(summary["eval_oracle_top1_accuracy_all_requests"]) != 0.0:
        raise SystemExit("CSR linear ranker eval all-request accuracy mismatch")
    if float(summary["eval_profiled_success_selection_rate"]) != 0.0:
        raise SystemExit("CSR linear ranker eval success selection rate mismatch")
    if int(summary["eval_non_success_selection_count"]) != 3:
        raise SystemExit("CSR linear ranker eval non-success count mismatch")
    if summary["eval_mean_regret_ms"] is not None:
        raise SystemExit("CSR linear ranker eval mean regret mismatch")
    if summary["eval_max_regret_ms"] is not None:
        raise SystemExit("CSR linear ranker eval max regret mismatch")
    if manifest.metadata["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR linear ranker manifest schema mismatch")
    if manifest.metadata["model_family"] != summary["model_family"]:
        raise SystemExit("CSR linear ranker manifest family mismatch")
    if manifest.metadata["model_trained"] is not True:
        raise SystemExit("CSR linear ranker manifest model flag mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR linear ranker manifest runtime flag mismatch")
    if manifest.metadata["num_features"] != summary["num_features"]:
        raise SystemExit("CSR linear ranker manifest feature count mismatch")
    if manifest.metadata["num_pairwise_constraints"] != summary["num_pairwise_constraints"]:
        raise SystemExit("CSR linear ranker manifest constraint count mismatch")
    if manifest.metadata["num_pairwise_updates"] != summary["num_pairwise_updates"]:
        raise SystemExit("CSR linear ranker manifest update count mismatch")
    expected_statuses = {
        "oracle_match",
        "non_success_selected",
        "no_oracle_target",
        "profiled_success_non_oracle",
    }
    if {row["evaluation_status"] for row in predictions} != expected_statuses:
        raise SystemExit("CSR linear ranker prediction status set mismatch")
    if sum(1 for row in predictions if row["split"] == "train") != 9:
        raise SystemExit("CSR linear ranker train prediction count mismatch")
    if sum(1 for row in predictions if row["split"] == "eval") != 3:
        raise SystemExit("CSR linear ranker eval prediction count mismatch")
    for row in predictions:
        ranked = tuple(row["ranked_candidate_ids"])
        scores = row["scores"]
        if row["schema_version"] != summary["schema_version"]:
            raise SystemExit("CSR linear ranker prediction schema mismatch")
        if row["model_id"] != summary["model_id"]:
            raise SystemExit("CSR linear ranker prediction model mismatch")
        if len(ranked) != 9 or len(scores) != 9:
            raise SystemExit("CSR linear ranker prediction candidate count mismatch")
        if len(set(ranked)) != 9 or set(ranked) != set(scores):
            raise SystemExit("CSR linear ranker prediction rank/score set mismatch")
        if ranked[0] != row["selected_candidate_id"]:
            raise SystemExit("CSR linear ranker selected candidate is not top-ranked")
        if abs(float(scores[ranked[0]]) - float(row["selected_score"])) > 1.0e-12:
            raise SystemExit("CSR linear ranker selected score mismatch")
        for left, right in zip(ranked, ranked[1:]):
            if float(scores[left]) < float(scores[right]):
                raise SystemExit("CSR linear ranker scores are not sorted")
        if row["evaluation_status"] not in expected_statuses:
            raise SystemExit("CSR linear ranker unexpected prediction status")
        if row["oracle_candidate_id"] is None and row["oracle_rank"] is not None:
            raise SystemExit("CSR linear ranker no-oracle row has oracle rank")
        if row["oracle_candidate_id"] is not None:
            if row["oracle_candidate_id"] not in ranked:
                raise SystemExit("CSR linear ranker oracle candidate missing from rank")
            if int(row["oracle_rank"]) != ranked.index(row["oracle_candidate_id"]) + 1:
                raise SystemExit("CSR linear ranker oracle rank mismatch")


def _verify_csr_selector_model_eval(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR selector model eval artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_selector_model_eval_rows.jsonl")
    summary = json.loads((path / "csr_selector_model_eval_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_selector_model_eval_schema.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_selector_model_quality_gate":
        raise SystemExit("unexpected CSR selector model eval artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR selector model eval did not pass artifact checks")
    if summary["schema_version"] != "phase1_csr_selector_model_eval_v1":
        raise SystemExit("unexpected CSR selector model eval schema version")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR selector model eval schema/summary mismatch")
    if schema["evaluation_id"] != summary["evaluation_id"]:
        raise SystemExit("CSR selector model eval id mismatch")
    if schema["runtime_selector_changed"] is not False:
        raise SystemExit("CSR selector model eval schema should not change runtime")
    if schema["integration_boundary"]["status"] != "offline_quality_gate_only":
        raise SystemExit("CSR selector model eval boundary mismatch")
    if summary["evaluation_id"] != "csr_selector_model_quality_gate_v1":
        raise SystemExit("CSR selector model eval id unexpected")
    if summary["baseline_model_id"] != "candidate_prior_success_median_v1":
        raise SystemExit("CSR selector model eval baseline mismatch")
    if summary["challenger_model_id"] != "csr_pairwise_linear_ranker_v1":
        raise SystemExit("CSR selector model eval challenger mismatch")
    if summary["best_offline_model_id"] != "candidate_prior_success_median_v1":
        raise SystemExit("CSR selector model eval best model mismatch")
    if summary["runtime_selected_model_id"] is not None:
        raise SystemExit("CSR selector model eval should not select runtime model")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR selector model eval summary should not change runtime")
    if summary["num_models"] != 2 or len(rows) != 2:
        raise SystemExit("CSR selector model eval expected two model rows")
    if summary["num_eval_predictions"] != 3 or summary["num_eval_oracle_requests"] != 2:
        raise SystemExit("CSR selector model eval eval count mismatch")
    if summary["min_required_eval_oracle_requests"] != 10:
        raise SystemExit("CSR selector model eval min oracle gate mismatch")
    if float(summary["min_runtime_oracle_top1_accuracy"]) != 0.5:
        raise SystemExit("CSR selector model eval oracle gate mismatch")
    if abs(float(summary["min_runtime_profiled_success_rate"]) - (2.0 / 3.0)) > 1.0e-12:
        raise SystemExit("CSR selector model eval success-rate gate mismatch")
    if summary["challenger_beats_baseline"] is not False:
        raise SystemExit("CSR selector model eval challenger comparison mismatch")
    if summary["challenger_runtime_eligible"] is not False:
        raise SystemExit("CSR selector model eval challenger should not be runtime eligible")
    expected_failures = {
        "insufficient_eval_oracle_requests:2<10",
        "below_min_oracle_top1",
        "below_min_profiled_success_rate",
        "below_baseline_oracle_top1",
        "below_baseline_profiled_success_rate",
        "non_success_eval_selections",
    }
    if set(summary["challenger_gate_failures"]) != expected_failures:
        raise SystemExit("CSR selector model eval gate failures mismatch")
    if summary["recommendation"] != "keep_runtime_artifact_backed_and_expand_benchmark_coverage":
        raise SystemExit("CSR selector model eval recommendation mismatch")
    if manifest.metadata["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR selector model eval manifest schema mismatch")
    if manifest.metadata["baseline_model_id"] != summary["baseline_model_id"]:
        raise SystemExit("CSR selector model eval manifest baseline mismatch")
    if manifest.metadata["challenger_model_id"] != summary["challenger_model_id"]:
        raise SystemExit("CSR selector model eval manifest challenger mismatch")
    if manifest.metadata["best_offline_model_id"] != summary["best_offline_model_id"]:
        raise SystemExit("CSR selector model eval manifest best model mismatch")
    if manifest.metadata["challenger_runtime_eligible"] is not False:
        raise SystemExit("CSR selector model eval manifest eligibility mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR selector model eval manifest runtime flag mismatch")
    rows_by_role = {row["comparison_role"]: row for row in rows}
    if set(rows_by_role) != {"baseline", "challenger"}:
        raise SystemExit("CSR selector model eval role set mismatch")
    baseline = rows_by_role["baseline"]
    challenger = rows_by_role["challenger"]
    if baseline["model_id"] != summary["baseline_model_id"]:
        raise SystemExit("CSR selector model eval baseline row id mismatch")
    if challenger["model_id"] != summary["challenger_model_id"]:
        raise SystemExit("CSR selector model eval challenger row id mismatch")
    if baseline["model_trained"] is not False or challenger["model_trained"] is not True:
        raise SystemExit("CSR selector model eval model_trained flags mismatch")
    if baseline["runtime_selector_changed"] is not False:
        raise SystemExit("CSR selector model eval baseline runtime flag mismatch")
    if challenger["runtime_selector_changed"] is not False:
        raise SystemExit("CSR selector model eval challenger runtime flag mismatch")
    if baseline["runtime_eligible"] is not False or challenger["runtime_eligible"] is not False:
        raise SystemExit("CSR selector model eval runtime eligibility mismatch")
    if baseline["gate_failures"] != ["comparison_baseline_not_runtime_candidate"]:
        raise SystemExit("CSR selector model eval baseline gate failure mismatch")
    if set(challenger["gate_failures"]) != expected_failures:
        raise SystemExit("CSR selector model eval challenger row failures mismatch")
    if challenger["beats_baseline"] is not False:
        raise SystemExit("CSR selector model eval challenger row comparison mismatch")
    if abs(float(baseline["eval_oracle_top1_accuracy"]) - (1.0 / 3.0)) > 1.0e-12:
        raise SystemExit("CSR selector model eval baseline accuracy mismatch")
    if abs(float(baseline["eval_profiled_success_selection_rate"]) - (2.0 / 3.0)) > 1.0e-12:
        raise SystemExit("CSR selector model eval baseline success rate mismatch")
    if int(baseline["eval_non_success_selection_count"]) != 1:
        raise SystemExit("CSR selector model eval baseline non-success count mismatch")
    if float(challenger["eval_oracle_top1_accuracy"]) != 0.0:
        raise SystemExit("CSR selector model eval challenger accuracy mismatch")
    if float(challenger["eval_profiled_success_selection_rate"]) != 0.0:
        raise SystemExit("CSR selector model eval challenger success rate mismatch")
    if int(challenger["eval_non_success_selection_count"]) != 3:
        raise SystemExit("CSR selector model eval challenger non-success count mismatch")
    for row in rows:
        if row["schema_version"] != summary["schema_version"]:
            raise SystemExit("CSR selector model eval row schema mismatch")
        if row["evaluation_id"] != summary["evaluation_id"]:
            raise SystemExit("CSR selector model eval row id mismatch")
        if int(row["num_eval_predictions"]) != summary["num_eval_predictions"]:
            raise SystemExit("CSR selector model eval row prediction count mismatch")
        if int(row["num_eval_oracle_requests"]) != summary["num_eval_oracle_requests"]:
            raise SystemExit("CSR selector model eval row oracle count mismatch")
        for key in ("eval_oracle_top1_accuracy", "eval_profiled_success_selection_rate"):
            value = float(row[key])
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise SystemExit("CSR selector model eval row metric invalid")


def _verify_csr_benchmark_expansion_plan(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR benchmark expansion plan artifact manifest: {stale}")
    matrix_rows = read_jsonl(path / "csr_benchmark_matrix_queue.jsonl")
    candidate_rows = read_jsonl(path / "csr_benchmark_candidate_queue.jsonl")
    summary = json.loads((path / "csr_benchmark_expansion_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_benchmark_expansion_schema.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_benchmark_expansion_plan":
        raise SystemExit("unexpected CSR benchmark expansion artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR benchmark expansion plan did not pass")
    if summary["schema_version"] != "phase1_csr_benchmark_expansion_plan_v1":
        raise SystemExit("unexpected CSR benchmark expansion schema version")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR benchmark expansion schema/summary mismatch")
    if schema["runtime_selector_changed"] is not False or schema["executes_gpu"] is not False:
        raise SystemExit("CSR benchmark expansion should be plan-only")
    if schema["integration_boundary"]["status"] != "plan_only_no_benchmark_execution":
        raise SystemExit("CSR benchmark expansion boundary mismatch")
    if summary["plan_id"] != "phase1_csr_benchmark_expansion_m49":
        raise SystemExit("CSR benchmark expansion plan id mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR benchmark expansion must not change runtime selector")
    if summary["executes_gpu"] is not False:
        raise SystemExit("CSR benchmark expansion plan should not execute GPU")
    if summary["selected_source_matrices"] != 64:
        raise SystemExit("CSR benchmark expansion source matrix count mismatch")
    if summary["already_profiled_matrices"] != 12:
        raise SystemExit("CSR benchmark expansion profiled count mismatch")
    if summary["eligible_unprofiled_matrices"] != 36:
        raise SystemExit("CSR benchmark expansion eligible count mismatch")
    if summary["planned_matrices"] != 8 or len(matrix_rows) != 8:
        raise SystemExit("CSR benchmark expansion matrix queue count mismatch")
    if summary["planned_candidate_jobs"] != 24 or len(candidate_rows) != 24:
        raise SystemExit("CSR benchmark expansion candidate queue count mismatch")
    if summary["planned_gpu_solve_attempts"] != 24:
        raise SystemExit("CSR benchmark expansion solve attempt count mismatch")
    if summary["measurement_repeats"] != 1:
        raise SystemExit("CSR benchmark expansion repeat count mismatch")
    if summary["max_new_matrices"] != 8 or summary["max_planned_gpu_solves"] != 24:
        raise SystemExit("CSR benchmark expansion budget mismatch")
    if summary["max_rows"] != 10_000 or summary["max_cols"] != 10_000:
        raise SystemExit("CSR benchmark expansion shape budget mismatch")
    if summary["max_nnz"] != 100_000:
        raise SystemExit("CSR benchmark expansion nnz budget mismatch")
    if summary["max_archive_size_bytes"] != 8_000_000:
        raise SystemExit("CSR benchmark expansion archive budget mismatch")
    if summary["max_iter"] != 256:
        raise SystemExit("CSR benchmark expansion max_iter mismatch")
    if float(summary["tolerance_rel"]) != 1.0e-5:
        raise SystemExit("CSR benchmark expansion tolerance mismatch")
    if summary["total_planned_nnz"] != 82789:
        raise SystemExit("CSR benchmark expansion total nnz mismatch")
    if summary["estimated_total_nnz_visits"] != 92343696:
        raise SystemExit("CSR benchmark expansion estimated nnz visits mismatch")
    if summary["by_candidate_profile"] != {"general": 4, "symmetric": 4}:
        raise SystemExit("CSR benchmark expansion profile split mismatch")
    if summary["by_solver"] != {
        "bicgstab": 8,
        "cg": 4,
        "gmres": 4,
        "pcg": 4,
        "richardson": 4,
    }:
        raise SystemExit("CSR benchmark expansion solver count mismatch")
    if summary["skipped_existing_matrices"] != 12:
        raise SystemExit("CSR benchmark expansion existing skip mismatch")
    if summary["skipped_resource_limit_matrices"] != 16:
        raise SystemExit("CSR benchmark expansion resource skip mismatch")
    if summary["skipped_budget_matrices"] != 28:
        raise SystemExit("CSR benchmark expansion budget skip mismatch")
    if manifest.metadata["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR benchmark expansion manifest schema mismatch")
    if manifest.metadata["planned_matrices"] != summary["planned_matrices"]:
        raise SystemExit("CSR benchmark expansion manifest matrix count mismatch")
    if manifest.metadata["planned_candidate_jobs"] != summary["planned_candidate_jobs"]:
        raise SystemExit("CSR benchmark expansion manifest candidate count mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR benchmark expansion manifest runtime flag mismatch")
    if manifest.metadata["executes_gpu"] is not False:
        raise SystemExit("CSR benchmark expansion manifest executes flag mismatch")
    expected_matrices = {
        "suitesparse:MathWorks/tomography",
        "suitesparse:HB/jgl009",
        "suitesparse:HB/ibm32",
        "suitesparse:SNAP/email-Eu-core",
        "suitesparse:Oberwolfach/t2dal_e",
        "suitesparse:HB/bcsstk07",
        "suitesparse:HB/lshp1009",
        "suitesparse:Gset/G17",
    }
    if {row["matrix_id"] for row in matrix_rows} != expected_matrices:
        raise SystemExit("CSR benchmark expansion matrix set mismatch")
    if sum(1 for row in matrix_rows if row["candidate_profile"] == "general") != 4:
        raise SystemExit("CSR benchmark expansion general matrix count mismatch")
    if sum(1 for row in matrix_rows if row["candidate_profile"] == "symmetric") != 4:
        raise SystemExit("CSR benchmark expansion symmetric matrix count mismatch")
    if any(row["planned_import"] is not True for row in matrix_rows):
        raise SystemExit("CSR benchmark expansion matrix import flag mismatch")
    if any(row["planned_gpu_benchmark"] is not True for row in matrix_rows):
        raise SystemExit("CSR benchmark expansion matrix benchmark flag mismatch")
    if any(row["skipped_reason"] is not None for row in matrix_rows):
        raise SystemExit("CSR benchmark expansion planned matrix has skip reason")
    if any(int(row["n_rows"]) > summary["max_rows"] for row in matrix_rows):
        raise SystemExit("CSR benchmark expansion planned row count exceeds budget")
    if any(int(row["n_cols"]) > summary["max_cols"] for row in matrix_rows):
        raise SystemExit("CSR benchmark expansion planned col count exceeds budget")
    if any(int(row["nnz"]) > summary["max_nnz"] for row in matrix_rows):
        raise SystemExit("CSR benchmark expansion planned nnz exceeds budget")
    if any(int(row["archive_size_bytes"]) > summary["max_archive_size_bytes"] for row in matrix_rows):
        raise SystemExit("CSR benchmark expansion planned archive exceeds budget")
    candidate_matrix_ids = {row["matrix_id"] for row in candidate_rows}
    if candidate_matrix_ids != expected_matrices:
        raise SystemExit("CSR benchmark expansion candidate matrix set mismatch")
    if any(row["planned_status"] != "queued_cpu_screen_required" for row in candidate_rows):
        raise SystemExit("CSR benchmark expansion candidate status mismatch")
    if any(row["requires_cpu_screen"] is not True for row in candidate_rows):
        raise SystemExit("CSR benchmark expansion CPU-screen flag mismatch")
    if any(row["measurement_repeats"] != 1 for row in candidate_rows):
        raise SystemExit("CSR benchmark expansion candidate repeat mismatch")
    if any(row["max_iter"] != 256 for row in candidate_rows):
        raise SystemExit("CSR benchmark expansion candidate max_iter mismatch")
    if any(row["precision"] != "float64" for row in candidate_rows):
        raise SystemExit("CSR benchmark expansion candidate precision mismatch")
    for row in candidate_rows:
        if row["solver"] == "gmres" and row["solver_parameters"] != {"restart": 16}:
            raise SystemExit("CSR benchmark expansion GMRES restart mismatch")
        if row["solver"] == "richardson" and row["solver_parameters"] != {
            "omega_source": "cpu_screen_required"
        }:
            raise SystemExit("CSR benchmark expansion Richardson parameter mismatch")
        if row["solver"] in {"cg", "pcg", "bicgstab"} and row["solver_parameters"] != {}:
            raise SystemExit("CSR benchmark expansion unexpected solver parameters")
    if sum(int(row["estimated_nnz_visits"]) for row in candidate_rows) != summary[
        "estimated_total_nnz_visits"
    ]:
        raise SystemExit("CSR benchmark expansion estimated work mismatch")


def _verify_csr_full_dataset_queue(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR full dataset queue artifact manifest: {stale}")
    matrix_rows = read_jsonl(path / "csr_full_dataset_queue_matrices.jsonl")
    job_rows = read_jsonl(path / "csr_full_dataset_queue_jobs.jsonl")
    batch_rows = read_jsonl(path / "csr_full_dataset_queue_batches.jsonl")
    state = json.loads((path / "csr_full_dataset_queue_state.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "csr_full_dataset_queue_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_full_dataset_queue_schema.json").read_text(encoding="utf-8"))
    report = (path / "csr_full_dataset_queue_report.md").read_text(encoding="utf-8")
    if manifest.artifact_kind != "csr_full_dataset_queue":
        raise SystemExit("unexpected CSR full dataset queue artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR full dataset queue did not pass")
    if summary["schema_version"] != "phase1_csr_full_dataset_queue_v1":
        raise SystemExit("CSR full dataset queue schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR full dataset queue schema/summary mismatch")
    if schema["runtime_selector_changed"] is not False:
        raise SystemExit("CSR full dataset queue changed runtime selector")
    if schema["executes_gpu"] is not False or schema["imports_matrices"] is not False:
        raise SystemExit("CSR full dataset queue should be plan-only")
    if schema["execution_boundary"]["automatic_execution"] is not False:
        raise SystemExit("CSR full dataset queue automatic execution mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR full dataset queue summary runtime mismatch")
    if summary["executes_gpu"] is not False or summary["imports_matrices"] is not False:
        raise SystemExit("CSR full dataset queue summary execution mismatch")
    if summary["resumable"] is not True:
        raise SystemExit("CSR full dataset queue should be resumable")
    if summary["external_drive_required"] is not True:
        raise SystemExit("CSR full dataset queue should require external drive")
    if summary["index_matrices"] != 2904:
        raise SystemExit("CSR full dataset queue index count mismatch")
    if summary["index_present_archives"] != 2904:
        raise SystemExit("CSR full dataset queue present archive count mismatch")
    if summary["already_profiled_matrices"] < 12:
        raise SystemExit("CSR full dataset queue profiled count unexpectedly small")
    if summary["eligible_matrices"] <= summary["already_profiled_matrices"]:
        raise SystemExit("CSR full dataset queue lacks unprofiled eligible matrices")
    if summary["queued_matrices"] <= 0 or summary["queued_jobs"] <= 0:
        raise SystemExit("CSR full dataset queue is empty")
    if summary["queued_batches"] <= 0:
        raise SystemExit("CSR full dataset queue has no batches")
    if len(job_rows) != summary["queued_jobs"]:
        raise SystemExit("CSR full dataset queue job row count mismatch")
    if len(batch_rows) != summary["queued_batches"]:
        raise SystemExit("CSR full dataset queue batch row count mismatch")
    queued_matrix_rows = [row for row in matrix_rows if row["queue_status"] == "queued"]
    if len(queued_matrix_rows) != summary["queued_matrices"]:
        raise SystemExit("CSR full dataset queue matrix row count mismatch")
    if any(row["queue_status"] != "pending_import" for row in job_rows):
        raise SystemExit("CSR full dataset queue job status mismatch")
    if any(row["requires_import"] is not True for row in job_rows):
        raise SystemExit("CSR full dataset queue import flag mismatch")
    if any(row["requires_cpu_screen"] is not True for row in job_rows):
        raise SystemExit("CSR full dataset queue CPU-screen flag mismatch")
    if any(row["precision"] != "float64" for row in job_rows):
        raise SystemExit("CSR full dataset queue precision mismatch")
    if any(row["measurement_repeats"] != summary["measurement_repeats"] for row in job_rows):
        raise SystemExit("CSR full dataset queue repeat mismatch")
    if any(row["max_iter"] != summary["max_iter"] for row in job_rows):
        raise SystemExit("CSR full dataset queue max_iter mismatch")
    if any(row["queue_status"] != "pending" for row in batch_rows):
        raise SystemExit("CSR full dataset queue batch status mismatch")
    if sum(row["job_count"] for row in batch_rows) != summary["queued_jobs"]:
        raise SystemExit("CSR full dataset queue batch job total mismatch")
    if sum(row["matrix_count"] for row in batch_rows) != summary["queued_matrices"]:
        raise SystemExit("CSR full dataset queue batch matrix total mismatch")
    if sum(row["planned_gpu_solve_attempts"] for row in batch_rows) != summary[
        "planned_gpu_solve_attempts"
    ]:
        raise SystemExit("CSR full dataset queue batch attempt total mismatch")
    if sum(row["estimated_nnz_visits"] for row in job_rows) != summary[
        "estimated_total_nnz_visits"
    ]:
        raise SystemExit("CSR full dataset queue estimated work mismatch")
    if summary["first_pending_batch_id"] != batch_rows[0]["batch_id"]:
        raise SystemExit("CSR full dataset queue first batch mismatch")
    if state["resume_policy"]["first_pending_batch_id"] != summary["first_pending_batch_id"]:
        raise SystemExit("CSR full dataset queue state resume mismatch")
    if len(state["resume_policy"]["pending_batch_ids"]) != summary["queued_batches"]:
        raise SystemExit("CSR full dataset queue pending batch state mismatch")
    if "explicit_user_start_for_gpu_benchmark" not in set(state["execution_requires"]):
        raise SystemExit("CSR full dataset queue missing explicit execution guard")
    if "do_not_change_runtime_selector" not in set(state["failure_policy"]):
        raise SystemExit("CSR full dataset queue missing runtime failure guard")
    if "Execution Boundary" not in report:
        raise SystemExit("CSR full dataset queue report missing boundary")
    if manifest.metadata["queued_matrices"] != summary["queued_matrices"]:
        raise SystemExit("CSR full dataset queue manifest matrix mismatch")
    if manifest.metadata["queued_jobs"] != summary["queued_jobs"]:
        raise SystemExit("CSR full dataset queue manifest job mismatch")
    if manifest.metadata["queued_batches"] != summary["queued_batches"]:
        raise SystemExit("CSR full dataset queue manifest batch mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR full dataset queue manifest runtime mismatch")
    if manifest.metadata["executes_gpu"] is not False:
        raise SystemExit("CSR full dataset queue manifest execution mismatch")
    if manifest.metadata["imports_matrices"] is not False:
        raise SystemExit("CSR full dataset queue manifest import mismatch")
    if manifest.metadata["resumable"] is not True:
        raise SystemExit("CSR full dataset queue manifest resumable mismatch")


def _verify_completed_csr_queue_batches(runs_root: Path) -> None:
    completed_paths: list[Path] = []
    for path in sorted(runs_root.glob("phase1_csr_queue_batch_*"), key=lambda item: item.name):
        summary_path = path / "csr_queue_batch_execution_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "passed":
            completed_paths.append(path)
    if not completed_paths:
        raise SystemExit("no completed CSR queue batches found")
    exact_expectations = {
        "batch_00001": {
            "expected_batch_outcome": "profiled_with_gpu_success",
            "expected_planned_total_nnz": 15342,
            "expected_planned_max_matrix_nnz": 4054,
            "expected_gpu_success_rows": 2,
            "expected_cpu_screened_out_rows": 22,
            "expected_selector_oracle_rows": 2,
            "expected_resume_status": "completed",
        },
        "batch_00002": {
            "expected_batch_outcome": "screen_only_no_oracle",
            "expected_planned_total_nnz": 27563,
            "expected_planned_max_matrix_nnz": 6511,
            "expected_gpu_success_rows": 0,
            "expected_cpu_screened_out_rows": 24,
            "expected_selector_oracle_rows": 0,
            "expected_resume_status": "completed_no_oracle",
        },
        "batch_00003": {
            "expected_batch_outcome": "profiled_with_gpu_success",
            "expected_planned_total_nnz": 49029,
            "expected_planned_max_matrix_nnz": 21842,
            "expected_gpu_success_rows": 5,
            "expected_cpu_screened_out_rows": 19,
            "expected_selector_oracle_rows": 3,
            "expected_resume_status": "completed",
        },
    }
    for path in completed_paths:
        summary = json.loads(
            (path / "csr_queue_batch_execution_summary.json").read_text(
                encoding="utf-8"
            )
        )
        batch_id = str(summary["batch_id"])
        expected = exact_expectations.get(
            batch_id,
            {
                "expected_batch_outcome": str(summary["batch_outcome"]),
                "expected_planned_total_nnz": int(summary["planned_total_nnz"]),
                "expected_planned_max_matrix_nnz": int(summary["planned_max_matrix_nnz"]),
                "expected_gpu_success_rows": int(summary["gpu_success_rows"]),
                "expected_cpu_screened_out_rows": int(summary["cpu_screened_out_rows"]),
                "expected_selector_oracle_rows": int(summary["selector_oracle_rows"]),
                "expected_resume_status": str(summary["resume_status"]),
            },
        )
        _verify_csr_queue_batch_execution(
            path,
            expected_batch_id=batch_id,
            **expected,
        )


def _verify_csr_queue_batch_execution(
    path: Path,
    *,
    expected_batch_id: str,
    expected_batch_outcome: str,
    expected_planned_total_nnz: int,
    expected_planned_max_matrix_nnz: int,
    expected_gpu_success_rows: int,
    expected_cpu_screened_out_rows: int,
    expected_selector_oracle_rows: int,
    expected_resume_status: str,
) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR queue batch execution manifest: {stale}")
    batch_matrices = read_jsonl(path / "csr_queue_batch_matrix_queue.jsonl")
    batch_jobs = read_jsonl(path / "csr_queue_batch_candidate_queue.jsonl")
    csr_rows = read_jsonl(path / "csr_matrices.jsonl")
    results = read_jsonl(path / "csr_micro_campaign_results.jsonl")
    selector_rows = read_jsonl(path / "csr_micro_selector_rows.jsonl")
    summary = json.loads(
        (path / "csr_queue_batch_execution_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_queue_batch_execution_schema.json").read_text(encoding="utf-8")
    )
    state = json.loads(
        (path / "csr_queue_batch_execution_state.json").read_text(encoding="utf-8")
    )
    micro_summary = json.loads(
        (path / "csr_micro_campaign_summary.json").read_text(encoding="utf-8")
    )
    selector_summary = json.loads(
        (path / "csr_micro_selector_summary.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_queue_batch_execution":
        raise SystemExit("unexpected CSR queue batch execution artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR queue batch execution did not pass")
    if summary["schema_version"] != "phase1_csr_queue_batch_execution_v1":
        raise SystemExit("CSR queue batch execution schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR queue batch execution schema/summary mismatch")
    if summary.get("batch_outcome") != expected_batch_outcome:
        raise SystemExit("CSR queue batch execution outcome mismatch")
    if expected_batch_outcome == "profiled_with_gpu_success":
        if micro_summary["status"] != "passed" or selector_summary["status"] != "passed":
            raise SystemExit("CSR queue batch execution micro artifacts did not pass")
    elif expected_batch_outcome == "screen_only_no_oracle":
        if micro_summary["status"] != "failed" or selector_summary["status"] != "failed":
            raise SystemExit("CSR queue batch screen-only micro status mismatch")
    else:
        raise SystemExit("CSR queue batch execution unknown expected outcome")
    if summary["batch_id"] != expected_batch_id:
        raise SystemExit("CSR queue batch execution batch id mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue batch execution changed runtime selector")
    if summary["executes_gpu"] is not True or schema["executes_gpu"] is not True:
        raise SystemExit("CSR queue batch execution did not execute GPU")
    if summary["imports_matrices"] is not True or schema["imports_matrices"] is not True:
        raise SystemExit("CSR queue batch execution did not import matrices")
    if summary["cpu_screen_required"] is not True:
        raise SystemExit("CSR queue batch execution missing CPU screen")
    if summary["append_only_selector_rows"] is not True:
        raise SystemExit("CSR queue batch execution should be append-only")
    if summary["planned_matrices"] != 8 or summary["executed_matrices"] != 8:
        raise SystemExit("CSR queue batch execution matrix count mismatch")
    if summary["planned_jobs"] != 24 or summary["executed_jobs"] != 24:
        raise SystemExit("CSR queue batch execution job count mismatch")
    if len(batch_matrices) != 8 or len(csr_rows) != 8:
        raise SystemExit("CSR queue batch execution CSR row count mismatch")
    if len(batch_jobs) != 24 or len(results) != 24:
        raise SystemExit("CSR queue batch execution result row count mismatch")
    if summary["imported_matrices"] != 8:
        raise SystemExit("CSR queue batch execution import count mismatch")
    if summary["candidate_jobs"] != 24:
        raise SystemExit("CSR queue batch execution candidate count mismatch")
    if summary["gpu_failed_rows"] != 0:
        raise SystemExit("CSR queue batch execution contains GPU failures")
    if summary["gpu_success_rows"] != expected_gpu_success_rows:
        raise SystemExit("CSR queue batch execution success count mismatch")
    if summary["cpu_screened_out_rows"] != expected_cpu_screened_out_rows:
        raise SystemExit("CSR queue batch execution screen count mismatch")
    if summary["gpu_success_rows"] + summary["cpu_screened_out_rows"] != 24:
        raise SystemExit("CSR queue batch execution status total mismatch")
    if summary["selector_rows"] != 24 or len(selector_rows) != 24:
        raise SystemExit("CSR queue batch execution selector row count mismatch")
    if summary["selector_oracle_rows"] != expected_selector_oracle_rows:
        raise SystemExit("CSR queue batch execution oracle selector count mismatch")
    if summary["completed_without_oracle"] is not (
        expected_batch_outcome == "screen_only_no_oracle"
    ):
        raise SystemExit("CSR queue batch execution no-oracle flag mismatch")
    if summary["has_oracle_rows"] is not (expected_selector_oracle_rows > 0):
        raise SystemExit("CSR queue batch execution oracle flag mismatch")
    if summary["matrices_with_selector_rows"] != 8:
        raise SystemExit("CSR queue batch execution selector matrix count mismatch")
    if summary["planned_total_nnz"] != expected_planned_total_nnz:
        raise SystemExit("CSR queue batch execution planned nnz mismatch")
    if summary["planned_max_matrix_nnz"] != expected_planned_max_matrix_nnz:
        raise SystemExit("CSR queue batch execution max nnz mismatch")
    if expected_gpu_success_rows > 0:
        if float(summary["max_final_relative_residual"]) > 1.0e-5:
            raise SystemExit("CSR queue batch execution trace residual check failed")
        if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
            raise SystemExit("CSR queue batch execution CPU residual check failed")
        if float(summary["max_solution_relative_error"]) > 5.0e-3:
            raise SystemExit("CSR queue batch execution solution error check failed")
    elif (
        float(summary["max_final_relative_residual"]) != 0.0
        or float(summary["max_cpu_recomputed_relative_residual"]) != 0.0
        or float(summary["max_solution_relative_error"]) != 0.0
    ):
        raise SystemExit("CSR queue batch screen-only residual summary mismatch")
    gpu_rows = [row for row in results if row["backend"] == "taichi_gpu"]
    screen_rows = [row for row in results if row["backend"] == "cpu_reference_screen"]
    if len(gpu_rows) != summary["gpu_success_rows"]:
        raise SystemExit("CSR queue batch execution GPU row count mismatch")
    if len(screen_rows) != summary["cpu_screened_out_rows"]:
        raise SystemExit("CSR queue batch execution screen row count mismatch")
    for row in gpu_rows:
        if row["status"] != "success":
            raise SystemExit("CSR queue batch execution GPU row is not success")
        if row["success_rate"] != 1.0 or row["measurement_repeats"] != 1:
            raise SystemExit("CSR queue batch execution repeat aggregation mismatch")
        if float(row["final_relative_residual"]) > 1.0e-5:
            raise SystemExit("CSR queue batch execution GPU trace residual mismatch")
        if float(row["cpu_recomputed_relative_residual"]) > 1.0e-4:
            raise SystemExit("CSR queue batch execution GPU CPU residual mismatch")
        if float(row["solution_relative_error"]) > 5.0e-3:
            raise SystemExit("CSR queue batch execution GPU solution error mismatch")
    for row in screen_rows:
        if row["status"] != "screened_out":
            raise SystemExit("CSR queue batch execution screen row status mismatch")
        if not str(row.get("failure_reason", "")).startswith("cpu_reference_"):
            raise SystemExit("CSR queue batch execution screen row reason mismatch")
    if state["resume_status"] != expected_resume_status:
        raise SystemExit("CSR queue batch execution state not completed")
    if state["batch_outcome"] != expected_batch_outcome:
        raise SystemExit("CSR queue batch execution state outcome mismatch")
    if state["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue batch execution state runtime mismatch")
    if manifest.metadata["batch_id"] != summary["batch_id"]:
        raise SystemExit("CSR queue batch execution manifest batch mismatch")
    if manifest.metadata["batch_outcome"] != expected_batch_outcome:
        raise SystemExit("CSR queue batch execution manifest outcome mismatch")
    if manifest.metadata["gpu_success_rows"] != summary["gpu_success_rows"]:
        raise SystemExit("CSR queue batch execution manifest success mismatch")
    if manifest.metadata["cpu_screened_out_rows"] != summary["cpu_screened_out_rows"]:
        raise SystemExit("CSR queue batch execution manifest screen mismatch")
    if manifest.metadata["executes_gpu"] is not True:
        raise SystemExit("CSR queue batch execution manifest GPU mismatch")
    if manifest.metadata["imports_matrices"] is not True:
        raise SystemExit("CSR queue batch execution manifest import mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue batch execution manifest runtime mismatch")


def _count_selector_target_status(rows: list[dict], status: str) -> int:
    return sum(1 for row in rows if row["target_status"] == status)


def _verify_csr_queue_training_pool(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR queue training pool manifest: {stale}")
    selector_rows = read_jsonl(path / "csr_queue_training_pool_selector_rows.jsonl")
    source_rows = read_jsonl(path / "csr_queue_training_pool_sources.jsonl")
    membership = read_jsonl(path / "csr_queue_training_pool_membership.jsonl")
    summary = json.loads(
        (path / "csr_queue_training_pool_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_queue_training_pool_schema.json").read_text(encoding="utf-8")
    )
    state = json.loads(
        (path / "csr_queue_training_pool_state.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_queue_training_pool":
        raise SystemExit("unexpected CSR queue training pool artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR queue training pool did not pass")
    if summary["schema_version"] != "phase1_csr_queue_training_pool_v1":
        raise SystemExit("CSR queue training pool schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR queue training pool schema/summary mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue training pool changed runtime selector")
    if summary["executes_gpu"] is not False or summary["imports_matrices"] is not False:
        raise SystemExit("CSR queue training pool should be non-executing")
    if summary["append_only"] is not True or summary["training_pool_ready"] is not True:
        raise SystemExit("CSR queue training pool append/ready flags mismatch")
    full_queue_batches = read_jsonl(summary["source_queue_batch_path"])
    queue_sources = [
        row for row in source_rows if row["source_kind"] == "queue_batch_execution"
    ]
    base_sources = [row for row in source_rows if row["source_kind"] == "base_selector"]
    completed_batch_ids = [row["batch_id"] for row in queue_sources]
    pending_batch_ids = [
        row["batch_id"]
        for row in full_queue_batches
        if row["batch_id"] not in set(completed_batch_ids)
    ]
    if summary["source_count"] != len(source_rows):
        raise SystemExit("CSR queue training pool source count mismatch")
    if summary["base_source_count"] != len(base_sources) or len(base_sources) != 2:
        raise SystemExit("CSR queue training pool source split mismatch")
    if summary["queue_batch_source_count"] != len(queue_sources):
        raise SystemExit("CSR queue training pool source split mismatch")
    if summary["completed_queue_batches"] != len(queue_sources):
        raise SystemExit("CSR queue training pool completed batch count mismatch")
    if summary["completed_batch_ids"] != completed_batch_ids:
        raise SystemExit("CSR queue training pool completed batch id mismatch")
    if summary["queue_batch_oracle_batches"] != sum(
        1 for row in queue_sources if row["batch_outcome"] == "profiled_with_gpu_success"
    ):
        raise SystemExit("CSR queue training pool oracle batch count mismatch")
    if summary["queue_batch_screen_only_batches"] != sum(
        1 for row in queue_sources if row["batch_outcome"] == "screen_only_no_oracle"
    ):
        raise SystemExit("CSR queue training pool screen-only batch count mismatch")
    if summary["full_queue_batches"] != len(full_queue_batches):
        raise SystemExit("CSR queue training pool full queue progress mismatch")
    if summary["remaining_queue_batches"] != len(pending_batch_ids):
        raise SystemExit("CSR queue training pool full queue progress mismatch")
    if summary["next_pending_batch_id"] != pending_batch_ids[0]:
        raise SystemExit("CSR queue training pool next batch mismatch")
    if summary["selector_rows"] != len(selector_rows):
        raise SystemExit("CSR queue training pool selector count mismatch")
    if summary["membership_rows"] != len(membership) or len(membership) != len(selector_rows):
        raise SystemExit("CSR queue training pool membership count mismatch")
    if summary["matrices"] != len({row["matrix_id"] for row in selector_rows}):
        raise SystemExit("CSR queue training pool matrix count mismatch")
    if summary["success_rows"] != _count_selector_target_status(selector_rows, "success"):
        raise SystemExit("CSR queue training pool success count mismatch")
    if summary["screened_out_rows"] != _count_selector_target_status(selector_rows, "screened_out"):
        raise SystemExit("CSR queue training pool screen count mismatch")
    if summary["not_profiled_rows"] != _count_selector_target_status(selector_rows, "not_profiled"):
        raise SystemExit("CSR queue training pool not-profiled count mismatch")
    if summary["not_applicable_rows"] != _count_selector_target_status(selector_rows, "not_applicable"):
        raise SystemExit("CSR queue training pool not-applicable count mismatch")
    if summary["oracle_rows"] != sum(1 for row in selector_rows if row["label_is_oracle"]):
        raise SystemExit("CSR queue training pool oracle count mismatch")
    if summary["queue_batch_selector_rows"] != sum(row["selector_rows"] for row in queue_sources):
        raise SystemExit("CSR queue training pool queue selector count mismatch")
    if summary["non_queue_selector_rows"] != sum(row["selector_rows"] for row in base_sources):
        raise SystemExit("CSR queue training pool non-queue selector count mismatch")
    sources_by_id = {row["source_id"]: row for row in source_rows}
    if not {
        "base_01",
        "base_02",
        "queue_batch_00001",
        "queue_batch_00002",
        "queue_batch_00003",
    } <= set(sources_by_id):
        raise SystemExit("CSR queue training pool source ids mismatch")
    if sources_by_id["base_01"]["selector_rows"] != 108:
        raise SystemExit("CSR queue training pool base_01 rows mismatch")
    if sources_by_id["base_02"]["selector_rows"] != 24:
        raise SystemExit("CSR queue training pool base_02 rows mismatch")
    if sources_by_id["queue_batch_00001"]["selector_rows"] != 24:
        raise SystemExit("CSR queue training pool queue batch rows mismatch")
    if sources_by_id["queue_batch_00002"]["selector_rows"] != 24:
        raise SystemExit("CSR queue training pool queue batch 00002 rows mismatch")
    if sources_by_id["queue_batch_00003"]["selector_rows"] != 24:
        raise SystemExit("CSR queue training pool queue batch 00003 rows mismatch")
    if sources_by_id["queue_batch_00001"]["executes_gpu"] is not True:
        raise SystemExit("CSR queue training pool queue source GPU flag mismatch")
    if sources_by_id["queue_batch_00002"]["executes_gpu"] is not True:
        raise SystemExit("CSR queue training pool queue source GPU flag mismatch")
    if sources_by_id["queue_batch_00003"]["executes_gpu"] is not True:
        raise SystemExit("CSR queue training pool queue source GPU flag mismatch")
    if sources_by_id["queue_batch_00001"]["batch_outcome"] != "profiled_with_gpu_success":
        raise SystemExit("CSR queue training pool queue 00001 outcome mismatch")
    if sources_by_id["queue_batch_00002"]["batch_outcome"] != "screen_only_no_oracle":
        raise SystemExit("CSR queue training pool queue 00002 outcome mismatch")
    if sources_by_id["queue_batch_00003"]["batch_outcome"] != "profiled_with_gpu_success":
        raise SystemExit("CSR queue training pool queue 00003 outcome mismatch")
    if sources_by_id["queue_batch_00002"]["completed_without_oracle"] is not True:
        raise SystemExit("CSR queue training pool queue 00002 no-oracle flag mismatch")
    if sources_by_id["queue_batch_00002"]["oracle_rows"] != 0:
        raise SystemExit("CSR queue training pool queue 00002 oracle mismatch")
    if sources_by_id["queue_batch_00003"]["oracle_rows"] != 3:
        raise SystemExit("CSR queue training pool queue 00003 oracle mismatch")
    for source in queue_sources:
        if source["selector_rows"] <= 0:
            raise SystemExit("CSR queue training pool queue source has no rows")
        if source["executes_gpu"] is not True:
            raise SystemExit("CSR queue training pool queue source GPU flag mismatch")
        if source["runtime_selector_changed"] is not False:
            raise SystemExit("CSR queue training pool queue source runtime mismatch")
    if sources_by_id["queue_batch_00001"]["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue training pool queue source runtime mismatch")
    if state["training_pool_ready"] is not True:
        raise SystemExit("CSR queue training pool state ready mismatch")
    if state["next_pending_batch_id"] != summary["next_pending_batch_id"]:
        raise SystemExit("CSR queue training pool state next batch mismatch")
    if manifest.metadata["training_pool_ready"] is not True:
        raise SystemExit("CSR queue training pool manifest ready mismatch")
    if manifest.metadata["selector_rows"] != summary["selector_rows"]:
        raise SystemExit("CSR queue training pool manifest selector mismatch")
    if manifest.metadata["completed_batch_ids"] != summary["completed_batch_ids"]:
        raise SystemExit("CSR queue training pool manifest completed batch mismatch")
    if (
        manifest.metadata["queue_batch_screen_only_batches"]
        != summary["queue_batch_screen_only_batches"]
    ):
        raise SystemExit("CSR queue training pool manifest screen-only mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue training pool manifest runtime mismatch")


def _verify_csr_queue_batch_training_bundle(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR queue batch training bundle manifest: {stale}")
    selector_rows = read_jsonl(path / "combined_csr_selector_rows.jsonl")
    requests = read_jsonl(path / "csr_transformer_model_requests.jsonl")
    targets = read_jsonl(path / "csr_transformer_model_targets.jsonl")
    request_index = read_jsonl(path / "csr_transformer_request_index.jsonl")
    summary = json.loads(
        (path / "csr_transformer_ready_summary.json").read_text(encoding="utf-8")
    )
    tensor_summary = json.loads(
        (path / "csr_transformer_tensor_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_transformer_ready_schema.json").read_text(encoding="utf-8")
    )
    arrays = json.loads(
        (path / "csr_transformer_training_tensors.json").read_text(encoding="utf-8")
    )
    pool_summary = json.loads(
        Path("runs/phase1_csr_queue_training_pool/csr_queue_training_pool_summary.json")
        .read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_queue_batch_training_bundle":
        raise SystemExit("unexpected CSR queue batch training bundle artifact kind")
    if summary["status"] != "passed" or summary["transformer_connectable"] is not True:
        raise SystemExit("CSR queue batch training bundle did not pass")
    if summary["schema_version"] != "phase1_csr_transformer_ready_v1":
        raise SystemExit("CSR queue batch training bundle schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR queue batch training bundle schema/summary mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue batch training bundle changed runtime selector")
    if summary["model_required"] is not False:
        raise SystemExit("CSR queue batch training bundle should not require a model")
    if len(summary["source_selector_paths"]) != 1:
        raise SystemExit("CSR queue batch training bundle source path count mismatch")
    if "runs/phase1_csr_queue_training_pool/csr_queue_training_pool_selector_rows.jsonl" not in set(
        summary["source_selector_paths"]
    ):
        raise SystemExit("CSR queue batch training bundle missing queue training pool source")
    if len(selector_rows) != summary["num_selector_rows"]:
        raise SystemExit("CSR queue batch training bundle selector count mismatch")
    if len(requests) != summary["num_model_requests"]:
        raise SystemExit("CSR queue batch training bundle request count mismatch")
    if len(targets) != summary["num_model_targets"]:
        raise SystemExit("CSR queue batch training bundle target count mismatch")
    if len(request_index) != summary["num_tensor_requests"]:
        raise SystemExit("CSR queue batch training bundle request-index count mismatch")
    if summary["num_selector_rows"] != pool_summary["selector_rows"]:
        raise SystemExit("CSR queue batch training bundle selector count mismatch")
    if summary["num_matrices"] != pool_summary["matrices"]:
        raise SystemExit("CSR queue batch training bundle matrix count mismatch")
    if summary["num_success_rows"] != pool_summary["success_rows"]:
        raise SystemExit("CSR queue batch training bundle success count mismatch")
    if summary["num_screened_out_rows"] != pool_summary["screened_out_rows"]:
        raise SystemExit("CSR queue batch training bundle screen count mismatch")
    if summary["num_oracle_rows"] != pool_summary["oracle_rows"]:
        raise SystemExit("CSR queue batch training bundle oracle count mismatch")
    if summary["num_learning_rows"] != len(selector_rows):
        raise SystemExit("CSR queue batch training bundle learning count mismatch")
    if (
        summary["num_learning_train_rows"] + summary["num_learning_eval_rows"]
        != len(selector_rows)
    ):
        raise SystemExit("CSR queue batch training bundle split count mismatch")
    if summary["num_global_candidates"] != 9:
        raise SystemExit("CSR queue batch training bundle candidate count mismatch")
    if summary["matrix_feature_dim"] != 21 or summary["candidate_feature_dim"] != 16:
        raise SystemExit("CSR queue batch training bundle feature dim mismatch")
    if summary["num_active_candidate_slots"] != len(selector_rows):
        raise SystemExit("CSR queue batch training bundle active slot mismatch")
    if summary["label_class_counts"]["success_oracle"] != pool_summary["oracle_rows"]:
        raise SystemExit("CSR queue batch training bundle label counts mismatch")
    if summary["label_class_counts"]["screened_out"] != pool_summary["screened_out_rows"]:
        raise SystemExit("CSR queue batch training bundle label counts mismatch")
    if summary["target_status_counts"] != {
        status: _count_selector_target_status(selector_rows, status)
        for status in ("not_applicable", "not_profiled", "screened_out", "success")
    }:
        raise SystemExit("CSR queue batch training bundle target counts mismatch")
    if not 0.0 <= float(summary["eval_oracle_top1_accuracy"]) <= 1.0:
        raise SystemExit("CSR queue batch training bundle oracle accuracy mismatch")
    if not 0.0 <= float(summary["eval_profiled_selection_rate"]) <= 1.0:
        raise SystemExit("CSR queue batch training bundle profiled rate mismatch")
    if summary["validation_error_count"] != 0:
        raise SystemExit("CSR queue batch training bundle validation errors found")
    if tensor_summary["status"] != "passed":
        raise SystemExit("CSR queue batch training tensor summary did not pass")
    if tensor_summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue batch training tensor changed runtime selector")
    if tensor_summary["model_required"] is not False:
        raise SystemExit("CSR queue batch training tensor should not require model")
    if tensor_summary["num_requests"] != summary["num_tensor_requests"]:
        raise SystemExit("CSR queue batch training tensor request mismatch")
    if tensor_summary["num_active_candidate_slots"] != summary["num_active_candidate_slots"]:
        raise SystemExit("CSR queue batch training active slot mismatch")
    if len(arrays["request_ids"]) != summary["num_tensor_requests"]:
        raise SystemExit("CSR queue batch training array request axis mismatch")
    if len(arrays["global_candidate_ids"]) != summary["num_global_candidates"]:
        raise SystemExit("CSR queue batch training candidate axis mismatch")
    if sum(sum(row) for row in arrays["candidate_mask"]) != summary[
        "num_active_candidate_slots"
    ]:
        raise SystemExit("CSR queue batch training candidate mask mismatch")
    if manifest.metadata["queue_batch_selector_rows"] != pool_summary[
        "queue_batch_selector_rows"
    ]:
        raise SystemExit("CSR queue batch training manifest queue row mismatch")
    if manifest.metadata["queue_pool_ready"] is not True:
        raise SystemExit("CSR queue batch training manifest pool ready mismatch")
    if manifest.metadata["queue_pool_summary"] != (
        "runs/phase1_csr_queue_training_pool/csr_queue_training_pool_summary.json"
    ):
        raise SystemExit("CSR queue batch training manifest pool path mismatch")
    if manifest.metadata["num_selector_rows"] != summary["num_selector_rows"]:
        raise SystemExit("CSR queue batch training manifest selector mismatch")
    if manifest.metadata["transformer_connectable"] is not True:
        raise SystemExit("CSR queue batch training manifest connectable mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue batch training manifest runtime mismatch")


def _verify_csr_queue_batch_reference_ranker(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR queue batch reference ranker manifest: {stale}")
    model = json.loads(
        (path / "csr_queue_batch_reference_ranker_model.json").read_text(
            encoding="utf-8"
        )
    )
    predictions = read_jsonl(path / "csr_queue_batch_reference_ranker_predictions.jsonl")
    summary = json.loads(
        (path / "csr_queue_batch_reference_ranker_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_queue_batch_reference_ranker_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_queue_batch_reference_ranker":
        raise SystemExit("unexpected CSR queue batch reference ranker artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR queue batch reference ranker did not pass")
    if summary["schema_version"] != "phase1_csr_transformer_ranker_v1":
        raise SystemExit("CSR queue batch reference ranker schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR queue batch reference ranker schema/summary mismatch")
    if summary["model_family"] != "masked_self_attention_ranker_v1":
        raise SystemExit("CSR queue batch reference ranker model family mismatch")
    if summary["model_id"] != "csr_queue_batch_shadow_ranker_v1":
        raise SystemExit("CSR queue batch reference ranker model id mismatch")
    if summary["model_trained"] is not True:
        raise SystemExit("CSR queue batch reference ranker did not train")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue batch reference ranker changed runtime selector")
    train_rows = [row for row in predictions if row["split"] == "train"]
    eval_rows = [row for row in predictions if row["split"] == "eval"]
    train_oracle_rows = [row for row in train_rows if row["oracle_candidate_id"]]
    eval_oracle_rows = [row for row in eval_rows if row["oracle_candidate_id"]]
    eval_profiled_success_rows = [
        row
        for row in eval_rows
        if row["evaluation_status"]
        in {"oracle_match", "profiled_success_non_oracle"}
    ]
    eval_non_success_rows = [
        row for row in eval_rows if row["selected_target_status"] != "success"
    ]
    if summary["num_requests"] != len(predictions) or summary["num_predictions"] != len(predictions):
        raise SystemExit("CSR queue batch reference ranker request count mismatch")
    if summary["num_train_requests"] != len(train_rows) or summary["num_eval_requests"] != len(eval_rows):
        raise SystemExit("CSR queue batch reference ranker split count mismatch")
    if (
        summary["num_train_oracle_requests"] != len(train_oracle_rows)
        or summary["num_eval_oracle_requests"] != len(eval_oracle_rows)
    ):
        raise SystemExit("CSR queue batch reference ranker oracle count mismatch")
    if summary["num_global_candidates"] != 9:
        raise SystemExit("CSR queue batch reference ranker candidate count mismatch")
    if summary["token_feature_dim"] != 37:
        raise SystemExit("CSR queue batch reference ranker token dim mismatch")
    if summary["d_model"] != 24 or summary["num_attention_heads"] != 4:
        raise SystemExit("CSR queue batch reference ranker attention shape mismatch")
    if summary["feedforward_dim"] != 48 or summary["scorer_feature_dim"] != 62:
        raise SystemExit("CSR queue batch reference ranker scorer dim mismatch")
    if summary["num_epochs"] != 160:
        raise SystemExit("CSR queue batch reference ranker epoch count mismatch")
    if abs(float(summary["learning_rate"]) - 0.03) > 1.0e-12:
        raise SystemExit("CSR queue batch reference ranker learning rate mismatch")
    if summary["num_pairwise_constraints"] <= 0:
        raise SystemExit("CSR queue batch reference ranker constraint count mismatch")
    if summary["num_pairwise_updates"] <= 0:
        raise SystemExit("CSR queue batch reference ranker update count mismatch")
    if not math.isfinite(float(summary["final_train_pairwise_loss"])):
        raise SystemExit("CSR queue batch reference ranker train loss mismatch")
    if not 0.0 <= float(summary["train_oracle_top1_accuracy"]) <= 1.0:
        raise SystemExit("CSR queue batch reference ranker train accuracy mismatch")
    if not 0.0 <= float(summary["eval_oracle_top1_accuracy"]) <= 1.0:
        raise SystemExit("CSR queue batch reference ranker eval oracle mismatch")
    if not 0.0 <= float(summary["eval_oracle_top1_accuracy_all_requests"]) <= 1.0:
        raise SystemExit("CSR queue batch reference ranker all-request accuracy mismatch")
    if abs(float(summary["eval_profiled_success_selection_rate"]) - (
        len(eval_profiled_success_rows) / len(eval_rows)
    )) > 1.0e-12:
        raise SystemExit("CSR queue batch reference ranker profiled rate mismatch")
    if summary["eval_non_success_selection_count"] != len(eval_non_success_rows):
        raise SystemExit("CSR queue batch reference ranker non-success count mismatch")
    if not math.isfinite(float(summary["eval_mean_regret_ms"])):
        raise SystemExit("CSR queue batch reference ranker mean regret mismatch")
    if not math.isfinite(float(summary["eval_max_regret_ms"])):
        raise SystemExit("CSR queue batch reference ranker max regret mismatch")
    if model["runtime_integration"]["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue batch reference ranker model runtime mismatch")
    if len(model["scorer_head"]) != summary["scorer_feature_dim"]:
        raise SystemExit("CSR queue batch reference ranker scorer head mismatch")
    if len(model["token_feature_names"]) != summary["token_feature_dim"]:
        raise SystemExit("CSR queue batch reference ranker token features mismatch")
    if {row["evaluation_status"] for row in predictions} != {
        "oracle_match",
        "non_success_selected",
        "profiled_success_non_oracle",
        "no_oracle_target",
    }:
        raise SystemExit("CSR queue batch reference ranker status set mismatch")
    if sum(1 for row in eval_rows if row["evaluation_status"] == "oracle_match") > len(eval_oracle_rows):
        raise SystemExit("CSR queue batch reference ranker eval oracle-match mismatch")
    for row in predictions:
        ranked = row["ranked_candidate_ids"]
        if not ranked or row["selected_candidate_id"] != ranked[0]:
            raise SystemExit("CSR queue batch reference ranker selected/rank mismatch")
        if len(ranked) != len(set(ranked)) or set(ranked) != set(row["scores"]):
            raise SystemExit("CSR queue batch reference ranker score/rank mismatch")
        if not all(math.isfinite(float(value)) for value in row["scores"].values()):
            raise SystemExit("CSR queue batch reference ranker nonfinite score")
    if manifest.metadata["shadow_only"] is not True:
        raise SystemExit("CSR queue batch reference ranker manifest shadow mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue batch reference ranker manifest runtime mismatch")
    if manifest.metadata["eval_non_success_selection_count"] != summary[
        "eval_non_success_selection_count"
    ]:
        raise SystemExit("CSR queue batch reference ranker manifest non-success mismatch")


def _verify_csr_queue_batch_model_replay(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR queue batch model replay manifest: {stale}")
    predictions = read_jsonl(path / "csr_queue_batch_model_replay_predictions.jsonl")
    comparison = read_jsonl(path / "csr_queue_batch_model_replay_comparison.jsonl")
    summary = json.loads(
        (path / "csr_queue_batch_model_replay_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_queue_batch_model_replay_schema.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_queue_batch_model_replay":
        raise SystemExit("unexpected CSR queue batch model replay artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR queue batch model replay did not pass")
    if summary["schema_version"] != "phase1_csr_transformer_model_replay_v1":
        raise SystemExit("CSR queue batch model replay schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR queue batch model replay schema/summary mismatch")
    if summary["source_model_schema_version"] != "phase1_csr_transformer_ranker_v1":
        raise SystemExit("CSR queue batch model replay source schema mismatch")
    if summary["model_family"] != "masked_self_attention_ranker_v1":
        raise SystemExit("CSR queue batch model replay model family mismatch")
    if summary["model_id"] != "csr_queue_batch_shadow_ranker_v1":
        raise SystemExit("CSR queue batch model replay model id mismatch")
    if summary["model_loaded"] is not True or summary["model_trained"] is not True:
        raise SystemExit("CSR queue batch model replay load/train flags mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue batch model replay changed runtime selector")
    if summary["num_predictions"] != len(predictions):
        raise SystemExit("CSR queue batch model replay count mismatch")
    if summary["num_reference_predictions"] != len(comparison):
        raise SystemExit("CSR queue batch model replay count mismatch")
    if len(predictions) != len(comparison):
        raise SystemExit("CSR queue batch model replay row count mismatch")
    if summary["exact_replay"] is not True:
        raise SystemExit("CSR queue batch model replay was not exact")
    if summary["selected_candidate_mismatch_count"] != 0:
        raise SystemExit("CSR queue batch model replay selected mismatch")
    if summary["ranked_order_mismatch_count"] != 0:
        raise SystemExit("CSR queue batch model replay rank mismatch")
    if summary["evaluation_status_mismatch_count"] != 0:
        raise SystemExit("CSR queue batch model replay status mismatch")
    if summary["missing_reference_count"] != 0:
        raise SystemExit("CSR queue batch model replay missing reference")
    if float(summary["max_abs_score_delta"]) != 0.0:
        raise SystemExit("CSR queue batch model replay score delta mismatch")
    if summary["prediction_contract_valid"] is not True:
        raise SystemExit("CSR queue batch model replay prediction contract invalid")
    if any(row["selected_candidate_match"] is not True for row in comparison):
        raise SystemExit("CSR queue batch model replay selected comparison mismatch")
    if any(row["ranked_order_match"] is not True for row in comparison):
        raise SystemExit("CSR queue batch model replay rank comparison mismatch")
    if any(row["evaluation_status_match"] is not True for row in comparison):
        raise SystemExit("CSR queue batch model replay status comparison mismatch")
    if any(float(row["max_abs_score_delta"]) != 0.0 for row in comparison):
        raise SystemExit("CSR queue batch model replay comparison score mismatch")
    if manifest.metadata["shadow_only"] is not True:
        raise SystemExit("CSR queue batch model replay manifest shadow mismatch")
    if manifest.metadata["exact_replay"] is not True:
        raise SystemExit("CSR queue batch model replay manifest exact mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue batch model replay manifest runtime mismatch")


def _verify_csr_queue_candidate_coverage(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR queue candidate coverage manifest: {stale}")
    batch_rows = read_jsonl(path / "csr_queue_candidate_coverage_batch_rows.jsonl")
    candidate_rows = read_jsonl(
        path / "csr_queue_candidate_coverage_candidate_rows.jsonl"
    )
    gap_rows = read_jsonl(path / "csr_queue_candidate_coverage_gap_rows.jsonl")
    summary = json.loads(
        (path / "csr_queue_candidate_coverage_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_queue_candidate_coverage_schema.json").read_text(
            encoding="utf-8"
        )
    )
    pool_summary = json.loads(
        Path("runs/phase1_csr_queue_training_pool/csr_queue_training_pool_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_queue_candidate_coverage":
        raise SystemExit("unexpected CSR queue candidate coverage artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR queue candidate coverage did not pass")
    if summary["schema_version"] != "phase1_csr_queue_candidate_coverage_v1":
        raise SystemExit("CSR queue candidate coverage schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR queue candidate coverage schema/summary mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue candidate coverage changed runtime selector")
    if summary["executes_gpu"] is not False or summary["imports_matrices"] is not False:
        raise SystemExit("CSR queue candidate coverage should be diagnostic-only")
    if schema["integration_boundary"]["does_not_modify_full_dataset_queue"] is not True:
        raise SystemExit("CSR queue candidate coverage boundary mismatch")
    if summary["completed_queue_batches"] != pool_summary["completed_queue_batches"]:
        raise SystemExit("CSR queue candidate coverage completed batch mismatch")
    if summary["next_pending_batch_id"] != pool_summary["next_pending_batch_id"]:
        raise SystemExit("CSR queue candidate coverage next batch mismatch")
    if len(batch_rows) != summary["completed_queue_batches"]:
        raise SystemExit("CSR queue candidate coverage batch row count mismatch")
    if summary["source_batch_count"] != len(batch_rows):
        raise SystemExit("CSR queue candidate coverage source count mismatch")
    if batch_rows[-1]["batch_id"] != summary["latest_batch_id"]:
        raise SystemExit("CSR queue candidate coverage latest batch mismatch")
    if len(candidate_rows) != summary["candidate_coverage_rows"]:
        raise SystemExit("CSR queue candidate coverage candidate row count mismatch")
    if len(gap_rows) != summary["planned_gap_count"]:
        raise SystemExit("CSR queue candidate coverage gap row count mismatch")
    if sum(1 for row in gap_rows if row["queue_ready"]) != summary["queue_ready_gap_count"]:
        raise SystemExit("CSR queue candidate coverage queue-ready count mismatch")
    if sum(1 for row in gap_rows if not row["queue_ready"]) != summary[
        "cpu_screen_blocked_gap_count"
    ]:
        raise SystemExit("CSR queue candidate coverage blocked count mismatch")
    if "chebyshev" not in set(summary["queue_missing_supported_solvers"]):
        raise SystemExit("CSR queue candidate coverage missed Chebyshev gap")
    if "ilu0" not in set(summary["queue_missing_supported_preconditioners"]):
        raise SystemExit("CSR queue candidate coverage missed ILU0 gap")
    if "row_column_equilibration" not in set(
        summary["queue_missing_supported_preconditioners"]
    ):
        raise SystemExit("CSR queue candidate coverage missed row/column gap")
    if "symmetric_equilibration" not in set(
        summary["queue_missing_supported_preconditioners"]
    ):
        raise SystemExit("CSR queue candidate coverage missed symmetric scaling gap")
    gap_ids = {row["gap_id"] for row in gap_rows}
    required_gaps = {
        "general_gmres_jacobi_restart32",
        "general_gmres_jacobi_restart64",
        "general_bicgstab_ilu0",
        "symmetric_chebyshev_jacobi",
        "symmetric_pcg_symmetric_equilibration",
    }
    if not required_gaps.issubset(gap_ids):
        raise SystemExit("CSR queue candidate coverage required gaps missing")
    queue_ready_ids = {row["gap_id"] for row in gap_rows if row["queue_ready"]}
    if {
        "general_gmres_jacobi_restart32",
        "general_gmres_jacobi_restart64",
    } - queue_ready_ids:
        raise SystemExit("CSR queue candidate coverage GMRES gaps not queue-ready")
    if any(row["queue_ready"] and row["blocker"] for row in gap_rows):
        raise SystemExit("CSR queue candidate coverage queue-ready blocker mismatch")
    if not any(row["blocker"] for row in gap_rows if not row["queue_ready"]):
        raise SystemExit("CSR queue candidate coverage blocked gaps missing blocker")
    if summary["latest_batch_outcome"] == "screen_only_no_oracle" and summary[
        "recent_screen_only_batches"
    ] < 1:
        raise SystemExit("CSR queue candidate coverage recent screen-only mismatch")
    if manifest.metadata["completed_queue_batches"] != summary["completed_queue_batches"]:
        raise SystemExit("CSR queue candidate coverage manifest batch mismatch")
    if manifest.metadata["planned_gap_count"] != summary["planned_gap_count"]:
        raise SystemExit("CSR queue candidate coverage manifest gap mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR queue candidate coverage manifest runtime mismatch")


def _verify_csr_gmres_restart_coverage(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR GMRES restart coverage manifest: {stale}")
    matrix_queue = read_jsonl(path / "csr_gmres_restart_matrix_queue.jsonl")
    candidate_queue = read_jsonl(path / "csr_gmres_restart_candidate_queue.jsonl")
    results = read_jsonl(path / "csr_micro_campaign_results.jsonl")
    selector_rows = read_jsonl(path / "csr_micro_selector_rows.jsonl")
    summary = json.loads(
        (path / "csr_gmres_restart_coverage_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_gmres_restart_coverage_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_gmres_restart_coverage":
        raise SystemExit("unexpected CSR GMRES restart coverage artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR GMRES restart coverage did not pass")
    if summary["schema_version"] != "phase1_csr_gmres_restart_coverage_v1":
        raise SystemExit("CSR GMRES restart coverage schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR GMRES restart coverage schema/summary mismatch")
    if schema["coverage_boundary"]["does_not_promote_runtime_selector"] is not True:
        raise SystemExit("CSR GMRES restart coverage promoted runtime selector")
    if schema["coverage_boundary"]["requires_cpu_screen_before_gpu_solve"] is not True:
        raise SystemExit("CSR GMRES restart coverage missing CPU screen boundary")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR GMRES restart coverage changed runtime selector")
    if summary["executes_gpu"] is not True or summary["imports_matrices"] is not True:
        raise SystemExit("CSR GMRES restart coverage execution flags mismatch")
    if summary["cpu_screen_required"] is not True:
        raise SystemExit("CSR GMRES restart coverage CPU screen flag mismatch")
    if summary["coverage_source_latest_batch_id"] != "batch_00010":
        raise SystemExit("CSR GMRES restart coverage source batch mismatch")
    if summary["coverage_trigger_candidate_coverage"] is not True:
        raise SystemExit("CSR GMRES restart coverage trigger mismatch")
    if summary["selected_matrices"] != len(matrix_queue) or summary["selected_matrices"] != 7:
        raise SystemExit("CSR GMRES restart coverage selected matrix count mismatch")
    if summary["candidate_jobs"] != 14:
        raise SystemExit("CSR GMRES restart coverage candidate job count mismatch")
    if len(candidate_queue) != summary["candidate_jobs"] or len(results) != summary["candidate_jobs"]:
        raise SystemExit("CSR GMRES restart coverage result count mismatch")
    if summary["selector_rows"] != len(selector_rows) or len(selector_rows) != summary["candidate_jobs"]:
        raise SystemExit("CSR GMRES restart coverage selector row count mismatch")
    if set(summary["restarts"]) != {32, 64}:
        raise SystemExit("CSR GMRES restart coverage restart set mismatch")
    if {row["solver"] for row in candidate_queue} != {"gmres"}:
        raise SystemExit("CSR GMRES restart coverage solver mismatch")
    if {row["preconditioner"] for row in candidate_queue} != {"jacobi"}:
        raise SystemExit("CSR GMRES restart coverage preconditioner mismatch")
    if {int(row["solver_parameters"]["restart"]) for row in candidate_queue} != {32, 64}:
        raise SystemExit("CSR GMRES restart coverage candidate restart mismatch")
    accounted = (
        summary["gpu_success_rows"]
        + summary["cpu_screened_out_rows"]
        + summary["gpu_failed_rows"]
    )
    if accounted != summary["candidate_jobs"]:
        raise SystemExit("CSR GMRES restart coverage accounting mismatch")
    if summary["gpu_failed_rows"] != 0:
        raise SystemExit("CSR GMRES restart coverage contains GPU failures")
    expected_merge_ready = summary["gpu_success_rows"] > 0 and summary["selector_oracle_rows"] > 0
    if summary["queue_merge_ready"] != expected_merge_ready:
        raise SystemExit("CSR GMRES restart coverage merge readiness mismatch")
    if set(summary["by_restart"]) != {"32", "64"}:
        raise SystemExit("CSR GMRES restart coverage by-restart keys mismatch")
    for restart in ("32", "64"):
        if sum(summary["by_restart"][restart].values()) != summary["selected_matrices"]:
            raise SystemExit("CSR GMRES restart coverage by-restart count mismatch")
    if summary["gpu_success_rows"] > 0:
        if summary["coverage_outcome"] != "profiled_with_gpu_success":
            raise SystemExit("CSR GMRES restart coverage success outcome mismatch")
        if summary["max_final_relative_residual"] > 1.0e-5:
            raise SystemExit("CSR GMRES restart coverage final residual check failed")
        if summary["max_cpu_recomputed_relative_residual"] > 1.0e-4:
            raise SystemExit("CSR GMRES restart coverage CPU residual check failed")
        if summary["max_solution_relative_error"] > 5.0e-3:
            raise SystemExit("CSR GMRES restart coverage solution error check failed")
    elif summary["coverage_outcome"] != "screen_only_no_oracle":
        raise SystemExit("CSR GMRES restart coverage screen-only outcome mismatch")
    if manifest.metadata["candidate_jobs"] != summary["candidate_jobs"]:
        raise SystemExit("CSR GMRES restart coverage manifest job mismatch")
    if manifest.metadata["queue_merge_ready"] != summary["queue_merge_ready"]:
        raise SystemExit("CSR GMRES restart coverage manifest merge mismatch")


def _verify_csr_blocked_gap_probe(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR blocked gap probe manifest: {stale}")
    matrix_queue = read_jsonl(path / "csr_blocked_gap_matrix_queue.jsonl")
    candidate_queue = read_jsonl(path / "csr_blocked_gap_candidate_queue.jsonl")
    results = read_jsonl(path / "csr_micro_campaign_results.jsonl")
    selector_rows = read_jsonl(path / "csr_micro_selector_rows.jsonl")
    summary = json.loads(
        (path / "csr_blocked_gap_probe_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_blocked_gap_probe_schema.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_blocked_gap_probe":
        raise SystemExit("unexpected CSR blocked gap probe artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR blocked gap probe did not pass")
    if summary["schema_version"] != "phase1_csr_blocked_gap_probe_v1":
        raise SystemExit("CSR blocked gap probe schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR blocked gap probe schema/summary mismatch")
    boundary = schema["coverage_boundary"]
    if boundary["does_not_promote_runtime_selector"] is not True:
        raise SystemExit("CSR blocked gap probe promoted runtime selector")
    if boundary["requires_cpu_screen_before_gpu_solve"] is not True:
        raise SystemExit("CSR blocked gap probe missing CPU screen boundary")
    if boundary["gpu_solve_only_after_screen_success"] is not True:
        raise SystemExit("CSR blocked gap probe GPU boundary mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR blocked gap probe changed runtime selector")
    if summary["executes_gpu"] is not True or summary["imports_matrices"] is not True:
        raise SystemExit("CSR blocked gap probe execution flags mismatch")
    if summary["cpu_screen_required"] is not True:
        raise SystemExit("CSR blocked gap probe CPU screen flag mismatch")
    if summary["coverage_source_latest_batch_id"] != "batch_00010":
        raise SystemExit("CSR blocked gap probe source batch mismatch")
    if summary["coverage_trigger_candidate_coverage"] is not True:
        raise SystemExit("CSR blocked gap probe trigger mismatch")
    if summary["selected_general_matrices"] != 7:
        raise SystemExit("CSR blocked gap probe general matrix count mismatch")
    if summary["selected_symmetric_matrices"] != 1:
        raise SystemExit("CSR blocked gap probe symmetric matrix count mismatch")
    if summary["selected_matrices"] != len(matrix_queue) or len(matrix_queue) != 8:
        raise SystemExit("CSR blocked gap probe selected matrix count mismatch")
    if summary["candidate_jobs"] != 16:
        raise SystemExit("CSR blocked gap probe candidate job count mismatch")
    if len(candidate_queue) != summary["candidate_jobs"] or len(results) != summary["candidate_jobs"]:
        raise SystemExit("CSR blocked gap probe result count mismatch")
    if summary["selector_rows"] != len(selector_rows) or len(selector_rows) != summary["candidate_jobs"]:
        raise SystemExit("CSR blocked gap probe selector row count mismatch")
    required_gaps = {
        "general_bicgstab_ilu0",
        "general_bicgstab_row_column_equilibration",
        "symmetric_chebyshev_jacobi",
        "symmetric_pcg_symmetric_equilibration",
    }
    if set(summary["blocked_gap_ids"]) != required_gaps:
        raise SystemExit("CSR blocked gap probe gap set mismatch")
    if summary["newly_cpu_screen_integrated_gap_count"] != 4:
        raise SystemExit("CSR blocked gap probe integrated gap count mismatch")
    accounted = (
        summary["gpu_success_rows"]
        + summary["cpu_screened_out_rows"]
        + summary["gpu_failed_rows"]
    )
    if accounted != summary["candidate_jobs"]:
        raise SystemExit("CSR blocked gap probe accounting mismatch")
    if summary["gpu_failed_rows"] != 0:
        raise SystemExit("CSR blocked gap probe contains GPU failures")
    if summary["coverage_outcome"] != "screen_only_no_oracle":
        raise SystemExit("CSR blocked gap probe outcome mismatch")
    if summary["gpu_success_rows"] != 0 or summary["cpu_screened_out_rows"] != 16:
        raise SystemExit("CSR blocked gap probe screen-only count mismatch")
    if summary["queue_merge_ready"] is not False or summary["queue_merge_ready_gap_ids"]:
        raise SystemExit("CSR blocked gap probe merge readiness mismatch")
    by_gap = summary["by_gap_status"]
    if by_gap.get("general_bicgstab_ilu0") != {"screened_out": 7}:
        raise SystemExit("CSR blocked gap probe ILU0 gap count mismatch")
    if by_gap.get("general_bicgstab_row_column_equilibration") != {"screened_out": 7}:
        raise SystemExit("CSR blocked gap probe row/column gap count mismatch")
    if by_gap.get("symmetric_chebyshev_jacobi") != {"screened_out": 1}:
        raise SystemExit("CSR blocked gap probe Chebyshev gap count mismatch")
    if by_gap.get("symmetric_pcg_symmetric_equilibration") != {"screened_out": 1}:
        raise SystemExit("CSR blocked gap probe symmetric gap count mismatch")
    if {row["backend"] for row in results} != {"cpu_reference_screen"}:
        raise SystemExit("CSR blocked gap probe unexpectedly executed GPU rows")
    if {row["status"] for row in results} != {"screened_out"}:
        raise SystemExit("CSR blocked gap probe result status mismatch")
    if manifest.metadata["candidate_jobs"] != summary["candidate_jobs"]:
        raise SystemExit("CSR blocked gap probe manifest job mismatch")
    if manifest.metadata["queue_merge_ready"] is not False:
        raise SystemExit("CSR blocked gap probe manifest merge mismatch")


def _verify_csr_blocked_gap_positive_search(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR blocked gap positive search manifest: {stale}")
    candidates = read_jsonl(path / "csr_blocked_gap_positive_candidates.jsonl")
    search_rows = read_jsonl(path / "csr_blocked_gap_positive_search_rows.jsonl")
    results = read_jsonl(path / "csr_blocked_gap_positive_results.jsonl")
    diagnostics = read_jsonl(path / "csr_blocked_gap_positive_diagnostic_rows.jsonl")
    selector_rows = read_jsonl(path / "csr_blocked_gap_positive_selector_rows.jsonl")
    summary = json.loads((path / "csr_blocked_gap_positive_search_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_blocked_gap_positive_search_schema.json").read_text(encoding="utf-8"))
    selector_summary = json.loads((path / "csr_blocked_gap_positive_selector_summary.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_blocked_gap_positive_search":
        raise SystemExit("unexpected CSR blocked gap positive search artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR blocked gap positive search did not pass")
    if summary["schema_version"] != "phase1_csr_blocked_gap_positive_search_v1":
        raise SystemExit("CSR blocked gap positive search schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR blocked gap positive search schema/summary mismatch")
    boundary = schema["execution_boundary"]
    if boundary["executes_gpu"] is not True or boundary["cpu_screen_before_gpu"] is not True:
        raise SystemExit("CSR blocked gap positive search execution boundary mismatch")
    if boundary["runtime_selector_changed"] is not False or boundary["generic_queue_merge"] is not False:
        raise SystemExit("CSR blocked gap positive search changed runtime/queue boundary")
    if summary["runtime_selector_changed"] is not False or summary["executes_gpu"] is not True:
        raise SystemExit("CSR blocked gap positive search runtime flags mismatch")
    if summary["queue_merge_ready"] is not False or summary["positive_evidence_found"] is not True:
        raise SystemExit("CSR blocked gap positive evidence/merge mismatch")
    required_gaps = {"general_bicgstab_ilu0", "general_bicgstab_row_column_equilibration", "symmetric_chebyshev_jacobi", "symmetric_pcg_symmetric_equilibration"}
    if set(summary["required_gap_ids"]) != required_gaps or set(summary["positive_gpu_gap_ids"]) != required_gaps:
        raise SystemExit("CSR blocked gap positive gap set mismatch")
    if summary["source_records"] != 91:
        raise SystemExit("CSR blocked gap positive source record count mismatch")
    if summary["search_rows"] != len(search_rows) or len(search_rows) != 50:
        raise SystemExit("CSR blocked gap positive search row count mismatch")
    if summary["selected_candidates"] != len(candidates) or len(candidates) != 8:
        raise SystemExit("CSR blocked gap positive selected count mismatch")
    if summary["candidate_jobs"] != len(results) or len(results) != 8:
        raise SystemExit("CSR blocked gap positive result count mismatch")
    if summary["selector_rows"] != len(selector_rows) or len(selector_rows) != 8 or len(diagnostics) != 8:
        raise SystemExit("CSR blocked gap positive selector/diagnostic count mismatch")
    if summary["selector_oracle_rows"] != 5:
        raise SystemExit("CSR blocked gap positive selector oracle count mismatch")
    if selector_summary["status"] != "passed" or selector_summary["num_selector_rows"] != len(selector_rows):
        raise SystemExit("CSR blocked gap positive selector summary mismatch")
    if selector_summary["num_oracle_rows"] != summary["selector_oracle_rows"]:
        raise SystemExit("CSR blocked gap positive selector oracle mismatch")
    if selector_summary["num_success_rows"] != summary["gpu_success_rows"]:
        raise SystemExit("CSR blocked gap positive selector success mismatch")
    if summary["gpu_success_rows"] != 8 or summary["gpu_failed_rows"] != 0:
        raise SystemExit("CSR blocked gap positive GPU count mismatch")
    if summary["by_status"] != {"success": 8}:
        raise SystemExit("CSR blocked gap positive status count mismatch")
    if summary["by_search_status"] != {"screened_out": 19, "selected_positive": 8, "skipped": 23}:
        raise SystemExit("CSR blocked gap positive search status count mismatch")
    if summary["by_gap_status"] != {"general_bicgstab_ilu0": {"success": 2}, "general_bicgstab_row_column_equilibration": {"success": 2}, "symmetric_chebyshev_jacobi": {"success": 2}, "symmetric_pcg_symmetric_equilibration": {"success": 2}}:
        raise SystemExit("CSR blocked gap positive by-gap count mismatch")
    if {row["backend"] for row in results} != {"taichi_gpu"}:
        raise SystemExit("CSR blocked gap positive contains non-GPU rows")
    if {row["status"] for row in results} != {"success"}:
        raise SystemExit("CSR blocked gap positive contains non-success rows")
    if {row["coverage_gap_id"] for row in results} != required_gaps or {row["coverage_gap_id"] for row in candidates} != required_gaps:
        raise SystemExit("CSR blocked gap positive row gap mismatch")
    if any(float(row["success_rate"]) != 1.0 for row in results):
        raise SystemExit("CSR blocked gap positive success-rate mismatch")
    if any(float(row["final_relative_residual"]) > 1.0e-5 for row in results):
        raise SystemExit("CSR blocked gap positive residual check failed")
    if any(float(row["cpu_recomputed_relative_residual"]) > 1.0e-5 for row in results):
        raise SystemExit("CSR blocked gap positive CPU residual check failed")
    if any(float(row["solution_relative_error"]) > 5.0e-3 for row in results):
        raise SystemExit("CSR blocked gap positive solution error check failed")
    if {(row["solver"], row["preconditioner"]) for row in results} != {("bicgstab", "ilu0"), ("bicgstab", "row_column_equilibration"), ("chebyshev", "jacobi"), ("pcg", "symmetric_equilibration")}: 
        raise SystemExit("CSR blocked gap positive solver/preconditioner mismatch")
    if set(summary["selected_matrix_ids"]) != {"suitesparse:FIDAP/ex5", "suitesparse:Grund/b1_ss", "suitesparse:HB/bcsstk01", "suitesparse:HB/curtis54", "suitesparse:JGD_Trefethen/Trefethen_20b"}:
        raise SystemExit("CSR blocked gap positive selected matrix mismatch")
    if manifest.metadata["candidate_jobs"] != summary["candidate_jobs"]:
        raise SystemExit("CSR blocked gap positive manifest job mismatch")
    if manifest.metadata["positive_evidence_found"] is not True or manifest.metadata["queue_merge_ready"] is not False:
        raise SystemExit("CSR blocked gap positive manifest evidence/merge mismatch")

def _verify_csr_micro_campaign(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR micro-campaign artifact manifest: {stale}")
    csr_rows = read_jsonl(path / "csr_matrices.jsonl")
    results = read_jsonl(path / "csr_micro_campaign_results.jsonl")
    selector_rows = read_jsonl(path / "csr_micro_selector_rows.jsonl")
    summary = json.loads((path / "csr_micro_campaign_summary.json").read_text(encoding="utf-8"))
    selector_summary = json.loads((path / "csr_micro_selector_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_micro_campaign_schema.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_micro_campaign":
        raise SystemExit("unexpected CSR micro-campaign artifact kind")
    if summary["status"] != "passed" or selector_summary["status"] != "passed":
        raise SystemExit("CSR micro-campaign did not pass")
    if summary["schema_version"] != "phase1_csr_micro_campaign_v1":
        raise SystemExit("CSR micro-campaign schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR micro-campaign schema/summary mismatch")
    if schema["executes_gpu"] is not True or schema["runtime_selector_changed"] is not False:
        raise SystemExit("CSR micro-campaign execution/runtime flags mismatch")
    if summary["imported_matrices"] != 8 or len(csr_rows) != 8:
        raise SystemExit("CSR micro-campaign import count mismatch")
    if summary["candidate_jobs"] != 24 or len(results) != 24:
        raise SystemExit("CSR micro-campaign result count mismatch")
    if summary["gpu_success_rows"] != 8:
        raise SystemExit("CSR micro-campaign GPU success count mismatch")
    if summary["cpu_screened_out_rows"] != 16:
        raise SystemExit("CSR micro-campaign screen-out count mismatch")
    if summary["gpu_failed_rows"] != 0:
        raise SystemExit("CSR micro-campaign contains GPU failures")
    if summary["selector_rows"] != 24 or len(selector_rows) != 24:
        raise SystemExit("CSR micro-campaign selector row count mismatch")
    if summary["selector_oracle_rows"] != 4:
        raise SystemExit("CSR micro-campaign oracle count mismatch")
    if summary["by_status"] != {"screened_out": 16, "success": 8}:
        raise SystemExit("CSR micro-campaign status count mismatch")
    if summary["by_solver_status"] != {
        "bicgstab:screened_out": 4,
        "bicgstab:success": 4,
        "cg:screened_out": 4,
        "gmres:screened_out": 2,
        "gmres:success": 2,
        "pcg:screened_out": 3,
        "pcg:success": 1,
        "richardson:screened_out": 3,
        "richardson:success": 1,
    }:
        raise SystemExit("CSR micro-campaign solver/status count mismatch")
    if float(summary["max_final_relative_residual"]) > 1.0e-5:
        raise SystemExit("CSR micro-campaign trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("CSR micro-campaign CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("CSR micro-campaign solution error check failed")
    if any(row["status"] == "failed" for row in results):
        raise SystemExit("CSR micro-campaign has failed result row")
    if sum(1 for row in results if row["backend"] == "taichi_gpu") != 8:
        raise SystemExit("CSR micro-campaign GPU row count mismatch")
    if sum(1 for row in results if row["backend"] == "cpu_reference_screen") != 16:
        raise SystemExit("CSR micro-campaign screen row count mismatch")
    if any(row["status"] != "success" for row in results if row["backend"] == "taichi_gpu"):
        raise SystemExit("CSR micro-campaign GPU row is not success")
    if any(row["success_rate"] != 1.0 for row in results if row["status"] == "success"):
        raise SystemExit("CSR micro-campaign success rows must have success_rate 1")
    if any(row["success_rate"] != 0.0 for row in results if row["status"] == "screened_out"):
        raise SystemExit("CSR micro-campaign screen rows must have success_rate 0")
    if manifest.metadata["gpu_success_rows"] != summary["gpu_success_rows"]:
        raise SystemExit("CSR micro-campaign manifest success count mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR micro-campaign manifest runtime flag mismatch")


def _verify_csr_transformer_ready(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR Transformer-ready artifact manifest: {stale}")
    selector_rows = read_jsonl(path / "combined_csr_selector_rows.jsonl")
    requests = read_jsonl(path / "csr_transformer_model_requests.jsonl")
    targets = read_jsonl(path / "csr_transformer_model_targets.jsonl")
    request_index = read_jsonl(path / "csr_transformer_request_index.jsonl")
    summary = json.loads((path / "csr_transformer_ready_summary.json").read_text(encoding="utf-8"))
    tensor_summary = json.loads((path / "csr_transformer_tensor_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_transformer_ready_schema.json").read_text(encoding="utf-8"))
    arrays = json.loads((path / "csr_transformer_training_tensors.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_transformer_ready_bundle":
        raise SystemExit("unexpected CSR Transformer-ready artifact kind")
    if summary["status"] != "passed" or summary["transformer_connectable"] is not True:
        raise SystemExit("CSR Transformer-ready bundle did not pass")
    if summary["schema_version"] != "phase1_csr_transformer_ready_v1":
        raise SystemExit("CSR Transformer-ready schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer-ready schema/summary mismatch")
    if schema["transformer_connectable"] is not True:
        raise SystemExit("CSR Transformer-ready schema not connectable")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer-ready should not change runtime selector")
    if summary["model_required"] is not False:
        raise SystemExit("CSR Transformer-ready should not require a model")
    if len(selector_rows) != 132 or summary["num_selector_rows"] != 132:
        raise SystemExit("CSR Transformer-ready selector row count mismatch")
    if summary["num_matrices"] != 20:
        raise SystemExit("CSR Transformer-ready matrix count mismatch")
    if summary["num_success_rows"] != 23:
        raise SystemExit("CSR Transformer-ready success count mismatch")
    if summary["num_screened_out_rows"] != 31:
        raise SystemExit("CSR Transformer-ready screen-out count mismatch")
    if summary["num_oracle_rows"] != 8:
        raise SystemExit("CSR Transformer-ready oracle count mismatch")
    if summary["num_model_requests"] != 20 or len(requests) != 20:
        raise SystemExit("CSR Transformer-ready request count mismatch")
    if summary["num_model_targets"] != 20 or len(targets) != 20:
        raise SystemExit("CSR Transformer-ready target count mismatch")
    if summary["num_tensor_requests"] != 20 or len(request_index) != 20:
        raise SystemExit("CSR Transformer-ready tensor request count mismatch")
    if summary["num_global_candidates"] != 9:
        raise SystemExit("CSR Transformer-ready global candidate count mismatch")
    if summary["num_active_candidate_slots"] != 132:
        raise SystemExit("CSR Transformer-ready active slot count mismatch")
    if summary["matrix_feature_dim"] != 21 or summary["candidate_feature_dim"] != 16:
        raise SystemExit("CSR Transformer-ready tensor feature dimensions mismatch")
    if summary["num_oracle_targets"] != 8 or summary["num_requests_without_oracle"] != 12:
        raise SystemExit("CSR Transformer-ready oracle target count mismatch")
    if summary["label_class_counts"] != {
        "not_applicable": 34,
        "not_profiled": 44,
        "screened_out": 31,
        "success_non_oracle": 15,
        "success_oracle": 8,
    }:
        raise SystemExit("CSR Transformer-ready label counts mismatch")
    if summary["target_status_counts"] != {
        "not_applicable": 34,
        "not_profiled": 44,
        "screened_out": 31,
        "success": 23,
    }:
        raise SystemExit("CSR Transformer-ready target status counts mismatch")
    if summary["validation_error_count"] != 0:
        raise SystemExit("CSR Transformer-ready validation errors found")
    if abs(float(summary["eval_oracle_top1_accuracy"]) - 0.6) > 1.0e-12:
        raise SystemExit("CSR Transformer-ready eval oracle accuracy mismatch")
    if abs(float(summary["eval_profiled_selection_rate"]) - 0.8) > 1.0e-12:
        raise SystemExit("CSR Transformer-ready profiled selection mismatch")
    if tensor_summary["has_missing_candidate_slots"] is not True:
        raise SystemExit("CSR Transformer-ready expected sparse candidate masks")
    if len(arrays["request_ids"]) != summary["num_tensor_requests"]:
        raise SystemExit("CSR Transformer-ready tensor request axis mismatch")
    if len(arrays["global_candidate_ids"]) != summary["num_global_candidates"]:
        raise SystemExit("CSR Transformer-ready tensor candidate axis mismatch")
    if len(arrays["matrix_features"]) != summary["num_tensor_requests"]:
        raise SystemExit("CSR Transformer-ready matrix feature rows mismatch")
    if any(len(row) != summary["matrix_feature_dim"] for row in arrays["matrix_features"]):
        raise SystemExit("CSR Transformer-ready matrix feature dim mismatch")
    if any(len(row) != summary["num_global_candidates"] for row in arrays["candidate_mask"]):
        raise SystemExit("CSR Transformer-ready candidate mask width mismatch")
    if sum(sum(row) for row in arrays["candidate_mask"]) != 132:
        raise SystemExit("CSR Transformer-ready candidate mask count mismatch")
    if manifest.metadata["transformer_connectable"] is not True:
        raise SystemExit("CSR Transformer-ready manifest connectable flag mismatch")


def _verify_csr_transformer_ranker(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR Transformer ranker artifact manifest: {stale}")
    predictions = read_jsonl(path / "csr_transformer_ranker_predictions.jsonl")
    summary = json.loads(
        (path / "csr_transformer_ranker_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_transformer_ranker_schema.json").read_text(encoding="utf-8")
    )
    model = json.loads(
        (path / "csr_transformer_ranker_model.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_transformer_ranker":
        raise SystemExit("unexpected CSR Transformer ranker artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR Transformer ranker did not pass")
    if summary["schema_version"] != "phase1_csr_transformer_ranker_v1":
        raise SystemExit("CSR Transformer ranker schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer ranker schema/summary mismatch")
    if model["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer ranker model schema mismatch")
    if summary["model_family"] != "masked_self_attention_ranker_v1":
        raise SystemExit("CSR Transformer ranker model family mismatch")
    if summary["model_id"] != "csr_masked_self_attention_ranker_v1":
        raise SystemExit("CSR Transformer ranker model id mismatch")
    if summary["model_trained"] is not True:
        raise SystemExit("CSR Transformer ranker expected trained model")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer ranker should not change runtime selector")
    if summary["num_requests"] != 20 or summary["num_predictions"] != 20:
        raise SystemExit("CSR Transformer ranker request count mismatch")
    if len(predictions) != summary["num_predictions"]:
        raise SystemExit("CSR Transformer ranker prediction count mismatch")
    if summary["num_train_requests"] != 15 or summary["num_eval_requests"] != 5:
        raise SystemExit("CSR Transformer ranker split count mismatch")
    if summary["num_train_oracle_requests"] != 4 or summary["num_eval_oracle_requests"] != 4:
        raise SystemExit("CSR Transformer ranker oracle count mismatch")
    if summary["num_global_candidates"] != 9:
        raise SystemExit("CSR Transformer ranker candidate count mismatch")
    if summary["token_feature_dim"] != 37:
        raise SystemExit("CSR Transformer ranker token feature dimension mismatch")
    if summary["d_model"] != 24 or summary["num_attention_heads"] != 4:
        raise SystemExit("CSR Transformer ranker attention shape mismatch")
    if summary["feedforward_dim"] != 48 or summary["scorer_feature_dim"] != 62:
        raise SystemExit("CSR Transformer ranker scorer shape mismatch")
    if summary["num_epochs"] != 160:
        raise SystemExit("CSR Transformer ranker epoch count mismatch")
    if abs(float(summary["learning_rate"]) - 0.03) > 1.0e-12:
        raise SystemExit("CSR Transformer ranker learning rate mismatch")
    if summary["num_pairwise_constraints"] != 116 or summary["num_pairwise_updates"] != 3264:
        raise SystemExit("CSR Transformer ranker pairwise update count mismatch")
    if abs(float(summary["train_oracle_top1_accuracy"]) - 0.5) > 1.0e-12:
        raise SystemExit("CSR Transformer ranker train oracle accuracy mismatch")
    if abs(float(summary["eval_oracle_top1_accuracy"]) - 0.5) > 1.0e-12:
        raise SystemExit("CSR Transformer ranker eval oracle accuracy mismatch")
    if abs(float(summary["eval_profiled_success_selection_rate"]) - 0.6) > 1.0e-12:
        raise SystemExit("CSR Transformer ranker eval success-rate mismatch")
    if int(summary["eval_non_success_selection_count"]) != 2:
        raise SystemExit("CSR Transformer ranker non-success count mismatch")
    if abs(float(summary["eval_mean_regret_ms"]) - 36.56358985851208) > 1.0e-9:
        raise SystemExit("CSR Transformer ranker mean regret mismatch")
    if abs(float(summary["eval_max_regret_ms"]) - 109.69076957553625) > 1.0e-9:
        raise SystemExit("CSR Transformer ranker max regret mismatch")
    eval_rows = [row for row in predictions if row["split"] == "eval"]
    if len(eval_rows) != summary["num_eval_requests"]:
        raise SystemExit("CSR Transformer ranker eval prediction count mismatch")
    if sum(1 for row in eval_rows if row["evaluation_status"] == "oracle_match") != 2:
        raise SystemExit("CSR Transformer ranker oracle-match count mismatch")
    if sum(1 for row in eval_rows if row["selected_target_status"] == "success") != 3:
        raise SystemExit("CSR Transformer ranker success selection count mismatch")
    for row in predictions:
        ranked = row["ranked_candidate_ids"]
        scores = row["scores"]
        if not ranked or len(ranked) != len(set(ranked)):
            raise SystemExit("CSR Transformer ranker invalid ranking")
        if row["selected_candidate_id"] != ranked[0]:
            raise SystemExit("CSR Transformer ranker selected candidate is not top-ranked")
        if set(ranked) != set(scores):
            raise SystemExit("CSR Transformer ranker score/ranking mismatch")
        if not all(math.isfinite(float(value)) for value in scores.values()):
            raise SystemExit("CSR Transformer ranker has non-finite score")
    if len(model["scorer_head"]) != summary["scorer_feature_dim"]:
        raise SystemExit("CSR Transformer ranker scorer head dimension mismatch")
    if manifest.metadata["model_trained"] is not True:
        raise SystemExit("CSR Transformer ranker manifest trained flag mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer ranker manifest runtime flag mismatch")
    if manifest.metadata["eval_non_success_selection_count"] != 2:
        raise SystemExit("CSR Transformer ranker manifest non-success count mismatch")


def _verify_csr_external_model_adapter(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR external model adapter manifest: {stale}")
    checkpoint = json.loads(
        (path / "external_csr_ranker_checkpoint.json").read_text(encoding="utf-8")
    )
    adapted_model = json.loads(
        (path / "adapted_csr_transformer_ranker_model.json").read_text(
            encoding="utf-8"
        )
    )
    predictions = read_jsonl(path / "adapted_csr_transformer_ranker_predictions.jsonl")
    ranker_summary = json.loads(
        (path / "adapted_csr_transformer_ranker_summary.json").read_text(
            encoding="utf-8"
        )
    )
    rows = read_jsonl(path / "csr_external_model_adapter_rows.jsonl")
    summary = json.loads(
        (path / "csr_external_model_adapter_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_external_model_adapter_schema.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_external_model_adapter":
        raise SystemExit("unexpected CSR external model adapter artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR external model adapter did not pass")
    if summary["schema_version"] != "phase1_csr_external_model_adapter_v1":
        raise SystemExit("CSR external model adapter schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR external model adapter schema/summary mismatch")
    if checkpoint["schema_version"] != "phase1_csr_external_ranker_checkpoint_v1":
        raise SystemExit("CSR external checkpoint schema mismatch")
    if checkpoint["checkpoint_kind"] != "csr_external_ranker_checkpoint":
        raise SystemExit("CSR external checkpoint kind mismatch")
    if checkpoint["adapter"] != "csr_external_json_ranker_checkpoint_v1":
        raise SystemExit("CSR external checkpoint adapter mismatch")
    if checkpoint["runtime_contract"]["guard_required"] is not True:
        raise SystemExit("CSR external checkpoint guard mismatch")
    if checkpoint["runtime_contract"]["runtime_selector_changed"] is not False:
        raise SystemExit("CSR external checkpoint changed runtime selector")
    if adapted_model["schema_version"] != "phase1_csr_transformer_ranker_v1":
        raise SystemExit("CSR adapted ranker schema mismatch")
    if adapted_model["model_family"] != "masked_self_attention_ranker_v1":
        raise SystemExit("CSR adapted ranker family mismatch")
    if ranker_summary["schema_version"] != adapted_model["schema_version"]:
        raise SystemExit("CSR adapted ranker summary schema mismatch")
    if ranker_summary["status"] != "passed":
        raise SystemExit("CSR adapted ranker summary did not pass")
    if summary["adapter_ready"] is not True:
        raise SystemExit("CSR external model adapter not ready")
    if summary["quality_gate_input_ready"] is not True:
        raise SystemExit("CSR external model adapter not quality-gate ready")
    if summary["policy_model_artifact_input_ready"] is not True:
        raise SystemExit("CSR external model adapter not model-artifact ready")
    if summary["prediction_contract_valid"] is not True:
        raise SystemExit("CSR external model adapter prediction contract mismatch")
    if summary["num_predictions"] != 20 or len(predictions) != 20:
        raise SystemExit("CSR external model adapter prediction count mismatch")
    if summary["num_eval_predictions"] != 5:
        raise SystemExit("CSR external model adapter eval count mismatch")
    if abs(float(summary["eval_oracle_top1_accuracy"]) - 0.5) > 1.0e-12:
        raise SystemExit("CSR external model adapter oracle accuracy mismatch")
    if abs(float(summary["eval_profiled_success_selection_rate"]) - 0.6) > 1.0e-12:
        raise SystemExit("CSR external model adapter success rate mismatch")
    if int(summary["eval_non_success_selection_count"]) != 2:
        raise SystemExit("CSR external model adapter non-success mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR external model adapter changed runtime selector")
    if summary["validation_error_count"] != 0:
        raise SystemExit("CSR external model adapter validation errors found")
    row_kinds = {row["row_kind"] for row in rows}
    if row_kinds != {
        "checkpoint_contract",
        "tensor_contract",
        "adapted_model",
        "prediction_contract",
        "runtime_boundary",
        "adapter_decision",
    }:
        raise SystemExit("CSR external model adapter row kind mismatch")
    for row in predictions:
        ranked = row["ranked_candidate_ids"]
        if not ranked or row["selected_candidate_id"] != ranked[0]:
            raise SystemExit("CSR external model adapter invalid prediction")
        if len(ranked) != len(set(ranked)) or set(ranked) != set(row["scores"]):
            raise SystemExit("CSR external model adapter score/rank mismatch")
    if manifest.metadata["adapter_ready"] is not True:
        raise SystemExit("CSR external model adapter manifest ready mismatch")
    if manifest.metadata["quality_gate_input_ready"] is not True:
        raise SystemExit("CSR external model adapter manifest quality mismatch")
    if manifest.metadata["policy_model_artifact_input_ready"] is not True:
        raise SystemExit("CSR external model adapter manifest model artifact mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR external model adapter manifest runtime mismatch")


def _verify_csr_transformer_model_replay(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR Transformer model replay artifact manifest: {stale}")
    predictions = read_jsonl(path / "csr_transformer_model_replay_predictions.jsonl")
    comparison = read_jsonl(path / "csr_transformer_model_replay_comparison.jsonl")
    summary = json.loads(
        (path / "csr_transformer_model_replay_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_transformer_model_replay_schema.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_transformer_model_replay":
        raise SystemExit("unexpected CSR Transformer model replay artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR Transformer model replay did not pass")
    if summary["schema_version"] != "phase1_csr_transformer_model_replay_v1":
        raise SystemExit("CSR Transformer model replay schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer model replay schema/summary mismatch")
    if summary["source_model_schema_version"] != "phase1_csr_transformer_ranker_v1":
        raise SystemExit("CSR Transformer model replay source schema mismatch")
    if summary["model_family"] != "masked_self_attention_ranker_v1":
        raise SystemExit("CSR Transformer model replay family mismatch")
    if summary["model_id"] != "csr_masked_self_attention_ranker_v1":
        raise SystemExit("CSR Transformer model replay model id mismatch")
    if summary["model_loaded"] is not True or summary["model_trained"] is not True:
        raise SystemExit("CSR Transformer model replay load/trained flag mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer model replay should not change runtime selector")
    if summary["num_predictions"] != 20 or len(predictions) != 20:
        raise SystemExit("CSR Transformer model replay prediction count mismatch")
    if summary["num_reference_predictions"] != 20:
        raise SystemExit("CSR Transformer model replay reference count mismatch")
    if len(comparison) != 20:
        raise SystemExit("CSR Transformer model replay comparison count mismatch")
    if summary["exact_replay"] is not True:
        raise SystemExit("CSR Transformer model replay expected exact replay")
    if summary["selected_candidate_mismatch_count"] != 0:
        raise SystemExit("CSR Transformer model replay selected mismatch")
    if summary["ranked_order_mismatch_count"] != 0:
        raise SystemExit("CSR Transformer model replay ranking mismatch")
    if summary["evaluation_status_mismatch_count"] != 0:
        raise SystemExit("CSR Transformer model replay status mismatch")
    if summary["missing_reference_count"] != 0:
        raise SystemExit("CSR Transformer model replay missing references")
    if float(summary["max_abs_score_delta"]) > 1.0e-9:
        raise SystemExit("CSR Transformer model replay score delta too large")
    if summary["prediction_contract_valid"] is not True:
        raise SystemExit("CSR Transformer model replay prediction contract invalid")
    if any(not row["selected_candidate_match"] for row in comparison):
        raise SystemExit("CSR Transformer model replay comparison selected mismatch")
    if any(not row["ranked_order_match"] for row in comparison):
        raise SystemExit("CSR Transformer model replay comparison rank mismatch")
    for row in predictions:
        ranked = row["ranked_candidate_ids"]
        if not ranked or row["selected_candidate_id"] != ranked[0]:
            raise SystemExit("CSR Transformer model replay invalid prediction ranking")
    if manifest.metadata["model_loaded"] is not True:
        raise SystemExit("CSR Transformer model replay manifest loaded mismatch")
    if manifest.metadata["exact_replay"] is not True:
        raise SystemExit("CSR Transformer model replay manifest exact replay mismatch")


def _verify_csr_transformer_quality_gate(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR Transformer quality gate artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_transformer_quality_gate_rows.jsonl")
    summary = json.loads(
        (path / "csr_transformer_quality_gate_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_transformer_quality_gate_schema.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_transformer_quality_gate":
        raise SystemExit("unexpected CSR Transformer quality gate artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR Transformer quality gate did not pass")
    if summary["schema_version"] != "phase1_csr_selector_model_eval_v1":
        raise SystemExit("CSR Transformer quality gate schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer quality gate schema/summary mismatch")
    if summary["evaluation_id"] != "csr_selector_model_quality_gate_v1":
        raise SystemExit("CSR Transformer quality gate evaluation id mismatch")
    if summary["baseline_model_id"] != "candidate_prior_success_median_v1":
        raise SystemExit("CSR Transformer quality gate baseline mismatch")
    if summary["challenger_model_id"] != "csr_masked_self_attention_ranker_v1":
        raise SystemExit("CSR Transformer quality gate challenger mismatch")
    if summary["best_offline_model_id"] != "candidate_prior_success_median_v1":
        raise SystemExit("CSR Transformer quality gate best model mismatch")
    if summary["runtime_selected_model_id"] is not None:
        raise SystemExit("CSR Transformer quality gate should not select runtime model")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer quality gate should not change runtime selector")
    if summary["challenger_beats_baseline"] is not False:
        raise SystemExit("CSR Transformer quality gate challenger comparison mismatch")
    if summary["challenger_runtime_eligible"] is not False:
        raise SystemExit("CSR Transformer quality gate should block runtime integration")
    expected_failures = {
        "below_min_profiled_success_rate",
        "below_baseline_oracle_top1",
        "below_baseline_profiled_success_rate",
        "non_success_eval_selections",
    }
    if set(summary["challenger_gate_failures"]) != expected_failures:
        raise SystemExit("CSR Transformer quality gate failures mismatch")
    if summary["recommendation"] != "keep_runtime_artifact_backed_and_expand_benchmark_coverage":
        raise SystemExit("CSR Transformer quality gate recommendation mismatch")
    if summary["num_models"] != 2 or summary["num_eval_predictions"] != 5:
        raise SystemExit("CSR Transformer quality gate model/eval count mismatch")
    if summary["num_eval_oracle_requests"] != 4:
        raise SystemExit("CSR Transformer quality gate oracle count mismatch")
    if summary["min_required_eval_oracle_requests"] != 4:
        raise SystemExit("CSR Transformer quality gate min oracle count mismatch")
    if abs(float(summary["min_runtime_oracle_top1_accuracy"]) - 0.5) > 1.0e-12:
        raise SystemExit("CSR Transformer quality gate min oracle accuracy mismatch")
    if abs(float(summary["min_runtime_profiled_success_rate"]) - 0.8) > 1.0e-12:
        raise SystemExit("CSR Transformer quality gate min success-rate mismatch")
    if len(rows) != 2:
        raise SystemExit("CSR Transformer quality gate row count mismatch")
    rows_by_role = {row["comparison_role"]: row for row in rows}
    if set(rows_by_role) != {"baseline", "challenger"}:
        raise SystemExit("CSR Transformer quality gate role mismatch")
    baseline = rows_by_role["baseline"]
    challenger = rows_by_role["challenger"]
    if baseline["runtime_eligible"] is not False or challenger["runtime_eligible"] is not False:
        raise SystemExit("CSR Transformer quality gate runtime eligibility mismatch")
    if challenger["beats_baseline"] is not False:
        raise SystemExit("CSR Transformer quality gate challenger comparison mismatch")
    if abs(float(baseline["eval_oracle_top1_accuracy"]) - 0.6) > 1.0e-12:
        raise SystemExit("CSR Transformer quality gate baseline oracle accuracy mismatch")
    if abs(float(challenger["eval_oracle_top1_accuracy"]) - 0.5) > 1.0e-12:
        raise SystemExit("CSR Transformer quality gate challenger oracle accuracy mismatch")
    if abs(float(baseline["eval_profiled_success_selection_rate"]) - 0.8) > 1.0e-12:
        raise SystemExit("CSR Transformer quality gate baseline success-rate mismatch")
    if abs(float(challenger["eval_profiled_success_selection_rate"]) - 0.6) > 1.0e-12:
        raise SystemExit("CSR Transformer quality gate challenger success-rate mismatch")
    if int(challenger["eval_non_success_selection_count"]) != 2:
        raise SystemExit("CSR Transformer quality gate challenger non-success mismatch")
    if manifest.metadata["challenger_runtime_eligible"] is not False:
        raise SystemExit("CSR Transformer quality gate manifest eligibility mismatch")
    if manifest.metadata["best_offline_model_id"] != summary["best_offline_model_id"]:
        raise SystemExit("CSR Transformer quality gate manifest best model mismatch")


def _verify_csr_policy_model_artifact(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR policy model artifact manifest: {stale}")
    artifact = json.loads(
        (path / "csr_policy_model_artifact.json").read_text(encoding="utf-8")
    )
    rows = read_jsonl(path / "csr_policy_model_artifact_rows.jsonl")
    summary = json.loads(
        (path / "csr_policy_model_artifact_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_policy_model_artifact_schema.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_policy_model_artifact":
        raise SystemExit("unexpected CSR policy model artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR policy model artifact did not pass")
    if summary["schema_version"] != "phase1_csr_policy_model_artifact_v1":
        raise SystemExit("CSR policy model artifact schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR policy model artifact schema/summary mismatch")
    if artifact["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR policy model artifact payload schema mismatch")
    if summary["artifact_ready"] is not True:
        raise SystemExit("CSR policy model artifact not ready")
    if summary["adapter"] != "csr_transformer_ranker_saved_model_v1":
        raise SystemExit("CSR policy model artifact adapter mismatch")
    if summary["model_id"] != "csr_masked_self_attention_ranker_v1":
        raise SystemExit("CSR policy model artifact model id mismatch")
    if summary["model_loaded"] is not True:
        raise SystemExit("CSR policy model artifact did not load model")
    if summary["prediction_contract_checked"] is not True:
        raise SystemExit("CSR policy model artifact prediction contract mismatch")
    if summary["num_predictions"] != 20 or summary["num_request_index_rows"] != 20:
        raise SystemExit("CSR policy model artifact prediction count mismatch")
    if summary["replay_exact"] is not True:
        raise SystemExit("CSR policy model artifact replay mismatch")
    if summary["guard_required"] is not True:
        raise SystemExit("CSR policy model artifact guard requirement mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR policy model artifact changed runtime selector")
    if summary["validation_error_count"] != 0:
        raise SystemExit("CSR policy model artifact validation errors present")
    if artifact["runtime_contract"]["guard_required"] is not True:
        raise SystemExit("CSR policy model artifact runtime guard mismatch")
    if artifact["runtime_contract"]["runtime_selector_changed"] is not False:
        raise SystemExit("CSR policy model artifact runtime selector mismatch")
    if {row["row_kind"] for row in rows} != {
        "model",
        "tensor_contract",
        "request_index",
        "quality_gate",
        "runtime_contract",
    }:
        raise SystemExit("CSR policy model artifact row kinds mismatch")
    if manifest.metadata["artifact_ready"] is not True:
        raise SystemExit("CSR policy model artifact manifest ready mismatch")
    if manifest.metadata["model_loaded"] is not True:
        raise SystemExit("CSR policy model artifact manifest load mismatch")
    if manifest.metadata["replay_exact"] is not True:
        raise SystemExit("CSR policy model artifact manifest replay mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR policy model artifact manifest runtime mismatch")


def _verify_csr_policy_model_acceptance(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR policy model acceptance manifest: {stale}")
    rows = read_jsonl(path / "csr_policy_model_acceptance_rows.jsonl")
    summary = json.loads(
        (path / "csr_policy_model_acceptance_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_policy_model_acceptance_schema.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_policy_model_acceptance":
        raise SystemExit("unexpected CSR policy model acceptance artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR policy model acceptance did not pass")
    if summary["schema_version"] != "phase1_csr_policy_model_acceptance_v1":
        raise SystemExit("CSR policy model acceptance schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR policy model acceptance schema/summary mismatch")
    if summary["accepted_for_shadow"] is not True:
        raise SystemExit("CSR policy model acceptance shadow flag mismatch")
    if summary["accepted_for_runtime_promotion"] is not False:
        raise SystemExit("CSR policy model acceptance promoted blocked model")
    if "quality_gate_not_runtime_eligible" not in set(summary["promotion_blockers"]):
        raise SystemExit("CSR policy model acceptance missing promotion blocker")
    if summary["model_loaded"] is not True:
        raise SystemExit("CSR policy model acceptance model load mismatch")
    if summary["prediction_contract_checked"] is not True:
        raise SystemExit("CSR policy model acceptance prediction contract mismatch")
    if summary["num_predictions"] != 20 or summary["num_request_index_rows"] != 20:
        raise SystemExit("CSR policy model acceptance prediction count mismatch")
    if summary["replay_exact"] is not True:
        raise SystemExit("CSR policy model acceptance replay mismatch")
    if summary["quality_gate_runtime_eligible"] is not False:
        raise SystemExit("CSR policy model acceptance quality gate mismatch")
    if summary["guard_plan_shadow_checked"] is not True:
        raise SystemExit("CSR policy model acceptance guard plan mismatch")
    if summary["guarded_gpu_shadow_smoke_checked"] is not True:
        raise SystemExit("CSR policy model acceptance GPU smoke mismatch")
    if summary["guarded_gpu_smoke_status"] != "passed":
        raise SystemExit("CSR policy model acceptance GPU smoke status mismatch")
    if summary["guarded_gpu_smoke_successes"] != 2:
        raise SystemExit("CSR policy model acceptance GPU smoke success mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR policy model acceptance changed runtime selector")
    if summary["validation_error_count"] != 0:
        raise SystemExit("CSR policy model acceptance validation errors found")
    row_kinds = {row["row_kind"] for row in rows}
    expected = {
        "model_artifact_contract",
        "prediction_contract",
        "quality_gate",
        "guarded_gpu_shadow_smoke",
        "acceptance_decision",
        "guard_plan_shadow",
        "guarded_gpu_shadow_result",
    }
    if row_kinds != expected:
        raise SystemExit("CSR policy model acceptance row kind mismatch")
    guard_rows = [row for row in rows if row["row_kind"] == "guard_plan_shadow"]
    if len(guard_rows) != 2:
        raise SystemExit("CSR policy model acceptance guard row count mismatch")
    for row in guard_rows:
        if row["learned_policy_source"]["source_kind"] != "model_artifact":
            raise SystemExit("CSR policy model acceptance guard source mismatch")
        if row["learned_policy_source"]["model_loaded"] is not True:
            raise SystemExit("CSR policy model acceptance guard model load mismatch")
        if row["fallback_chain_enforced"] is not True:
            raise SystemExit("CSR policy model acceptance guard fallback mismatch")
        if row["runtime_selector_changed"] is not False:
            raise SystemExit("CSR policy model acceptance guard runtime mismatch")
    gpu_rows = [row for row in rows if row["row_kind"] == "guarded_gpu_shadow_result"]
    if len(gpu_rows) != 2:
        raise SystemExit("CSR policy model acceptance GPU row count mismatch")
    for row in gpu_rows:
        if row["backend"] != "taichi_gpu":
            raise SystemExit("CSR policy model acceptance GPU backend mismatch")
        if row["learned_prediction_source"] != "model_artifact":
            raise SystemExit("CSR policy model acceptance GPU source mismatch")
    if manifest.metadata["accepted_for_shadow"] is not True:
        raise SystemExit("CSR policy model acceptance manifest shadow mismatch")
    if manifest.metadata["accepted_for_runtime_promotion"] is not False:
        raise SystemExit("CSR policy model acceptance manifest promotion mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR policy model acceptance manifest runtime mismatch")


def _verify_csr_policy_model_submission(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR policy model submission manifest: {stale}")
    submission = json.loads(
        (path / "csr_policy_model_submission_manifest.json").read_text(encoding="utf-8")
    )
    rows = read_jsonl(path / "csr_policy_model_submission_rows.jsonl")
    summary = json.loads(
        (path / "csr_policy_model_submission_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_policy_model_submission_schema.json").read_text(encoding="utf-8")
    )
    model_card = (path / "MODEL_CARD.md").read_text(encoding="utf-8")
    if manifest.artifact_kind != "csr_policy_model_submission":
        raise SystemExit("unexpected CSR policy model submission artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR policy model submission did not pass")
    if summary["schema_version"] != "phase1_csr_policy_model_submission_v1":
        raise SystemExit("CSR policy model submission schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR policy model submission schema/summary mismatch")
    if submission["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR policy model submission manifest schema mismatch")
    if summary["submission_ready"] is not True:
        raise SystemExit("CSR policy model submission not ready")
    if summary["shadow_submission_ready"] is not True:
        raise SystemExit("CSR policy model submission shadow flag mismatch")
    if summary["runtime_promotion_ready"] is not False:
        raise SystemExit("CSR policy model submission promoted blocked model")
    if summary["default_runtime_mode"] != "shadow":
        raise SystemExit("CSR policy model submission default mode mismatch")
    if summary["terms_acknowledged"] is not True:
        raise SystemExit("CSR policy model submission terms mismatch")
    if "MODEL_CONTRIBUTION_TERMS.md" not in summary["required_terms_documents"]:
        raise SystemExit("CSR policy model submission missing model terms document")
    if "CONTRIBUTOR_LICENSE_AGREEMENT.md" not in summary["required_terms_documents"]:
        raise SystemExit("CSR policy model submission missing contributor terms document")
    if summary["accepted_for_shadow"] is not True:
        raise SystemExit("CSR policy model submission shadow acceptance mismatch")
    if summary["accepted_for_runtime_promotion"] is not False:
        raise SystemExit("CSR policy model submission runtime acceptance mismatch")
    if "quality_gate_not_runtime_eligible" not in set(summary["promotion_blockers"]):
        raise SystemExit("CSR policy model submission missing quality blocker")
    if summary["guarded_gpu_shadow_smoke_checked"] is not True:
        raise SystemExit("CSR policy model submission GPU shadow check mismatch")
    if summary["guarded_gpu_smoke_successes"] != 2:
        raise SystemExit("CSR policy model submission GPU success count mismatch")
    if summary["num_packaged_files"] != 8:
        raise SystemExit("CSR policy model submission file count mismatch")
    if summary["all_file_checksums_present"] is not True:
        raise SystemExit("CSR policy model submission checksum mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR policy model submission changed runtime selector")
    if summary["validation_error_count"] != 0:
        raise SystemExit("CSR policy model submission validation errors found")
    if submission["runtime_boundary"]["guard_required"] is not True:
        raise SystemExit("CSR policy model submission guard boundary mismatch")
    if submission["runtime_boundary"]["runtime_selector_changed"] is not False:
        raise SystemExit("CSR policy model submission runtime boundary mismatch")
    if len(submission["files"]) != summary["num_packaged_files"]:
        raise SystemExit("CSR policy model submission file manifest mismatch")
    required_labels = {
        "policy_model_artifact",
        "saved_model",
        "transformer_tensors",
        "request_index",
        "quality_gate_summary",
        "replay_summary",
        "acceptance_summary",
        "acceptance_rows",
    }
    file_labels = {item["label"] for item in submission["files"]}
    if file_labels != required_labels:
        raise SystemExit("CSR policy model submission file label mismatch")
    for item in submission["files"]:
        if item["exists"] is not True or not item["sha256"]:
            raise SystemExit("CSR policy model submission missing file checksum")
        actual = Path(item["path"]).read_bytes()
        digest = hashlib.sha256(actual).hexdigest()
        if digest != item["sha256"]:
            raise SystemExit("CSR policy model submission checksum does not match file")
    row_kinds = {row["row_kind"] for row in rows}
    if row_kinds != {
        "submission_identity",
        "terms",
        "model_artifact",
        "acceptance_gate",
        "submission_decision",
        "file_checksum",
    }:
        raise SystemExit("CSR policy model submission row kind mismatch")
    checksum_rows = [row for row in rows if row["row_kind"] == "file_checksum"]
    if len(checksum_rows) != summary["num_packaged_files"]:
        raise SystemExit("CSR policy model submission checksum row count mismatch")
    if "Runtime Boundary" not in model_card:
        raise SystemExit("CSR policy model submission model card missing boundary")
    if "guarded_gpu_shadow_smoke_checked" not in model_card:
        raise SystemExit("CSR policy model submission model card missing GPU evidence")
    if manifest.metadata["submission_ready"] is not True:
        raise SystemExit("CSR policy model submission manifest ready mismatch")
    if manifest.metadata["shadow_submission_ready"] is not True:
        raise SystemExit("CSR policy model submission manifest shadow mismatch")
    if manifest.metadata["runtime_promotion_ready"] is not False:
        raise SystemExit("CSR policy model submission manifest promotion mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR policy model submission manifest runtime mismatch")


def _verify_csr_external_model_intake(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR external model intake manifest: {stale}")
    rows = read_jsonl(path / "csr_external_model_intake_rows.jsonl")
    summary = json.loads(
        (path / "csr_external_model_intake_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_external_model_intake_schema.json").read_text(encoding="utf-8")
    )
    report = (path / "csr_external_model_intake_report.md").read_text(encoding="utf-8")
    required_files = (
        "external_csr_ranker_checkpoint.json",
        "adapted_csr_transformer_ranker_model.json",
        "adapted_csr_transformer_ranker_predictions.jsonl",
        "csr_external_model_quality_gate_summary.json",
        "csr_policy_model_artifact.json",
        "csr_policy_model_acceptance_summary.json",
        "csr_policy_model_submission_manifest.json",
    )
    if manifest.artifact_kind != "csr_external_model_intake":
        raise SystemExit("unexpected CSR external model intake artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR external model intake did not pass")
    if summary["schema_version"] != "phase1_csr_external_model_intake_v1":
        raise SystemExit("CSR external model intake schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR external model intake schema/summary mismatch")
    if summary["intake_ready"] is not True:
        raise SystemExit("CSR external model intake not ready")
    if summary["shadow_submission_ready"] is not True:
        raise SystemExit("CSR external model intake shadow mismatch")
    if summary["runtime_promotion_ready"] is not False:
        raise SystemExit("CSR external model intake promoted blocked model")
    if summary["default_runtime_mode"] != "shadow":
        raise SystemExit("CSR external model intake default mode mismatch")
    if summary["adapter_ready"] is not True:
        raise SystemExit("CSR external model intake adapter mismatch")
    if summary["replay_exact"] is not True:
        raise SystemExit("CSR external model intake replay mismatch")
    if summary["quality_gate_runtime_eligible"] is not False:
        raise SystemExit("CSR external model intake quality gate mismatch")
    if summary["policy_model_artifact_ready"] is not True:
        raise SystemExit("CSR external model intake policy artifact mismatch")
    if summary["accepted_for_shadow"] is not True:
        raise SystemExit("CSR external model intake acceptance shadow mismatch")
    if summary["accepted_for_runtime_promotion"] is not False:
        raise SystemExit("CSR external model intake acceptance promotion mismatch")
    if "quality_gate_not_runtime_eligible" not in set(summary["promotion_blockers"]):
        raise SystemExit("CSR external model intake missing promotion blocker")
    if summary["guarded_gpu_shadow_smoke_checked"] is not True:
        raise SystemExit("CSR external model intake GPU smoke check mismatch")
    if summary["guarded_gpu_smoke_successes"] != 2:
        raise SystemExit("CSR external model intake GPU smoke count mismatch")
    if summary["num_predictions"] != 20:
        raise SystemExit("CSR external model intake prediction count mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR external model intake changed runtime selector")
    expected_stages = {
        "adapter",
        "replay",
        "quality_gate",
        "policy_model_artifact",
        "learned_guard",
        "acceptance",
        "submission",
    }
    if set(summary["stage_statuses"]) != expected_stages:
        raise SystemExit("CSR external model intake stage set mismatch")
    if any(status != "passed" for status in summary["stage_statuses"].values()):
        raise SystemExit("CSR external model intake stage failure")
    row_kinds = {row["row_kind"] for row in rows}
    if row_kinds != {"stage_status", "intake_decision"}:
        raise SystemExit("CSR external model intake row kind mismatch")
    if len([row for row in rows if row["row_kind"] == "stage_status"]) != len(
        expected_stages
    ):
        raise SystemExit("CSR external model intake stage row count mismatch")
    for name in required_files:
        if not (path / name).exists():
            raise SystemExit(f"CSR external model intake missing file: {name}")
    if "Promotion Blockers" not in report:
        raise SystemExit("CSR external model intake report missing blockers")
    if manifest.metadata["intake_ready"] is not True:
        raise SystemExit("CSR external model intake manifest ready mismatch")
    if manifest.metadata["shadow_submission_ready"] is not True:
        raise SystemExit("CSR external model intake manifest shadow mismatch")
    if manifest.metadata["runtime_promotion_ready"] is not False:
        raise SystemExit("CSR external model intake manifest promotion mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR external model intake manifest runtime mismatch")


def _verify_csr_transformer_training_entrypoint(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR Transformer training entrypoint manifest: {stale}")
    rows = read_jsonl(path / "csr_transformer_training_entrypoint_rows.jsonl")
    summary = json.loads(
        (path / "csr_transformer_training_entrypoint_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_transformer_training_entrypoint_schema.json").read_text(
            encoding="utf-8"
        )
    )
    job_spec = json.loads(
        (path / "csr_transformer_training_job_spec.json").read_text(encoding="utf-8")
    )
    quality_contract = json.loads(
        (path / "csr_transformer_training_quality_contract.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_transformer_training_entrypoint":
        raise SystemExit("unexpected CSR Transformer training entrypoint artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR Transformer training entrypoint did not pass")
    if summary["schema_version"] != "phase1_csr_transformer_training_entrypoint_v1":
        raise SystemExit("CSR Transformer training entrypoint schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer training entrypoint schema/summary mismatch")
    if job_spec["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer training job spec schema mismatch")
    if quality_contract["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer training quality contract schema mismatch")
    if len(rows) != 6:
        raise SystemExit("CSR Transformer training entrypoint row count mismatch")
    if {row["row_kind"] for row in rows} != {
        "input_tensor",
        "input_request_index",
        "input_model_contract",
        "offline_targets",
        "quality_gate",
        "runtime_guard",
    }:
        raise SystemExit("CSR Transformer training entrypoint row-kind mismatch")
    if summary["training_entrypoint_ready"] is not True:
        raise SystemExit("CSR Transformer training entrypoint not marked ready")
    if summary["model_training_required"] is not True:
        raise SystemExit("CSR Transformer training entrypoint should require training")
    if summary["model_trained"] is not False:
        raise SystemExit("CSR Transformer training entrypoint should not train model")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer training entrypoint changed runtime selector")
    if summary["transformer_connectable"] is not True:
        raise SystemExit("CSR Transformer training entrypoint not connectable")
    if summary["quality_gate_enforced"] is not True:
        raise SystemExit("CSR Transformer training entrypoint gate not enforced")
    if summary["current_quality_gate_runtime_eligible"] is not False:
        raise SystemExit("CSR Transformer training entrypoint should preserve current block")
    expected_failures = {
        "below_min_profiled_success_rate",
        "below_baseline_oracle_top1",
        "below_baseline_profiled_success_rate",
        "non_success_eval_selections",
    }
    if set(summary["current_quality_gate_failures"]) != expected_failures:
        raise SystemExit("CSR Transformer training entrypoint gate failures mismatch")
    if summary["num_selector_rows"] != 132:
        raise SystemExit("CSR Transformer training entrypoint selector count mismatch")
    if summary["num_matrices"] != 20:
        raise SystemExit("CSR Transformer training entrypoint matrix count mismatch")
    if summary["num_model_requests"] != 20 or summary["num_tensor_requests"] != 20:
        raise SystemExit("CSR Transformer training entrypoint request count mismatch")
    if summary["num_global_candidates"] != 9:
        raise SystemExit("CSR Transformer training entrypoint candidate count mismatch")
    if summary["matrix_feature_dim"] != 21 or summary["candidate_feature_dim"] != 16:
        raise SystemExit("CSR Transformer training entrypoint feature dims mismatch")
    if summary["validation_error_count"] != 0 or summary["validation_errors"]:
        raise SystemExit("CSR Transformer training entrypoint validation errors found")
    if schema["runtime_integration"]["selector_before_training"] != "artifact_backed":
        raise SystemExit("CSR Transformer training entrypoint runtime boundary mismatch")
    if job_spec["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer training job spec runtime mismatch")
    if "do_not_train_inside_this_entrypoint_artifact" not in set(job_spec["non_goals"]):
        raise SystemExit("CSR Transformer training job spec non-goal mismatch")
    if (
        quality_contract["current_gate_status"]["challenger_runtime_eligible"]
        is not False
    ):
        raise SystemExit("CSR Transformer training quality contract gate mismatch")
    if manifest.metadata["training_entrypoint_ready"] is not True:
        raise SystemExit("CSR Transformer training entrypoint manifest ready mismatch")
    if manifest.metadata["model_trained"] is not False:
        raise SystemExit("CSR Transformer training entrypoint manifest model mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer training entrypoint manifest runtime mismatch")


def _verify_csr_transformer_reference_training_export(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(
            f"stale CSR Transformer reference training export manifest: {stale}"
        )
    rows = read_jsonl(path / "csr_transformer_reference_training_export_rows.jsonl")
    summary = json.loads(
        (path / "csr_transformer_reference_training_export_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_transformer_reference_training_export_schema.json").read_text(
            encoding="utf-8"
        )
    )
    checkpoint = json.loads(
        (path / "reference_csr_external_ranker_checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    adapter_summary = json.loads(
        (path / "reference_csr_external_adapter_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_transformer_reference_training_export":
        raise SystemExit(
            "unexpected CSR Transformer reference training export artifact kind"
        )
    if summary["status"] != "passed":
        raise SystemExit("CSR Transformer reference training export did not pass")
    if (
        summary["schema_version"]
        != "phase1_csr_transformer_reference_training_export_v1"
    ):
        raise SystemExit("CSR Transformer reference training export schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR Transformer reference training export schema mismatch")
    if len(rows) != 6:
        raise SystemExit("CSR Transformer reference training export row count mismatch")
    if {row["row_kind"] for row in rows} != {
        "training_result",
        "external_checkpoint",
        "adapter_roundtrip",
        "intake_boundary",
        "runtime_boundary",
        "export_decision",
    }:
        raise SystemExit("CSR Transformer reference training export row kind mismatch")
    if summary["reference_training_export_ready"] is not True:
        raise SystemExit("CSR Transformer reference training export not ready")
    if summary["model_training_executed"] is not True:
        raise SystemExit("CSR Transformer reference training export did not train")
    if summary["model_trained"] is not True:
        raise SystemExit("CSR Transformer reference training export model mismatch")
    if summary["training_entrypoint_ready"] is not True:
        raise SystemExit("CSR Transformer reference training entrypoint mismatch")
    if summary["external_checkpoint_ready"] is not True:
        raise SystemExit("CSR Transformer reference checkpoint not ready")
    if summary["adapter_ready"] is not True:
        raise SystemExit("CSR Transformer reference adapter not ready")
    if summary["adapter_roundtrip_exact"] is not True:
        raise SystemExit("CSR Transformer reference adapter roundtrip mismatch")
    if summary["intake_input_ready"] is not True:
        raise SystemExit("CSR Transformer reference intake input not ready")
    if summary["default_runtime_mode"] != "shadow":
        raise SystemExit("CSR Transformer reference default mode mismatch")
    if summary["quality_gate_required"] is not True:
        raise SystemExit("CSR Transformer reference quality gate mismatch")
    if summary["current_quality_gate_runtime_eligible"] is not False:
        raise SystemExit("CSR Transformer reference should preserve current gate block")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer reference changed runtime selector")
    if summary["num_predictions"] != 20 or summary["num_eval_predictions"] != 5:
        raise SystemExit("CSR Transformer reference prediction count mismatch")
    if summary["num_eval_oracle_requests"] != 4:
        raise SystemExit("CSR Transformer reference oracle count mismatch")
    if summary["d_model"] != 24 or summary["num_attention_heads"] != 4:
        raise SystemExit("CSR Transformer reference attention shape mismatch")
    if summary["feedforward_dim"] != 48 or summary["num_epochs"] != 160:
        raise SystemExit("CSR Transformer reference training shape mismatch")
    if abs(float(summary["eval_oracle_top1_accuracy"]) - 0.5) > 1.0e-12:
        raise SystemExit("CSR Transformer reference eval accuracy mismatch")
    if (
        abs(float(summary["eval_profiled_success_selection_rate"]) - 0.6)
        > 1.0e-12
    ):
        raise SystemExit("CSR Transformer reference success-rate mismatch")
    if summary["eval_non_success_selection_count"] != 2:
        raise SystemExit("CSR Transformer reference non-success count mismatch")
    if summary["validation_error_count"] != 0 or summary["validation_errors"]:
        raise SystemExit("CSR Transformer reference validation errors found")
    if checkpoint["schema_version"] != "phase1_csr_external_ranker_checkpoint_v1":
        raise SystemExit("CSR Transformer reference checkpoint schema mismatch")
    if checkpoint["runtime_contract"]["default_mode"] != "shadow":
        raise SystemExit("CSR Transformer reference checkpoint default mode mismatch")
    if checkpoint["runtime_contract"]["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer reference checkpoint runtime mismatch")
    if adapter_summary["adapter_ready"] is not True:
        raise SystemExit("CSR Transformer reference adapter summary mismatch")
    if manifest.metadata["reference_training_export_ready"] is not True:
        raise SystemExit("CSR Transformer reference manifest ready mismatch")
    if manifest.metadata["model_trained"] is not True:
        raise SystemExit("CSR Transformer reference manifest model mismatch")
    if manifest.metadata["intake_input_ready"] is not True:
        raise SystemExit("CSR Transformer reference manifest intake mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR Transformer reference manifest runtime mismatch")


def _verify_csr_learned_runtime_guard(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR learned runtime guard artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_learned_guard_rows.jsonl")
    summary = json.loads((path / "csr_learned_guard_summary.json").read_text(encoding="utf-8"))
    schema = json.loads((path / "csr_learned_guard_schema.json").read_text(encoding="utf-8"))
    if manifest.artifact_kind != "csr_learned_runtime_guard":
        raise SystemExit("unexpected CSR learned guard artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR learned guard did not pass")
    if summary["schema_version"] != "phase1_csr_learned_runtime_guard_v1":
        raise SystemExit("CSR learned guard schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR learned guard schema/summary mismatch")
    if schema["default_mode"] != "shadow":
        raise SystemExit("CSR learned guard default mode mismatch")
    if schema["runtime_guard"]["preemptive_gpu_kill_supported"] is not False:
        raise SystemExit("CSR learned guard preemptive kill flag mismatch")
    if summary["num_checks"] != 4 or len(rows) != 4:
        raise SystemExit("CSR learned guard check count mismatch")
    for key in (
        "shadow_mode_checked",
        "quality_gate_blocks_checked",
        "confidence_threshold_checked",
        "saved_model_shadow_checked",
        "saved_model_quality_gate_checked",
        "model_artifact_shadow_checked",
        "model_artifact_quality_gate_checked",
        "fallback_chain_enforced_checked",
        "promotion_fixture_checked",
        "runtime_fallback_checked",
        "timeout_guard_checked",
    ):
        if summary[key] is not True:
            raise SystemExit(f"CSR learned guard missing {key}")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR learned guard artifact must not change runtime selector")
    if int(summary["saved_model_loaded_checks"]) != 2:
        raise SystemExit("CSR learned guard saved-model load count mismatch")
    rows_by_id = {row["check_id"]: row for row in rows}
    if rows_by_id["actual_shadow_mode"]["guard_status"] != "shadow_only":
        raise SystemExit("CSR learned guard shadow mode check mismatch")
    if rows_by_id["actual_shadow_mode"]["learned_policy_source"]["source_kind"] != "model_artifact":
        raise SystemExit("CSR learned guard shadow mode did not use model artifact")
    if rows_by_id["actual_shadow_mode"]["learned_policy_source"]["model_loaded"] is not True:
        raise SystemExit("CSR learned guard shadow mode did not load model")
    if (
        rows_by_id["actual_shadow_mode"]["learned_policy_source"]["adapter"]
        != "csr_transformer_ranker_saved_model_v1"
    ):
        raise SystemExit("CSR learned guard shadow mode adapter mismatch")
    if (
        rows_by_id["actual_quality_gate_blocks_promotion"]["guard_status"]
        != "blocked_quality_gate"
    ):
        raise SystemExit("CSR learned guard quality gate block mismatch")
    if (
        rows_by_id["actual_quality_gate_blocks_promotion"]["learned_policy_source"][
            "source_kind"
        ]
        != "model_artifact"
    ):
        raise SystemExit("CSR learned guard quality gate did not use model artifact")
    if (
        rows_by_id["actual_quality_gate_blocks_promotion"]["learned_policy_source"][
            "model_loaded"
        ]
        is not True
    ):
        raise SystemExit("CSR learned guard quality gate did not load model")
    if (
        rows_by_id["eligible_high_confidence_promotion_fixture"]["guard_status"]
        != "promoted"
    ):
        raise SystemExit("CSR learned guard promotion fixture mismatch")
    if (
        rows_by_id["eligible_high_confidence_promotion_fixture"][
            "fallback_chain_enforced"
        ]
        is not True
    ):
        raise SystemExit("CSR learned guard fallback enforcement mismatch")
    runtime_row = rows_by_id["runtime_timeout_fallback_fixture"]
    if runtime_row["result_status"] != "fallback_success":
        raise SystemExit("CSR learned guard runtime fallback result mismatch")
    if runtime_row["used_fallback"] is not True:
        raise SystemExit("CSR learned guard runtime fallback not used")
    if runtime_row["first_attempt_timed_out"] is not True:
        raise SystemExit("CSR learned guard timeout check mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR learned guard manifest runtime flag mismatch")
    if int(manifest.metadata["saved_model_loaded_checks"]) != 2:
        raise SystemExit("CSR learned guard manifest saved-model count mismatch")


def _verify_csr_guarded_auto_solve(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR guarded auto-solve artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_guarded_auto_solve_results.jsonl")
    summary = json.loads(
        (path / "csr_guarded_auto_solve_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_guarded_auto_solve_schema.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_guarded_auto_solve_smoke":
        raise SystemExit("unexpected CSR guarded auto-solve artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR guarded auto-solve did not pass")
    if summary["schema_version"] != "phase1_csr_guarded_auto_solve_v1":
        raise SystemExit("CSR guarded auto-solve schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR guarded auto-solve schema/summary mismatch")
    if schema["runtime"]["backend"] != "taichi_gpu":
        raise SystemExit("CSR guarded auto-solve backend mismatch")
    if schema["runtime"]["learned_policy_mode"] != "promote_if_safe":
        raise SystemExit("CSR guarded auto-solve learned mode mismatch")
    if len(rows) != 2 or summary["num_solves"] != 2:
        raise SystemExit("CSR guarded auto-solve row count mismatch")
    if summary["num_success"] != 2 or summary["num_failed"] != 0:
        raise SystemExit("CSR guarded auto-solve success count mismatch")
    if summary["quality_gate_blocks"] != 2:
        raise SystemExit("CSR guarded auto-solve quality-gate block count mismatch")
    if summary["learned_prediction_source"] != "model_artifact":
        raise SystemExit("CSR guarded auto-solve did not use model artifact")
    if int(summary["saved_model_loaded_count"]) != 2:
        raise SystemExit("CSR guarded auto-solve saved-model load count mismatch")
    if summary["learned_runtime_promotions"] != 0:
        raise SystemExit("CSR guarded auto-solve should not promote learned runtime")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR guarded auto-solve runtime selector changed unexpectedly")
    if summary["fallback_chain_enforced_count"] != 2:
        raise SystemExit("CSR guarded auto-solve fallback enforcement mismatch")
    if summary["runtime_guard_success_count"] != 2:
        raise SystemExit("CSR guarded auto-solve runtime guard success mismatch")
    if summary["runtime_fallback_used_count"] != 0:
        raise SystemExit("CSR guarded auto-solve unexpected fallback use")
    if float(summary["max_final_relative_residual"]) > 1.0e-5:
        raise SystemExit("CSR guarded auto-solve trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("CSR guarded auto-solve CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("CSR guarded auto-solve solution error check failed")
    expected_matrices = {"suitesparse:FIDAP/ex5", "suitesparse:HB/curtis54"}
    if {row["matrix_id"] for row in rows} != expected_matrices:
        raise SystemExit("CSR guarded auto-solve matrix set mismatch")
    for row in rows:
        if row["status"] != "success":
            raise SystemExit("CSR guarded auto-solve has failed row")
        if row["runtime_selection_source"] != "artifact":
            raise SystemExit("CSR guarded auto-solve selected learned runtime unexpectedly")
        if row["learned_policy_source_kind"] != "model_artifact":
            raise SystemExit("CSR guarded auto-solve row did not use model artifact")
        if row["learned_model_loaded"] is not True:
            raise SystemExit("CSR guarded auto-solve row did not load model")
        if row["learned_policy_source"]["adapter"] != "csr_transformer_ranker_saved_model_v1":
            raise SystemExit("CSR guarded auto-solve row adapter mismatch")
        if row["learned_guard_status"] != "blocked_quality_gate":
            raise SystemExit("CSR guarded auto-solve learned guard status mismatch")
        if "quality_gate:non_success_eval_selections" not in row["learned_guard_reasons"]:
            raise SystemExit("CSR guarded auto-solve missing quality gate reason")
        if row["fallback_chain_enforced"] is not True:
            raise SystemExit("CSR guarded auto-solve fallback chain not enforced")
        if row["runtime_guard_status"] != "success":
            raise SystemExit("CSR guarded auto-solve runtime guard status mismatch")
        if row["runtime_guard_used_fallback"] is not False:
            raise SystemExit("CSR guarded auto-solve unexpected fallback row")
        if int(row["runtime_guard_attempt_count"]) != 1:
            raise SystemExit("CSR guarded auto-solve attempt count mismatch")
        if row["trace"]["backend"] != "taichi_gpu":
            raise SystemExit("CSR guarded auto-solve did not use taichi_gpu")
        metadata = row["trace"]["metadata"]
        if "learned_policy_guard" not in metadata or "runtime_guard" not in metadata:
            raise SystemExit("CSR guarded auto-solve trace missing guard metadata")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR guarded auto-solve manifest runtime flag mismatch")
    if manifest.metadata["learned_prediction_source"] != "model_artifact":
        raise SystemExit("CSR guarded auto-solve manifest source mismatch")
    if int(manifest.metadata["saved_model_loaded_count"]) != 2:
        raise SystemExit("CSR guarded auto-solve manifest saved-model count mismatch")
    if manifest.metadata["quality_gate_blocks"] != 2:
        raise SystemExit("CSR guarded auto-solve manifest quality block mismatch")


def _verify_csr_guarded_promotion_readiness(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR guarded promotion readiness artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_guarded_promotion_readiness_results.jsonl")
    summary = json.loads(
        (path / "csr_guarded_promotion_readiness_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_guarded_promotion_readiness_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_guarded_promotion_readiness":
        raise SystemExit("unexpected CSR guarded promotion readiness artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR guarded promotion readiness did not pass")
    if summary["schema_version"] != "phase1_csr_guarded_promotion_readiness_v1":
        raise SystemExit("CSR guarded promotion readiness schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR guarded promotion readiness schema/summary mismatch")
    if schema["runtime"]["backend"] != "taichi_gpu":
        raise SystemExit("CSR guarded promotion readiness backend mismatch")
    if schema["runtime"]["production_runtime_selector_changed"] is not False:
        raise SystemExit("CSR guarded promotion readiness production runtime flag mismatch")
    if len(rows) != 3 or summary["num_scenarios"] != 3:
        raise SystemExit("CSR guarded promotion readiness row count mismatch")
    if summary["num_success"] != 3 or summary["num_failed"] != 0:
        raise SystemExit("CSR guarded promotion readiness success count mismatch")
    if summary["current_quality_gate_blocks"] != 1:
        raise SystemExit("CSR guarded promotion readiness quality block mismatch")
    if summary["fixture_learned_promotions"] != 1:
        raise SystemExit("CSR guarded promotion readiness promotion count mismatch")
    if summary["fixture_runtime_selector_changed_count"] != 1:
        raise SystemExit("CSR guarded promotion readiness fixture runtime flag mismatch")
    if summary["current_runtime_selector_changed"] is not False:
        raise SystemExit("CSR guarded promotion readiness current selector changed")
    if summary["runtime_exception_fallbacks"] != 1:
        raise SystemExit("CSR guarded promotion readiness fallback count mismatch")
    if summary["runtime_fallback_used_count"] != 1:
        raise SystemExit("CSR guarded promotion readiness fallback-use count mismatch")
    if summary["runtime_guard_success_count"] != 3:
        raise SystemExit("CSR guarded promotion readiness guard success mismatch")
    if summary["real_gpu_final_result_count"] != 3:
        raise SystemExit("CSR guarded promotion readiness final GPU result count mismatch")
    if set(summary["selected_solver_set"]) != {"bicgstab", "pcg"}:
        raise SystemExit("CSR guarded promotion readiness solver set mismatch")
    if float(summary["max_final_relative_residual"]) > 1.0e-5:
        raise SystemExit("CSR guarded promotion readiness trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("CSR guarded promotion readiness CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("CSR guarded promotion readiness solution error check failed")
    rows_by_kind = {row["scenario_kind"]: row for row in rows}
    if set(rows_by_kind) != {
        "current_ranker_blocked",
        "fixture_learned_promotion",
        "runtime_exception_fallback",
    }:
        raise SystemExit("CSR guarded promotion readiness scenario set mismatch")
    blocked = rows_by_kind["current_ranker_blocked"]
    if blocked["learned_guard_status"] != "blocked_quality_gate":
        raise SystemExit("CSR guarded promotion readiness blocked scenario mismatch")
    if blocked["runtime_selection_source"] != "artifact":
        raise SystemExit("CSR guarded promotion readiness blocked source mismatch")
    if blocked["learned_runtime_selector_changed"] is not False:
        raise SystemExit("CSR guarded promotion readiness blocked runtime flag mismatch")
    promoted = rows_by_kind["fixture_learned_promotion"]
    if promoted["learned_guard_status"] != "promoted":
        raise SystemExit("CSR guarded promotion readiness promotion status mismatch")
    if promoted["runtime_selection_source"] != "learned":
        raise SystemExit("CSR guarded promotion readiness promotion source mismatch")
    if promoted["candidate_id"] != "taichi_csr_pcg_jacobi_float64":
        raise SystemExit("CSR guarded promotion readiness promoted candidate mismatch")
    if promoted["learned_runtime_selector_changed"] is not True:
        raise SystemExit("CSR guarded promotion readiness promotion runtime flag mismatch")
    fallback = rows_by_kind["runtime_exception_fallback"]
    if fallback["runtime_guard_used_fallback"] is not True:
        raise SystemExit("CSR guarded promotion readiness fallback was not used")
    if fallback["final_result_status"] != "fallback_success":
        raise SystemExit("CSR guarded promotion readiness fallback result mismatch")
    if int(fallback["runtime_guard_attempt_count"]) != 2:
        raise SystemExit("CSR guarded promotion readiness fallback attempt count mismatch")
    first_attempt = fallback["runtime_guard_attempts"][0]
    if first_attempt["status"] != "exception" or first_attempt["exception_type"] != "ValueError":
        raise SystemExit("CSR guarded promotion readiness first attempt mismatch")
    for row in rows:
        if row["status"] != "success":
            raise SystemExit("CSR guarded promotion readiness failed row")
        if row["trace"]["backend"] != "taichi_gpu":
            raise SystemExit("CSR guarded promotion readiness final result not GPU")
        if row["trace"]["metadata"]["operator_backend"] != "taichi_csr":
            raise SystemExit("CSR guarded promotion readiness final result not CSR GPU")
        if row["failure_reasons"]:
            raise SystemExit("CSR guarded promotion readiness row recorded failure reasons")
    if manifest.metadata["current_runtime_selector_changed"] is not False:
        raise SystemExit("CSR guarded promotion readiness manifest runtime flag mismatch")
    if manifest.metadata["fixture_learned_promotions"] != 1:
        raise SystemExit("CSR guarded promotion readiness manifest promotion count mismatch")


def _verify_csr_guarded_promotion_coverage(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR guarded promotion coverage artifact manifest: {stale}")
    rows = read_jsonl(path / "csr_guarded_promotion_coverage_scenarios.jsonl")
    summary = json.loads(
        (path / "csr_guarded_promotion_coverage_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_guarded_promotion_coverage_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_guarded_promotion_coverage_plan":
        raise SystemExit("unexpected CSR guarded promotion coverage artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR guarded promotion coverage plan did not pass")
    if summary["schema_version"] != "phase1_csr_guarded_promotion_coverage_plan_v1":
        raise SystemExit("CSR guarded promotion coverage schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR guarded promotion coverage schema/summary mismatch")
    if schema["integration_boundary"]["status"] != "plan_only_no_gpu_execution":
        raise SystemExit("CSR guarded promotion coverage should be plan-only")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR guarded promotion coverage changed runtime selector")
    if summary["executes_gpu"] is not False:
        raise SystemExit("CSR guarded promotion coverage should not execute GPU")
    if len(rows) != 15 or summary["planned_scenarios"] != 15:
        raise SystemExit("CSR guarded promotion coverage row count mismatch")
    if summary["planned_gpu_final_solves"] != 11:
        raise SystemExit("CSR guarded promotion coverage planned GPU count mismatch")
    if summary["planned_guard_only_scenarios"] != 4:
        raise SystemExit("CSR guarded promotion coverage guard-only count mismatch")
    if summary["current_blocked_scenarios"] != 4:
        raise SystemExit("CSR guarded promotion coverage blocked count mismatch")
    if summary["fixture_promotion_scenarios"] != 4:
        raise SystemExit("CSR guarded promotion coverage promotion count mismatch")
    if summary["fixture_promotions_changing_candidate"] != 3:
        raise SystemExit("CSR guarded promotion coverage changed-candidate count mismatch")
    if summary["non_success_block_scenarios"] != 4:
        raise SystemExit("CSR guarded promotion coverage non-success count mismatch")
    if summary["runtime_fallback_scenarios"] != 3:
        raise SystemExit("CSR guarded promotion coverage fallback count mismatch")
    if summary["quality_gate_runtime_eligible"] is not False:
        raise SystemExit("CSR guarded promotion coverage expected current gate blocked")
    expected_gate_failures = {
        "below_min_profiled_success_rate",
        "below_baseline_oracle_top1",
        "below_baseline_profiled_success_rate",
        "non_success_eval_selections",
    }
    if set(summary["quality_gate_failures"]) != expected_gate_failures:
        raise SystemExit("CSR guarded promotion coverage gate failure mismatch")
    if summary["num_predictions"] != 20:
        raise SystemExit("CSR guarded promotion coverage prediction count mismatch")
    if summary["profiled_success_predictions"] != 7:
        raise SystemExit("CSR guarded promotion coverage profiled prediction mismatch")
    if summary["non_success_predictions"] != 13:
        raise SystemExit("CSR guarded promotion coverage non-success prediction mismatch")
    if summary["by_scenario_kind"] != {
        "current_ranker_quality_gate_block": 4,
        "fixture_non_success_candidate_block": 4,
        "fixture_profiled_success_promotion": 4,
        "runtime_exception_fallback": 3,
    }:
        raise SystemExit("CSR guarded promotion coverage scenario-kind counts mismatch")
    if summary["by_selected_target_status"] != {
        "not_applicable": 1,
        "not_profiled": 1,
        "screened_out": 2,
        "success": 8,
    }:
        raise SystemExit("CSR guarded promotion coverage target-status counts mismatch")
    if summary["by_runtime_solver"] != {
        "bicgstab": 3,
        "chebyshev": 3,
        "gmres": 4,
        "pcg": 1,
    }:
        raise SystemExit("CSR guarded promotion coverage runtime solver counts mismatch")
    if manifest.metadata["planned_scenarios"] != 15:
        raise SystemExit("CSR guarded promotion coverage manifest scenario count mismatch")
    if manifest.metadata["planned_gpu_final_solves"] != 11:
        raise SystemExit("CSR guarded promotion coverage manifest GPU count mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR guarded promotion coverage manifest runtime flag mismatch")
    rows_by_kind: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_kind.setdefault(row["scenario_kind"], []).append(row)
        if row["schema_version"] != summary["schema_version"]:
            raise SystemExit("CSR guarded promotion coverage row schema mismatch")
        if not row["matrix_source_path"]:
            raise SystemExit("CSR guarded promotion coverage missing matrix source")
        if row["requires_gpu_execution"] and not row["artifact_candidate_id"]:
            raise SystemExit("CSR guarded promotion coverage GPU row lacks artifact candidate")
    for row in rows_by_kind["current_ranker_quality_gate_block"]:
        if row["expected_guard_status"] != "blocked_quality_gate":
            raise SystemExit("CSR guarded promotion coverage blocked status mismatch")
        if row["expected_runtime_selection_source"] != "artifact":
            raise SystemExit("CSR guarded promotion coverage blocked source mismatch")
        if row["requires_gpu_execution"] is not True:
            raise SystemExit("CSR guarded promotion coverage blocked GPU flag mismatch")
    for row in rows_by_kind["fixture_profiled_success_promotion"]:
        if row["expected_guard_status"] != "promoted":
            raise SystemExit("CSR guarded promotion coverage promotion status mismatch")
        if row["expected_runtime_selection_source"] != "learned":
            raise SystemExit("CSR guarded promotion coverage promotion source mismatch")
        if row["selected_candidate_is_exact_profiled_success"] is not True:
            raise SystemExit("CSR guarded promotion coverage promotion not exact success")
    for row in rows_by_kind["fixture_non_success_candidate_block"]:
        if row["expected_guard_status"] != "blocked_non_success_candidate":
            raise SystemExit("CSR guarded promotion coverage non-success status mismatch")
        if row["requires_gpu_execution"] is not False:
            raise SystemExit("CSR guarded promotion coverage non-success GPU flag mismatch")
    for row in rows_by_kind["runtime_exception_fallback"]:
        if row["requires_runtime_exception_injection"] is not True:
            raise SystemExit("CSR guarded promotion coverage fallback injection flag mismatch")
        if row["expected_final_result_status"] != "fallback_success":
            raise SystemExit("CSR guarded promotion coverage fallback result mismatch")


def _verify_csr_guarded_promotion_coverage_exec(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR guarded promotion coverage exec manifest: {stale}")
    rows = read_jsonl(path / "csr_guarded_promotion_coverage_exec_results.jsonl")
    summary = json.loads(
        (path / "csr_guarded_promotion_coverage_exec_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_guarded_promotion_coverage_exec_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_guarded_promotion_coverage_exec":
        raise SystemExit("unexpected CSR guarded promotion coverage exec artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR guarded promotion coverage exec did not pass")
    if summary["schema_version"] != "phase1_csr_guarded_promotion_coverage_exec_v1":
        raise SystemExit("CSR guarded promotion coverage exec schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR guarded promotion coverage exec schema/summary mismatch")
    if schema["source_plan_schema"] != "phase1_csr_guarded_promotion_coverage_plan_v1":
        raise SystemExit("CSR guarded promotion coverage exec source schema mismatch")
    if len(rows) != 12 or summary["num_scenarios"] != 12:
        raise SystemExit("CSR guarded promotion coverage exec row count mismatch")
    if summary["num_success"] != 12 or summary["num_failed"] != 0:
        raise SystemExit("CSR guarded promotion coverage exec success count mismatch")
    if summary["executed_gpu_scenarios"] != 8:
        raise SystemExit("CSR guarded promotion coverage exec GPU count mismatch")
    if summary["guard_only_scenarios"] != 4:
        raise SystemExit("CSR guarded promotion coverage exec guard-only count mismatch")
    if summary["current_quality_gate_blocks"] != 4:
        raise SystemExit("CSR guarded promotion coverage exec current block mismatch")
    if summary["fixture_learned_promotions"] != 2:
        raise SystemExit("CSR guarded promotion coverage exec promotion count mismatch")
    if summary["fixture_runtime_selector_changed_count"] != 2:
        raise SystemExit("CSR guarded promotion coverage exec fixture runtime flag mismatch")
    if summary["current_runtime_selector_changed"] is not False:
        raise SystemExit("CSR guarded promotion coverage exec current selector changed")
    if summary["non_success_blocks"] != 4:
        raise SystemExit("CSR guarded promotion coverage exec non-success count mismatch")
    if summary["runtime_exception_fallbacks"] != 2:
        raise SystemExit("CSR guarded promotion coverage exec fallback count mismatch")
    if summary["runtime_fallback_used_count"] != 2:
        raise SystemExit("CSR guarded promotion coverage exec fallback-use mismatch")
    if summary["runtime_guard_success_count"] != 8:
        raise SystemExit("CSR guarded promotion coverage exec runtime guard success mismatch")
    if set(summary["selected_solver_set"]) != {"bicgstab", "chebyshev", "gmres", "pcg"}:
        raise SystemExit("CSR guarded promotion coverage exec solver set mismatch")
    if float(summary["max_final_relative_residual"]) > 1.0e-5:
        raise SystemExit("CSR guarded promotion coverage exec trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("CSR guarded promotion coverage exec CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("CSR guarded promotion coverage exec solution error check failed")
    rows_by_kind: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_kind.setdefault(row["scenario_kind"], []).append(row)
        if row["status"] != "success":
            raise SystemExit("CSR guarded promotion coverage exec has failed row")
        if row["failure_reasons"]:
            raise SystemExit("CSR guarded promotion coverage exec recorded failure reasons")
        if row["requires_gpu_execution"]:
            if row["trace"]["backend"] != "taichi_gpu":
                raise SystemExit("CSR guarded promotion coverage exec final result not GPU")
            if row["trace"]["metadata"]["operator_backend"] != "taichi_csr":
                raise SystemExit("CSR guarded promotion coverage exec final result not CSR")
        else:
            if row["trace"] is not None:
                raise SystemExit("CSR guarded promotion coverage guard-only row executed GPU")
    if {key: len(value) for key, value in rows_by_kind.items()} != {
        "current_ranker_quality_gate_block": 4,
        "fixture_non_success_candidate_block": 4,
        "fixture_profiled_success_promotion": 2,
        "runtime_exception_fallback": 2,
    }:
        raise SystemExit("CSR guarded promotion coverage exec scenario counts mismatch")
    for row in rows_by_kind["current_ranker_quality_gate_block"]:
        if row["learned_guard_status"] != "blocked_quality_gate":
            raise SystemExit("CSR guarded promotion coverage exec current block mismatch")
        if row["runtime_selection_source"] != "artifact":
            raise SystemExit("CSR guarded promotion coverage exec current source mismatch")
        if row["learned_runtime_selector_changed"] is not False:
            raise SystemExit("CSR guarded promotion coverage exec current runtime flag mismatch")
    for row in rows_by_kind["fixture_profiled_success_promotion"]:
        if row["learned_guard_status"] != "promoted":
            raise SystemExit("CSR guarded promotion coverage exec promotion mismatch")
        if row["runtime_selection_source"] != "learned":
            raise SystemExit("CSR guarded promotion coverage exec promotion source mismatch")
        if row["learned_runtime_selector_changed"] is not True:
            raise SystemExit("CSR guarded promotion coverage exec promotion runtime flag mismatch")
    for row in rows_by_kind["fixture_non_success_candidate_block"]:
        if row["learned_guard_status"] not in {
            "blocked_non_success_candidate",
            "blocked_fallback_chain",
        }:
            raise SystemExit("CSR guarded promotion coverage exec non-success mismatch")
        if not any("not a profiled success" in reason for reason in row["learned_guard_reasons"]):
            raise SystemExit("CSR guarded promotion coverage exec non-success reason mismatch")
        if row["final_result_status"] != "guard_plan_only":
            raise SystemExit("CSR guarded promotion coverage exec non-success result mismatch")
        if row["runtime_guard_attempt_count"] != 0:
            raise SystemExit("CSR guarded promotion coverage exec non-success attempted runtime")
    for row in rows_by_kind["runtime_exception_fallback"]:
        if row["final_result_status"] != "fallback_success":
            raise SystemExit("CSR guarded promotion coverage exec fallback result mismatch")
        if row["runtime_guard_used_fallback"] is not True:
            raise SystemExit("CSR guarded promotion coverage exec fallback not used")
        if row["runtime_guard_attempt_count"] != 2:
            raise SystemExit("CSR guarded promotion coverage exec fallback attempt mismatch")
        if row["runtime_guard_attempts"][0]["status"] != "exception":
            raise SystemExit("CSR guarded promotion coverage exec first attempt mismatch")
    if manifest.metadata["executed_gpu_scenarios"] != 8:
        raise SystemExit("CSR guarded promotion coverage exec manifest GPU count mismatch")
    if manifest.metadata["current_runtime_selector_changed"] is not False:
        raise SystemExit("CSR guarded promotion coverage exec manifest runtime flag mismatch")


def _verify_csr_non_success_fallback_probe(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR non-success fallback probe manifest: {stale}")
    rows = read_jsonl(path / "csr_non_success_fallback_probe_results.jsonl")
    selector_rows = read_jsonl(path / "csr_non_success_fallback_selector_rows.jsonl")
    summary = json.loads(
        (path / "csr_non_success_fallback_probe_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_non_success_fallback_probe_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_non_success_fallback_probe":
        raise SystemExit("unexpected CSR non-success fallback probe artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR non-success fallback probe did not pass")
    if summary["schema_version"] != "phase1_csr_non_success_fallback_probe_v1":
        raise SystemExit("CSR non-success fallback probe schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR non-success fallback probe schema/summary mismatch")
    if schema["source_plan_schema"] != "phase1_csr_guarded_promotion_coverage_plan_v1":
        raise SystemExit("CSR non-success fallback probe source schema mismatch")
    if schema["runtime_selector_changed"] is not False:
        raise SystemExit("CSR non-success fallback probe changed runtime selector")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR non-success fallback probe summary changed selector")
    if summary["executes_gpu"] is not True:
        raise SystemExit("CSR non-success fallback probe should execute bounded GPU rows")
    if len(rows) != 15 or summary["candidate_attempts"] != 15:
        raise SystemExit("CSR non-success fallback probe row count mismatch")
    if len(selector_rows) != summary["selector_rows"]:
        raise SystemExit("CSR non-success fallback probe selector row count mismatch")
    if summary["target_scenarios"] != 4:
        raise SystemExit("CSR non-success fallback probe scenario count mismatch")
    if summary["gpu_success_rows"] != 4:
        raise SystemExit("CSR non-success fallback probe success count mismatch")
    if summary["cpu_screened_out_rows"] != 11:
        raise SystemExit("CSR non-success fallback probe screen-out count mismatch")
    if summary["gpu_failed_rows"] != 0:
        raise SystemExit("CSR non-success fallback probe contains GPU failures")
    if summary["selector_success_rows"] != summary["gpu_success_rows"]:
        raise SystemExit("CSR non-success fallback probe selector success mismatch")
    if summary["resolved_matrix_count"] != 2:
        raise SystemExit("CSR non-success fallback probe resolved count mismatch")
    if set(summary["resolved_matrices"]) != {
        "suitesparse:Bai/cdde1",
        "suitesparse:FIDAP/ex5",
    }:
        raise SystemExit("CSR non-success fallback probe resolved matrix mismatch")
    if set(summary["unresolved_matrices"]) != {
        "suitesparse:Gset/G17",
        "suitesparse:Zitney/extr1b",
    }:
        raise SystemExit("CSR non-success fallback probe unresolved matrix mismatch")
    if summary["unresolved_matrix_count"] != 2:
        raise SystemExit("CSR non-success fallback probe unresolved count mismatch")
    if summary["by_status"] != {"screened_out": 11, "success": 4}:
        raise SystemExit("CSR non-success fallback probe status split mismatch")
    if float(summary["max_final_relative_residual"]) > 1.0e-5:
        raise SystemExit("CSR non-success fallback probe trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("CSR non-success fallback probe CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("CSR non-success fallback probe solution error check failed")
    success_rows = [row for row in rows if row["status"] == "success"]
    screen_rows = [row for row in rows if row["status"] == "screened_out"]
    if len(success_rows) != 4 or len(screen_rows) != 11:
        raise SystemExit("CSR non-success fallback probe row status split mismatch")
    if {row["matrix_id"] for row in success_rows} != {
        "suitesparse:Bai/cdde1",
        "suitesparse:FIDAP/ex5",
    }:
        raise SystemExit("CSR non-success fallback probe success matrix mismatch")
    if {row["backend"] for row in success_rows} != {"taichi_gpu"}:
        raise SystemExit("CSR non-success fallback probe success rows not GPU")
    if {row["backend"] for row in screen_rows} != {"cpu_reference_screen"}:
        raise SystemExit("CSR non-success fallback probe screen rows not CPU")
    if any(row["gpu_executed"] is not True for row in success_rows):
        raise SystemExit("CSR non-success fallback probe success gpu flag mismatch")
    if any(row["gpu_executed"] is not False for row in screen_rows):
        raise SystemExit("CSR non-success fallback probe screen gpu flag mismatch")
    if any(row["cpu_screen_success"] is not True for row in success_rows):
        raise SystemExit("CSR non-success fallback probe success CPU screen mismatch")
    if any(row["cpu_screen_success"] is not False for row in screen_rows):
        raise SystemExit("CSR non-success fallback probe screen CPU flag mismatch")
    if {row["candidate_role"] for row in rows} != {
        "fallback_probe",
        "learned_non_success_candidate",
    }:
        raise SystemExit("CSR non-success fallback probe candidate role mismatch")
    if any(row["failure_reasons"] for row in success_rows):
        raise SystemExit("CSR non-success fallback probe success row has failure")
    if not all(row["failure_reasons"] for row in screen_rows):
        raise SystemExit("CSR non-success fallback probe screen row lacks reason")
    if {
        row["target_status"] for row in selector_rows
    } != {"screened_out", "success"}:
        raise SystemExit("CSR non-success fallback selector status mismatch")
    if sum(1 for row in selector_rows if row["label_is_oracle"]) != 2:
        raise SystemExit("CSR non-success fallback selector oracle count mismatch")
    if manifest.metadata["gpu_success_rows"] != 4:
        raise SystemExit("CSR non-success fallback probe manifest success mismatch")
    if manifest.metadata["cpu_screened_out_rows"] != 11:
        raise SystemExit("CSR non-success fallback probe manifest screen mismatch")
    if manifest.metadata["resolved_matrix_count"] != 2:
        raise SystemExit("CSR non-success fallback probe manifest resolved mismatch")
    if manifest.metadata["unresolved_matrix_count"] != 2:
        raise SystemExit("CSR non-success fallback probe manifest unresolved mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR non-success fallback probe manifest runtime flag mismatch")


def _verify_csr_guarded_non_success_fallback_integration(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(
            f"stale CSR guarded non-success fallback integration manifest: {stale}"
        )
    rows = read_jsonl(path / "csr_guarded_non_success_fallback_integration_results.jsonl")
    augmented_rows = read_jsonl(path / "augmented_csr_selector_rows.jsonl")
    predictions = read_jsonl(path / "fixture_non_success_predictions.jsonl")
    summary = json.loads(
        (path / "csr_guarded_non_success_fallback_integration_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_guarded_non_success_fallback_integration_schema.json").read_text(
            encoding="utf-8"
        )
    )
    quality_gate = json.loads(
        (path / "fixture_runtime_eligible_quality_gate_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_guarded_non_success_fallback_integration":
        raise SystemExit(
            "unexpected CSR guarded non-success fallback integration artifact kind"
        )
    if summary["status"] != "passed":
        raise SystemExit("CSR guarded non-success fallback integration did not pass")
    if (
        summary["schema_version"]
        != "phase1_csr_guarded_non_success_fallback_integration_v1"
    ):
        raise SystemExit("CSR guarded non-success fallback integration schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit(
            "CSR guarded non-success fallback integration schema/summary mismatch"
        )
    if schema["source_probe_schema"] != "phase1_csr_non_success_fallback_probe_v1":
        raise SystemExit(
            "CSR guarded non-success fallback integration source schema mismatch"
        )
    if schema["safety_policy"]["runtime_selector_changed"] is not False:
        raise SystemExit(
            "CSR guarded non-success fallback integration changed selector in schema"
        )
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit(
            "CSR guarded non-success fallback integration changed runtime selector"
        )
    if len(rows) != 4 or summary["target_non_success_scenarios"] != 4:
        raise SystemExit(
            "CSR guarded non-success fallback integration row count mismatch"
        )
    if len(predictions) != 4:
        raise SystemExit(
            "CSR guarded non-success fallback integration prediction count mismatch"
        )
    if quality_gate["challenger_runtime_eligible"] is not True:
        raise SystemExit(
            "CSR guarded non-success fallback integration fixture gate mismatch"
        )
    if len(augmented_rows) != summary["augmented_selector_rows"]:
        raise SystemExit(
            "CSR guarded non-success fallback integration selector count mismatch"
        )
    if summary["base_selector_rows"] != 132:
        raise SystemExit(
            "CSR guarded non-success fallback integration base selector count mismatch"
        )
    if summary["fallback_probe_selector_rows"] != 15:
        raise SystemExit(
            "CSR guarded non-success fallback integration probe selector count mismatch"
        )
    if summary["fallback_probe_success_rows_imported"] != 4:
        raise SystemExit(
            "CSR guarded non-success fallback integration imported success mismatch"
        )
    if summary["augmented_selector_rows"] != 136:
        raise SystemExit(
            "CSR guarded non-success fallback integration augmented selector mismatch"
        )
    if summary["resolved_exact_fallback_scenarios"] != 2:
        raise SystemExit(
            "CSR guarded non-success fallback integration resolved count mismatch"
        )
    if set(summary["resolved_matrices"]) != {
        "suitesparse:Bai/cdde1",
        "suitesparse:FIDAP/ex5",
    }:
        raise SystemExit(
            "CSR guarded non-success fallback integration resolved matrix mismatch"
        )
    if summary["unresolved_exact_fallback_scenarios"] != 2:
        raise SystemExit(
            "CSR guarded non-success fallback integration unresolved count mismatch"
        )
    if set(summary["unresolved_matrices"]) != {
        "suitesparse:Gset/G17",
        "suitesparse:Zitney/extr1b",
    }:
        raise SystemExit(
            "CSR guarded non-success fallback integration unresolved matrix mismatch"
        )
    if summary["executed_gpu_scenarios"] != 2:
        raise SystemExit(
            "CSR guarded non-success fallback integration GPU count mismatch"
        )
    if summary["guard_only_scenarios"] != 2:
        raise SystemExit(
            "CSR guarded non-success fallback integration guard-only count mismatch"
        )
    if summary["learned_non_success_blocks"] != 4:
        raise SystemExit(
            "CSR guarded non-success fallback integration learned block mismatch"
        )
    if summary["artifact_fallback_solves"] != 2:
        raise SystemExit(
            "CSR guarded non-success fallback integration artifact solve mismatch"
        )
    if summary["runtime_guard_success_count"] != 2:
        raise SystemExit(
            "CSR guarded non-success fallback integration runtime guard mismatch"
        )
    if set(summary["selected_solver_set"]) != {"bicgstab", "pcg"}:
        raise SystemExit(
            "CSR guarded non-success fallback integration solver set mismatch"
        )
    if float(summary["max_final_relative_residual"]) > 1.0e-5:
        raise SystemExit(
            "CSR guarded non-success fallback integration trace residual check failed"
        )
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit(
            "CSR guarded non-success fallback integration CPU residual check failed"
        )
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit(
            "CSR guarded non-success fallback integration solution check failed"
        )
    gpu_rows = [row for row in rows if row["executed_gpu"]]
    guard_rows = [row for row in rows if not row["executed_gpu"]]
    if len(gpu_rows) != 2 or len(guard_rows) != 2:
        raise SystemExit(
            "CSR guarded non-success fallback integration row split mismatch"
        )
    if {row["matrix_id"] for row in gpu_rows} != {
        "suitesparse:Bai/cdde1",
        "suitesparse:FIDAP/ex5",
    }:
        raise SystemExit(
            "CSR guarded non-success fallback integration GPU matrix mismatch"
        )
    for gpu_row in gpu_rows:
        if gpu_row["runtime_selection_source"] != "artifact":
            raise SystemExit(
                "CSR guarded non-success fallback integration GPU source mismatch"
            )
        if gpu_row["learned_guard_status"] != "blocked_non_success_candidate":
            raise SystemExit(
                "CSR guarded non-success fallback integration GPU guard mismatch"
            )
        if gpu_row["learned_runtime_selector_changed"] is not False:
            raise SystemExit(
                "CSR guarded non-success fallback integration GPU runtime flag mismatch"
            )
        if gpu_row["runtime_guard_status"] != "success":
            raise SystemExit(
                "CSR guarded non-success fallback integration GPU runtime guard mismatch"
            )
        if gpu_row["runtime_guard_used_fallback"] is not False:
            raise SystemExit(
                "CSR guarded non-success fallback integration unexpected runtime fallback"
            )
        if gpu_row["trace"]["backend"] != "taichi_gpu":
            raise SystemExit(
                "CSR guarded non-success fallback integration final result not GPU"
            )
    cdde1 = [row for row in gpu_rows if row["matrix_id"] == "suitesparse:Bai/cdde1"][0]
    if set(cdde1["exact_success_candidate_ids"]) != {
        "taichi_csr_bicgstab_none_float64",
        "taichi_csr_bicgstab_jacobi_float64",
        "taichi_csr_gmres_jacobi_restart32_float64",
    }:
        raise SystemExit(
            "CSR guarded non-success fallback integration exact fallback ids mismatch"
        )
    if any(row["trace"] is not None for row in guard_rows):
        raise SystemExit(
            "CSR guarded non-success fallback integration guard rows executed GPU"
        )
    if any(row["exact_success_candidate_count"] != 0 for row in guard_rows):
        raise SystemExit(
            "CSR guarded non-success fallback integration guard exact count mismatch"
        )
    if any(not _guard_blocks_non_success_candidate(row) for row in rows):
        raise SystemExit(
            "CSR guarded non-success fallback integration learned guard split mismatch"
        )
    if any(row["failure_reasons"] for row in rows):
        raise SystemExit(
            "CSR guarded non-success fallback integration recorded failure reasons"
        )
    if manifest.metadata["resolved_exact_fallback_scenarios"] != 2:
        raise SystemExit(
            "CSR guarded non-success fallback integration manifest resolved mismatch"
        )
    if manifest.metadata["unresolved_exact_fallback_scenarios"] != 2:
        raise SystemExit(
            "CSR guarded non-success fallback integration manifest unresolved mismatch"
        )
    if manifest.metadata["executed_gpu_scenarios"] != 2:
        raise SystemExit(
            "CSR guarded non-success fallback integration manifest GPU mismatch"
        )
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit(
            "CSR guarded non-success fallback integration manifest runtime flag mismatch"
        )


def _guard_blocks_non_success_candidate(row: dict) -> bool:
    status = row["learned_guard_status"]
    reasons = tuple(str(reason) for reason in row["learned_guard_reasons"])
    if status == "blocked_non_success_candidate":
        return True
    return status == "blocked_fallback_chain" and any(
        "not a profiled success" in reason for reason in reasons
    )


def _verify_csr_ilu0_guarded_integration(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR ILU0 guarded integration manifest: {stale}")
    rows = read_jsonl(path / "csr_ilu0_guarded_integration_results.jsonl")
    selector_rows = read_jsonl(path / "csr_ilu0_guarded_selector_rows.jsonl")
    predictions = read_jsonl(path / "fixture_ilu0_predictions.jsonl")
    summary = json.loads(
        (path / "csr_ilu0_guarded_integration_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_ilu0_guarded_integration_schema.json").read_text(
            encoding="utf-8"
        )
    )
    quality_gate = json.loads(
        (path / "fixture_runtime_eligible_quality_gate_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_ilu0_guarded_integration":
        raise SystemExit("unexpected CSR ILU0 guarded integration artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR ILU0 guarded integration did not pass")
    if summary["schema_version"] != "phase1_csr_ilu0_guarded_integration_v1":
        raise SystemExit("CSR ILU0 guarded integration schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR ILU0 guarded integration schema/summary mismatch")
    if schema["safety_policy"]["production_runtime_selector_changed"] is not False:
        raise SystemExit("CSR ILU0 guarded integration production selector mismatch")
    if quality_gate["challenger_runtime_eligible"] is not True:
        raise SystemExit("CSR ILU0 guarded fixture gate mismatch")
    if len(rows) != 2 or len(selector_rows) != 2 or len(predictions) != 2:
        raise SystemExit("CSR ILU0 guarded integration row count mismatch")
    if summary["ilu0_source_rows"] != 2:
        raise SystemExit("CSR ILU0 guarded integration source row count mismatch")
    if summary["ilu0_selector_rows"] != 2:
        raise SystemExit("CSR ILU0 guarded integration selector row count mismatch")
    if summary["ilu0_success_rows_imported"] != 1:
        raise SystemExit("CSR ILU0 guarded integration success import mismatch")
    if summary["ilu0_failed_numeric_gate_rows_imported"] != 1:
        raise SystemExit("CSR ILU0 guarded integration failure import mismatch")
    if summary["promoted_success_rows"] != 1:
        raise SystemExit("CSR ILU0 guarded integration promotion count mismatch")
    if summary["failed_numeric_gate_rows_blocked"] != 1:
        raise SystemExit("CSR ILU0 guarded integration blocked count mismatch")
    if summary["executed_gpu_scenarios"] != 1:
        raise SystemExit("CSR ILU0 guarded integration GPU count mismatch")
    if summary["guard_only_scenarios"] != 1:
        raise SystemExit("CSR ILU0 guarded integration guard-only count mismatch")
    if summary["production_runtime_selector_changed"] is not False:
        raise SystemExit("CSR ILU0 guarded integration changed production runtime")
    if summary["fixture_runtime_selector_changed_count"] != 1:
        raise SystemExit("CSR ILU0 guarded integration fixture runtime count mismatch")
    if set(summary["selected_solver_set"]) != {"bicgstab"}:
        raise SystemExit("CSR ILU0 guarded integration solver set mismatch")
    if set(summary["selected_preconditioner_set"]) != {"ilu0"}:
        raise SystemExit("CSR ILU0 guarded integration preconditioner set mismatch")
    if float(summary["max_final_relative_residual"]) > 1.0e-5:
        raise SystemExit("CSR ILU0 guarded integration trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("CSR ILU0 guarded integration CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("CSR ILU0 guarded integration solution error check failed")
    selector_by_matrix = {row["matrix_id"]: row for row in selector_rows}
    if selector_by_matrix["suitesparse:Bai/cdde1"]["target_status"] != "success":
        raise SystemExit("CSR ILU0 guarded integration missing Bai success row")
    if (
        selector_by_matrix["suitesparse:MathWorks/tomography"]["target_status"]
        != "failed_numeric_gate"
    ):
        raise SystemExit("CSR ILU0 guarded integration missing tomography failed row")
    rows_by_matrix = {row["matrix_id"]: row for row in rows}
    bai = rows_by_matrix["suitesparse:Bai/cdde1"]
    if bai["learned_guard_status"] != "promoted":
        raise SystemExit("CSR ILU0 guarded integration Bai was not promoted")
    if bai["runtime_selection_source"] != "learned":
        raise SystemExit("CSR ILU0 guarded integration Bai source mismatch")
    if bai["learned_runtime_selector_changed"] is not True:
        raise SystemExit("CSR ILU0 guarded integration Bai fixture runtime flag mismatch")
    if bai["executed_gpu"] is not True or bai["trace"]["backend"] != "taichi_gpu":
        raise SystemExit("CSR ILU0 guarded integration Bai did not execute GPU")
    if bai["trace"]["metadata"]["preconditioner"] != "ilu0":
        raise SystemExit("CSR ILU0 guarded integration Bai preconditioner mismatch")
    tomography = rows_by_matrix["suitesparse:MathWorks/tomography"]
    if tomography["learned_guard_status"] != "blocked_non_success_candidate":
        raise SystemExit("CSR ILU0 guarded integration tomography not blocked")
    if tomography["executed_gpu"] is not False or tomography["trace"] is not None:
        raise SystemExit("CSR ILU0 guarded integration tomography executed GPU")
    if tomography["learned_selected_target_status"] != "failed_numeric_gate":
        raise SystemExit("CSR ILU0 guarded integration tomography target mismatch")
    if tomography["learned_runtime_selector_changed"] is not False:
        raise SystemExit("CSR ILU0 guarded integration tomography changed runtime")
    if any(row["status"] != "success" or row["failure_reasons"] for row in rows):
        raise SystemExit("CSR ILU0 guarded integration recorded row failure")
    if manifest.metadata["ilu0_success_rows_imported"] != 1:
        raise SystemExit("CSR ILU0 guarded integration manifest success mismatch")
    if manifest.metadata["failed_numeric_gate_rows_blocked"] != 1:
        raise SystemExit("CSR ILU0 guarded integration manifest blocked mismatch")
    if manifest.metadata["production_runtime_selector_changed"] is not False:
        raise SystemExit("CSR ILU0 guarded integration manifest runtime mismatch")


def _verify_csr_ilu0_coverage_expansion(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR ILU0 coverage expansion manifest: {stale}")
    rows = read_jsonl(path / "csr_ilu0_coverage_results.jsonl")
    selector_rows = read_jsonl(path / "csr_ilu0_coverage_selector_rows.jsonl")
    summary = json.loads(
        (path / "csr_ilu0_coverage_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "csr_ilu0_coverage_schema.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "csr_ilu0_coverage_expansion":
        raise SystemExit("unexpected CSR ILU0 coverage expansion artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR ILU0 coverage expansion did not pass")
    if summary["schema_version"] != "phase1_csr_ilu0_coverage_expansion_v1":
        raise SystemExit("CSR ILU0 coverage expansion schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR ILU0 coverage expansion schema/summary mismatch")
    if schema["integration_boundary"]["production_runtime_selector_changed"] is not False:
        raise SystemExit("CSR ILU0 coverage expansion production selector mismatch")
    if schema["integration_boundary"]["merged_into_main_transformer_ready"] is not False:
        raise SystemExit("CSR ILU0 coverage expansion merge boundary mismatch")
    if len(rows) != 8 or len(selector_rows) != 8:
        raise SystemExit("CSR ILU0 coverage expansion row count mismatch")
    if summary["candidate_rows"] != 8 or summary["selector_rows"] != 8:
        raise SystemExit("CSR ILU0 coverage expansion summary row count mismatch")
    if summary["numeric_success_rows"] < 3:
        raise SystemExit("CSR ILU0 coverage expansion expected at least three successes")
    if summary["failed_numeric_gate_rows"] < 1:
        raise SystemExit("CSR ILU0 coverage expansion expected failed numeric row")
    if summary["setup_failed_rows"] < 1:
        raise SystemExit("CSR ILU0 coverage expansion expected setup failure row")
    if summary["merge_ready_success_rows"] != summary["numeric_success_rows"]:
        raise SystemExit("CSR ILU0 coverage expansion merge-ready mismatch")
    if summary["gpu_solve_rows"] != (
        summary["numeric_success_rows"] + summary["failed_numeric_gate_rows"]
    ):
        raise SystemExit("CSR ILU0 coverage expansion GPU solve count mismatch")
    if summary["gpu_attempted_rows"] != summary["candidate_rows"]:
        raise SystemExit("CSR ILU0 coverage expansion GPU attempted count mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR ILU0 coverage expansion changed runtime selector")
    if summary["merged_into_main_transformer_ready"] is not False:
        raise SystemExit("CSR ILU0 coverage expansion unexpectedly merged rows")
    if float(summary["max_success_final_relative_residual"]) > float(
        summary["tolerance_rel"]
    ):
        raise SystemExit("CSR ILU0 coverage expansion residual gate failed")
    if float(summary["max_success_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("CSR ILU0 coverage expansion CPU residual gate failed")
    if float(summary["max_success_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("CSR ILU0 coverage expansion solution error gate failed")
    required_successes = {
        "suitesparse:JGD_Trefethen/Trefethen_20b",
        "suitesparse:FIDAP/ex5",
        "suitesparse:Bai/cdde1",
    }
    if not required_successes.issubset(set(summary["success_matrix_ids"])):
        raise SystemExit("CSR ILU0 coverage expansion missing required successes")
    rows_by_matrix = {row["matrix_id"]: row for row in rows}
    if rows_by_matrix["suitesparse:HB/young3c"]["numeric_status"] != "failed_numeric_gate":
        raise SystemExit("CSR ILU0 coverage expansion missing young3c failure row")
    if rows_by_matrix["suitesparse:Grund/b1_ss"]["numeric_status"] != "setup_failed":
        raise SystemExit("CSR ILU0 coverage expansion missing setup failure row")
    selector_by_matrix = {row["matrix_id"]: row for row in selector_rows}
    for matrix_id in required_successes:
        row = rows_by_matrix[matrix_id]
        selector_row = selector_by_matrix[matrix_id]
        if row["gpu_executed"] is not True:
            raise SystemExit("CSR ILU0 coverage expansion success did not execute GPU")
        if row["candidate_promoted"] is not True:
            raise SystemExit("CSR ILU0 coverage expansion success not promotable")
        if row["runtime_selector_changed"] is not False:
            raise SystemExit("CSR ILU0 coverage expansion success changed selector")
        if selector_row["target_status"] != "success":
            raise SystemExit("CSR ILU0 coverage expansion selector success mismatch")
        if float(selector_row["target_success_rate"]) != 1.0:
            raise SystemExit("CSR ILU0 coverage expansion selector success rate mismatch")
    for row in rows:
        if row["solver"] != "bicgstab" or row["preconditioner"] != "ilu0":
            raise SystemExit("CSR ILU0 coverage expansion solver/preconditioner mismatch")
        if row["production_runtime_selector_changed"] is not False:
            raise SystemExit("CSR ILU0 coverage expansion production flag mismatch")
        if row["numeric_status"] != "success" and row["candidate_promoted"] is not False:
            raise SystemExit("CSR ILU0 coverage expansion promoted non-success row")
    if manifest.metadata["numeric_success_rows"] != summary["numeric_success_rows"]:
        raise SystemExit("CSR ILU0 coverage expansion manifest success mismatch")
    if manifest.metadata["setup_failed_rows"] != summary["setup_failed_rows"]:
        raise SystemExit("CSR ILU0 coverage expansion manifest setup mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR ILU0 coverage expansion manifest runtime mismatch")


def _verify_csr_unresolved_fallback_coverage_plan(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR unresolved fallback coverage plan manifest: {stale}")
    rows = read_jsonl(path / "csr_unresolved_fallback_candidate_plan.jsonl")
    diagnostics = read_jsonl(path / "csr_unresolved_fallback_matrix_diagnostics.jsonl")
    summary = json.loads(
        (path / "csr_unresolved_fallback_coverage_plan_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_unresolved_fallback_coverage_plan_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_unresolved_fallback_coverage_plan":
        raise SystemExit("unexpected CSR unresolved fallback coverage plan artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR unresolved fallback coverage plan did not pass")
    if summary["schema_version"] != "phase1_csr_unresolved_fallback_coverage_plan_v1":
        raise SystemExit("CSR unresolved fallback coverage plan schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR unresolved fallback coverage plan schema/summary mismatch")
    if schema["integration_boundary"]["plan_only"] is not True:
        raise SystemExit("CSR unresolved fallback coverage plan should be plan-only")
    if summary["executes_gpu"] is not False:
        raise SystemExit("CSR unresolved fallback coverage plan should not execute GPU")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR unresolved fallback coverage plan changed selector")
    if len(diagnostics) != 2 or summary["unresolved_matrix_count"] != 2:
        raise SystemExit("CSR unresolved fallback coverage diagnostic count mismatch")
    if len(rows) != 6 or summary["candidate_plan_rows"] != 6:
        raise SystemExit("CSR unresolved fallback coverage row count mismatch")
    if set(summary["unresolved_matrices"]) != {
        "suitesparse:Gset/G17",
        "suitesparse:Zitney/extr1b",
    }:
        raise SystemExit("CSR unresolved fallback coverage matrix set mismatch")
    if summary["cpu_screen_ready_candidates"] != 4:
        raise SystemExit("CSR unresolved fallback coverage CPU-ready count mismatch")
    if summary["future_dependency_candidates"] != 2:
        raise SystemExit("CSR unresolved fallback coverage future count mismatch")
    if summary["by_plan_kind"] != {
        "existing_solver_parameter_sweep": 4,
        "future_preconditioner_required": 1,
        "matrix_formulation_diagnostic": 1,
    }:
        raise SystemExit("CSR unresolved fallback coverage plan-kind mismatch")
    if summary["by_execution_stage"] != {
        "blocked_until_preconditioner_exists": 1,
        "cpu_screen_ready": 4,
        "diagnostic_only_no_gpu": 1,
    }:
        raise SystemExit("CSR unresolved fallback coverage execution-stage mismatch")
    if any(row["executes_gpu"] is not False for row in rows):
        raise SystemExit("CSR unresolved fallback coverage row executed GPU")
    if any(row["runtime_selector_changed"] is not False for row in rows):
        raise SystemExit("CSR unresolved fallback coverage row changed selector")
    if sum(1 for row in rows if row["cpu_screen_first"]) != 4:
        raise SystemExit("CSR unresolved fallback coverage CPU-screen flag mismatch")
    if sum(1 for row in rows if row["gpu_allowed_after_cpu_screen"]) != 4:
        raise SystemExit("CSR unresolved fallback coverage GPU-after-screen flag mismatch")
    if not all(row["source_evidence"]["probe_attempt_count"] >= 3 for row in rows):
        raise SystemExit("CSR unresolved fallback coverage missing probe evidence")
    if not any(
        row["matrix_id"] == "suitesparse:Gset/G17"
        and row["plan_kind"] == "matrix_formulation_diagnostic"
        and row["execution_stage"] == "diagnostic_only_no_gpu"
        and row["source_evidence"]["zero_diagonal_observed"] is True
        for row in rows
    ):
        raise SystemExit("CSR unresolved fallback coverage missing Gset diagnostic")
    if not any(
        row["matrix_id"] == "suitesparse:Zitney/extr1b"
        and row["plan_kind"] == "future_preconditioner_required"
        and row["execution_stage"] == "blocked_until_preconditioner_exists"
        for row in rows
    ):
        raise SystemExit("CSR unresolved fallback coverage missing Zitney preconditioner plan")
    if manifest.metadata["candidate_plan_rows"] != 6:
        raise SystemExit("CSR unresolved fallback coverage manifest row mismatch")
    if manifest.metadata["cpu_screen_ready_candidates"] != 4:
        raise SystemExit("CSR unresolved fallback coverage manifest CPU-ready mismatch")
    if manifest.metadata["future_dependency_candidates"] != 2:
        raise SystemExit("CSR unresolved fallback coverage manifest future mismatch")
    if manifest.metadata["executes_gpu"] is not False:
        raise SystemExit("CSR unresolved fallback coverage manifest GPU flag mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR unresolved fallback coverage manifest runtime flag mismatch")


def _verify_csr_unresolved_fallback_cpu_screen(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR unresolved fallback CPU screen manifest: {stale}")
    rows = read_jsonl(path / "csr_unresolved_fallback_cpu_screen_results.jsonl")
    summary = json.loads(
        (path / "csr_unresolved_fallback_cpu_screen_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_unresolved_fallback_cpu_screen_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_unresolved_fallback_cpu_screen":
        raise SystemExit("unexpected CSR unresolved fallback CPU screen artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR unresolved fallback CPU screen did not pass")
    if summary["schema_version"] != "phase1_csr_unresolved_fallback_cpu_screen_v1":
        raise SystemExit("CSR unresolved fallback CPU screen schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR unresolved fallback CPU screen schema/summary mismatch")
    if schema["integration_boundary"]["cpu_only"] is not True:
        raise SystemExit("CSR unresolved fallback CPU screen should be CPU-only")
    if summary["executes_gpu"] is not False:
        raise SystemExit("CSR unresolved fallback CPU screen should not execute GPU")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR unresolved fallback CPU screen changed selector")
    if len(rows) != 4 or summary["attempted_cpu_screens"] != 4:
        raise SystemExit("CSR unresolved fallback CPU screen row count mismatch")
    if summary["screen_success_rows"] != 0:
        raise SystemExit("CSR unresolved fallback CPU screen unexpectedly passed a row")
    if summary["cpu_screened_out_rows"] != 4:
        raise SystemExit("CSR unresolved fallback CPU screen screened-out mismatch")
    if summary["gpu_probe_ready_candidates"] != 0:
        raise SystemExit("CSR unresolved fallback CPU screen GPU-ready mismatch")
    if summary["unresolved_after_cpu_screen"] != 4:
        raise SystemExit("CSR unresolved fallback CPU screen unresolved mismatch")
    if summary["by_status"] != {"screened_out": 4}:
        raise SystemExit("CSR unresolved fallback CPU screen status mismatch")
    if summary["by_solver_status"] != {
        "bicgstab:screened_out": 1,
        "gmres:screened_out": 3,
    }:
        raise SystemExit("CSR unresolved fallback CPU screen solver/status mismatch")
    if summary["by_matrix_status"] != {
        "suitesparse:Gset/G17:screened_out": 2,
        "suitesparse:Zitney/extr1b:screened_out": 2,
    }:
        raise SystemExit("CSR unresolved fallback CPU screen matrix/status mismatch")
    if any(row["status"] != "screened_out" for row in rows):
        raise SystemExit("CSR unresolved fallback CPU screen row status mismatch")
    if any(row["gpu_probe_ready"] is not False for row in rows):
        raise SystemExit("CSR unresolved fallback CPU screen row GPU-ready mismatch")
    if any(row["executes_gpu"] is not False for row in rows):
        raise SystemExit("CSR unresolved fallback CPU screen row executed GPU")
    if any(row["runtime_selector_changed"] is not False for row in rows):
        raise SystemExit("CSR unresolved fallback CPU screen row changed selector")
    if any(not row["failure_reason"].startswith("cpu_screen_") for row in rows):
        raise SystemExit("CSR unresolved fallback CPU screen missing failure reasons")
    if not any(
        row["matrix_id"] == "suitesparse:Gset/G17"
        and row["candidate_id"] == "taichi_csr_bicgstab_none_maxiter2048_float64"
        and row["failure_reason"] == "cpu_screen_solution_error_above_tolerance"
        and float(row["solution_relative_error"]) > 5.0e-3
        for row in rows
    ):
        raise SystemExit("CSR unresolved fallback CPU screen missing Gset BiCGSTAB screen")
    if summary["next_step"] != (
        "defer_unresolved_to_future_preconditioners_or_formulation_diagnostics"
    ):
        raise SystemExit("CSR unresolved fallback CPU screen next step mismatch")
    if not any(
        row["matrix_id"] == "suitesparse:Zitney/extr1b"
        and row["candidate_id"] == "taichi_csr_gmres_none_restart64_float64"
        and row["failure_reason"] == "cpu_screen_residual_above_tolerance"
        for row in rows
    ):
        raise SystemExit("CSR unresolved fallback CPU screen missing Zitney GMRES screen")
    if manifest.metadata["attempted_cpu_screens"] != 4:
        raise SystemExit("CSR unresolved fallback CPU screen manifest row mismatch")
    if manifest.metadata["screen_success_rows"] != 0:
        raise SystemExit("CSR unresolved fallback CPU screen manifest success mismatch")
    if manifest.metadata["cpu_screened_out_rows"] != 4:
        raise SystemExit("CSR unresolved fallback CPU screen manifest screened mismatch")
    if manifest.metadata["gpu_probe_ready_candidates"] != 0:
        raise SystemExit("CSR unresolved fallback CPU screen manifest GPU-ready mismatch")
    if manifest.metadata["executes_gpu"] is not False:
        raise SystemExit("CSR unresolved fallback CPU screen manifest GPU flag mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR unresolved fallback CPU screen manifest runtime flag mismatch")


def _verify_csr_unresolved_matrix_diagnostics(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR unresolved matrix diagnostics manifest: {stale}")
    rows = read_jsonl(path / "csr_unresolved_matrix_diagnostics.jsonl")
    summary = json.loads(
        (path / "csr_unresolved_matrix_diagnostics_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_unresolved_matrix_diagnostics_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_unresolved_matrix_diagnostics":
        raise SystemExit("unexpected CSR unresolved matrix diagnostics artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR unresolved matrix diagnostics did not pass")
    if summary["schema_version"] != "phase1_csr_unresolved_matrix_diagnostics_v1":
        raise SystemExit("CSR unresolved matrix diagnostics schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR unresolved matrix diagnostics schema/summary mismatch")
    if schema["integration_boundary"]["diagnostic_only"] is not True:
        raise SystemExit("CSR unresolved matrix diagnostics should be diagnostic-only")
    if summary["executes_gpu"] is not False:
        raise SystemExit("CSR unresolved matrix diagnostics should not execute GPU")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("CSR unresolved matrix diagnostics changed selector")
    if len(rows) != 2 or summary["diagnosed_matrices"] != 2:
        raise SystemExit("CSR unresolved matrix diagnostics row count mismatch")
    if summary["source_cpu_screen_rows"] != 4:
        raise SystemExit("CSR unresolved matrix diagnostics source screen mismatch")
    if summary["source_gpu_probe_ready_candidates"] != 0:
        raise SystemExit("CSR unresolved matrix diagnostics GPU-ready source mismatch")
    if summary["by_recommended_route"] != {
        "formulation_diagnostic_required": 1,
        "ilu_or_nonsymmetric_scaling_preconditioner": 1,
    }:
        raise SystemExit("CSR unresolved matrix diagnostics route counts mismatch")
    if summary["matrix_routes"] != {
        "suitesparse:Gset/G17": "formulation_diagnostic_required",
        "suitesparse:Zitney/extr1b": "ilu_or_nonsymmetric_scaling_preconditioner",
    }:
        raise SystemExit("CSR unresolved matrix diagnostics matrix route mismatch")
    if any(row["executes_gpu"] is not False for row in rows):
        raise SystemExit("CSR unresolved matrix diagnostics row executed GPU")
    if any(row["runtime_selector_changed"] is not False for row in rows):
        raise SystemExit("CSR unresolved matrix diagnostics row changed selector")
    gset = _matrix_row(rows, "suitesparse:Gset/G17")
    if gset["recommended_route"] != "formulation_diagnostic_required":
        raise SystemExit("G17 diagnostic route mismatch")
    if int(gset["diagnostics"]["diagonal"]["zero_count"]) != 800:
        raise SystemExit("G17 zero diagonal diagnostic mismatch")
    if gset["diagnostics"]["dense_cholesky_probe"]["status"] != "failed":
        raise SystemExit("G17 Cholesky diagnostic mismatch")
    if float(gset["diagnostics"]["symmetric_spectrum_probe"]["ritz_min"]) >= 0.0:
        raise SystemExit("G17 spectrum diagnostic mismatch")
    zitney = _matrix_row(rows, "suitesparse:Zitney/extr1b")
    if zitney["recommended_route"] != "ilu_or_nonsymmetric_scaling_preconditioner":
        raise SystemExit("Zitney diagnostic route mismatch")
    if int(zitney["diagnostics"]["diagonal"]["zero_count"]) != 2834:
        raise SystemExit("Zitney zero diagonal diagnostic mismatch")
    if zitney["diagnostics"]["dense_cholesky_probe"]["status"] != "skipped_nonsymmetric":
        raise SystemExit("Zitney Cholesky diagnostic mismatch")
    if summary["next_step"] != (
        "implement_and_test_preconditioner_or_formulation_candidates_before_gpu_probe"
    ):
        raise SystemExit("CSR unresolved matrix diagnostics next step mismatch")
    if manifest.metadata["diagnosed_matrices"] != 2:
        raise SystemExit("CSR unresolved matrix diagnostics manifest row mismatch")
    if manifest.metadata["executes_gpu"] is not False:
        raise SystemExit("CSR unresolved matrix diagnostics manifest GPU flag mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("CSR unresolved matrix diagnostics manifest runtime flag mismatch")


def _matrix_row(rows: list[dict] | tuple[dict, ...], matrix_id: str) -> dict:
    for row in rows:
        if row["matrix_id"] == matrix_id:
            return row
    raise SystemExit(f"missing matrix row: {matrix_id}")


def _verify_public_release_hygiene(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale public-release hygiene manifest: {stale}")
    rows = read_jsonl(path / "public_release_hygiene_rows.jsonl")
    summary = json.loads(
        (path / "public_release_hygiene_summary.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (path / "public_release_hygiene_schema.json").read_text(encoding="utf-8")
    )
    if manifest.artifact_kind != "public_release_hygiene":
        raise SystemExit("unexpected public-release hygiene artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("public-release hygiene did not pass")
    if summary["schema_version"] != "phase1_public_release_hygiene_v1":
        raise SystemExit("public-release hygiene schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("public-release hygiene schema/summary mismatch")
    if summary["public_release_hygiene_ready"] is not True:
        raise SystemExit("public-release hygiene not ready")
    if summary["github_upload_ready"] is not True:
        raise SystemExit("public-release hygiene should be GitHub-upload ready")
    if summary["project_owner"] != "Wei CUI":
        raise SystemExit("public-release hygiene owner mismatch")
    if summary["citation_author"] != "Wei CUI":
        raise SystemExit("public-release hygiene citation mismatch")
    if summary["license_route_selected"] is not True:
        raise SystemExit("public-release hygiene license route mismatch")
    if summary["formal_license_finalized"] is not True:
        raise SystemExit("public-release hygiene license finalization mismatch")
    if summary["source_code_license"] != "PolyForm Noncommercial License 1.0.0":
        raise SystemExit("public-release hygiene source license mismatch")
    if summary["documentation_license_direction"] != "CC BY-NC 4.0":
        raise SystemExit("public-release hygiene documentation license mismatch")
    if summary["model_terms_documented"] is not True:
        raise SystemExit("public-release hygiene model terms mismatch")
    if summary["contributor_terms_documented"] is not True:
        raise SystemExit("public-release hygiene contributor terms mismatch")
    if summary["commercial_use_requires_permission"] is not True:
        raise SystemExit("public-release hygiene commercial-use mismatch")
    if summary["license_policy_documented"] is not True:
        raise SystemExit("public-release hygiene missing license policy")
    if summary["permissive_license_family_rejected"] is not True:
        raise SystemExit("public-release hygiene permissive-license flag mismatch")
    if summary["license_upload_blockers"] != []:
        raise SystemExit("public-release hygiene license blocker mismatch")
    if summary["runtime_selector_changed"] is not False:
        raise SystemExit("public-release hygiene changed runtime selector")
    if summary["git_repository_initialized"] is not True:
        raise SystemExit("public-release hygiene expected initialized git workspace")
    if summary["git_init_required"] is not False:
        raise SystemExit("public-release hygiene git-init flag mismatch")
    if summary["missing_gitignore_patterns"]:
        raise SystemExit("public-release hygiene missing gitignore patterns")
    if summary["large_unignored_file_count"] != 0:
        raise SystemExit("public-release hygiene large file count mismatch")
    if summary["secret_like_file_count"] != 0:
        raise SystemExit("public-release hygiene secret-like file count mismatch")
    if summary["validation_error_count"] != 0 or summary["validation_errors"]:
        raise SystemExit("public-release hygiene validation errors found")
    if summary["public_source_path_count"] < 10:
        raise SystemExit("public-release hygiene source path count mismatch")
    if summary["required_release_doc_count"] < 9:
        raise SystemExit("public-release hygiene release doc count mismatch")
    if not {
        "LICENSE",
        "docs/PROJECT_OVERVIEW.md",
        "LICENSE_POLICY.md",
        "COMMERCIAL_USE.md",
        "MODEL_CONTRIBUTION_TERMS.md",
        "CONTRIBUTOR_LICENSE_AGREEMENT.md",
        "CITATION.cff",
        "CONTRIBUTING.md",
    }.issubset(set(summary["required_release_docs"])):
        raise SystemExit("public-release hygiene release docs mismatch")
    if not {
        "ADVICE.md",
        "MILESTONE_LOG.md",
        "FOOTAGE_SPEC.md",
        "PUBLIC_RELEASE.md",
        "GITHUB_RELEASE_GATE.md",
        "footage/",
        "docs/toms_paper/",
        "reports/",
    }.issubset(set(summary["required_gitignore_patterns"])):
        raise SystemExit("public-release hygiene private doc ignore mismatch")
    if {
        "ADVICE.md",
        "MILESTONE_LOG.md",
        "FOOTAGE_SPEC.md",
        "PUBLIC_RELEASE.md",
        "GITHUB_RELEASE_GATE.md",
        "footage",
        "docs",
    } & set(summary["public_source_paths"]):
        raise SystemExit("public-release hygiene leaked private docs")
    if summary["workspace_file_count_scanned"] <= 0:
        raise SystemExit("public-release hygiene scan did not inspect files")
    if "commit small JSON/Markdown runs artifacts" not in summary["generated_artifact_policy"]:
        raise SystemExit("public-release hygiene generated artifact policy mismatch")
    if "SuiteSparse" not in summary["external_dataset_policy"]:
        raise SystemExit("public-release hygiene external dataset policy mismatch")
    if not (path / "public_release_checklist.md").exists():
        raise SystemExit("public-release hygiene missing checklist")
    if not (path / "public_release_hygiene_report.md").exists():
        raise SystemExit("public-release hygiene missing report")
    rows_by_kind: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_kind.setdefault(row["row_kind"], []).append(row)
        if row["schema_version"] != summary["schema_version"]:
            raise SystemExit("public-release hygiene row schema mismatch")
        if row["blocks_hygiene"] is not False:
            raise SystemExit("public-release hygiene row blocks release")
    if "gitignore_required_pattern" not in rows_by_kind:
        raise SystemExit("public-release hygiene missing gitignore rows")
    if "public_source_path" not in rows_by_kind:
        raise SystemExit("public-release hygiene missing source path rows")
    if "release_required_document" not in rows_by_kind:
        raise SystemExit("public-release hygiene missing release document rows")
    if "license_decision" not in rows_by_kind:
        raise SystemExit("public-release hygiene missing license decision rows")
    if "citation" not in rows_by_kind:
        raise SystemExit("public-release hygiene missing citation row")
    if not any(
        row["path"] == ".venv-tss"
        for row in rows_by_kind.get("local_only_dir_present", [])
    ):
        raise SystemExit("public-release hygiene missing venv local-only row")
    if manifest.metadata["public_release_hygiene_ready"] is not True:
        raise SystemExit("public-release hygiene manifest ready mismatch")
    if manifest.metadata["runtime_selector_changed"] is not False:
        raise SystemExit("public-release hygiene manifest runtime mismatch")


def _verify_transformer_readiness(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale transformer readiness artifact manifest: {stale}")
    rows = read_jsonl(path / "policy_training_rows.jsonl")
    schema = json.loads(
        (path / "transformer_feature_schema.json").read_text(encoding="utf-8")
    )
    if len(rows) != 21:
        raise SystemExit(f"expected 21 policy training rows, got {len(rows)}")
    if sum(1 for row in rows if row["label_is_oracle"]) != 7:
        raise SystemExit("expected seven oracle training labels")
    if schema["schema_version"] != "phase1_policy_features_v1":
        raise SystemExit("unexpected transformer feature schema version")
    if schema["integration_boundary"]["status"] != "ready_for_dataset_scaleout":
        raise SystemExit("transformer readiness boundary not marked ready")


def _verify_public_api_smoke(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale public API smoke artifact manifest: {stale}")
    payload = json.loads((path / "public_api_smoke.json").read_text(encoding="utf-8"))
    trace = payload["trace"]
    if payload["status"] != "success":
        raise SystemExit("public API smoke did not succeed")
    if trace["backend"] != "taichi_gpu":
        raise SystemExit("public API smoke did not use taichi_gpu")
    if trace["final_residual_norm"] > 1.0e-6:
        raise SystemExit("public API smoke residual check failed")
    if trace["metadata"]["relative_error_to_true"] >= 5.0e-3:
        raise SystemExit("public API smoke relative error check failed")
    if payload["num_training_rows"] != 21:
        raise SystemExit("public API smoke training row API check failed")
    if payload["num_dataset_plan_entries"] != 3:
        raise SystemExit("public API smoke dataset plan API check failed")


def _verify_csr_public_api_smoke(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR public API smoke artifact manifest: {stale}")
    payload = json.loads((path / "csr_public_api_smoke.json").read_text(encoding="utf-8"))
    summary = payload["summary"]
    diagnostics = payload["diagnostics"]
    results = payload["results"]
    if manifest.artifact_kind != "csr_public_api_smoke":
        raise SystemExit("unexpected CSR public API smoke artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR public API smoke did not pass")
    if summary["precision"] != "float64":
        raise SystemExit("CSR public API smoke expected float64 precision")
    if summary["num_matrices"] != 3 or summary["num_solves"] != 5:
        raise SystemExit("CSR public API smoke expected 3 matrices and 5 solves")
    if summary["num_success"] != 5 or summary["num_failed"] != 0:
        raise SystemExit("CSR public API smoke contains failed solves")
    if len(diagnostics) != 3 or len(results) != 5:
        raise SystemExit("CSR public API smoke payload count mismatch")
    cg_diagnostics = [
        row
        for row in diagnostics
        if row["matrix_id"]
        in {"suitesparse:JGD_Trefethen/Trefethen_20b", "suitesparse:FIDAP/ex5"}
    ]
    gmres_diagnostics = [
        row for row in diagnostics if row["matrix_id"] == "suitesparse:HB/curtis54"
    ]
    if any(not row["actual_symmetric"] or not row["cg_candidate"] for row in cg_diagnostics):
        raise SystemExit("CSR public API diagnostics rejected selected CG matrices")
    if any(row["actual_symmetric"] for row in gmres_diagnostics):
        raise SystemExit("CSR public API diagnostics expected nonsymmetric GMRES matrix")
    if float(summary["max_final_relative_residual"]) > 1.0e-5:
        raise SystemExit("CSR public API trace residual check failed")
    if float(summary["max_cpu_recomputed_relative_residual"]) > 1.0e-4:
        raise SystemExit("CSR public API CPU residual check failed")
    if float(summary["max_solution_relative_error"]) > 5.0e-3:
        raise SystemExit("CSR public API solution error check failed")
    if {row["solver"] for row in results} != {"cg", "pcg", "gmres"}:
        raise SystemExit("CSR public API solver set mismatch")
    for row in results:
        if row["status"] != "success":
            raise SystemExit("CSR public API result has non-success status")
        if row["trace"]["backend"] != "taichi_gpu":
            raise SystemExit("CSR public API result did not use taichi_gpu")
        if row["trace"]["metadata"]["operator_backend"] != "taichi_csr":
            raise SystemExit("CSR public API result did not use taichi_csr")


def _verify_solver_functional(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale solver functional artifact manifest: {stale}")
    rows = read_jsonl(path / "solver_functional_results.jsonl")
    if len(rows) != 2:
        raise SystemExit(f"expected 2 solver functional rows, got {len(rows)}")
    solvers = {row["solver"] for row in rows}
    if solvers != {"richardson", "chebyshev"}:
        raise SystemExit(f"unexpected solver functional solvers: {solvers}")
    if any(row["status"] != "success" for row in rows):
        raise SystemExit("solver functional smoke contains failed solver rows")
    if any(row["final_residual_norm"] > 1.0e-6 for row in rows):
        raise SystemExit("solver functional residual check failed")
    if any(row["relative_error_to_true"] >= 5.0e-3 for row in rows):
        raise SystemExit("solver functional relative error check failed")


def _verify_campaign(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale campaign artifact manifest: {stale}")
    summary = json.loads((path / "campaign_summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "passed":
        raise SystemExit("campaign summary is not passed")
    stage_ids = {stage["stage_id"] for stage in summary["stages"]}
    queue_batch_stage_ids = {
        stage_id
        for stage_id in stage_ids
        if stage_id.startswith("csr_queue_batch_")
        and stage_id.removeprefix("csr_queue_batch_").isdigit()
    }
    expected_stage_count = 74 + len(queue_batch_stage_ids)
    if len(summary["stages"]) != expected_stage_count:
        raise SystemExit("campaign summary stage count check failed")
    if "csr_learning_readiness" not in stage_ids:
        raise SystemExit("campaign summary missing CSR learning readiness stage")
    if "csr_model_contract" not in stage_ids:
        raise SystemExit("campaign summary missing CSR model contract stage")
    if "csr_training_tensors" not in stage_ids:
        raise SystemExit("campaign summary missing CSR training tensor stage")
    if "csr_linear_ranker" not in stage_ids:
        raise SystemExit("campaign summary missing CSR linear ranker stage")
    if "csr_selector_model_eval" not in stage_ids:
        raise SystemExit("campaign summary missing CSR selector model eval stage")
    if "csr_benchmark_expansion_plan" not in stage_ids:
        raise SystemExit("campaign summary missing CSR benchmark expansion plan stage")
    if "csr_full_dataset_queue" not in stage_ids:
        raise SystemExit("campaign summary missing CSR full dataset queue stage")
    if "csr_queue_batch_00001" not in stage_ids:
        raise SystemExit("campaign summary missing CSR queue batch stage")
    if "csr_queue_batch_00002" not in stage_ids:
        raise SystemExit("campaign summary missing CSR queue batch 00002 stage")
    if "csr_queue_batch_00003" not in stage_ids:
        raise SystemExit("campaign summary missing CSR queue batch 00003 stage")
    if "csr_queue_training_pool" not in stage_ids:
        raise SystemExit("campaign summary missing CSR queue training pool stage")
    if "csr_queue_batch_training_bundle" not in stage_ids:
        raise SystemExit("campaign summary missing CSR queue batch training bundle stage")
    if "csr_queue_batch_reference_ranker" not in stage_ids:
        raise SystemExit("campaign summary missing CSR queue batch reference ranker stage")
    if "csr_queue_batch_model_replay" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR queue batch model replay stage")
    if "csr_queue_candidate_coverage" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR queue candidate coverage stage")
    if "csr_gmres_restart_coverage" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR GMRES restart coverage stage")
    if "csr_blocked_gap_probe" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR blocked gap probe stage")
    if "csr_blocked_gap_positive_search" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR blocked gap positive search stage")
    if "csr_blocked_gap_training_integration" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR blocked gap training integration stage")
    if "csr_blocked_gap_augmented_ranker" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR blocked gap augmented ranker stage")
    if "csr_blocked_gap_guarded_replay" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR blocked gap guarded replay stage")
    if "csr_transformer_handoff_bundle" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR Transformer handoff bundle stage")
    if "csr_transformer_training_package" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR Transformer training package stage")
    if "csr_transformer_package_consumer_dry_run" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR Transformer package consumer dry run stage")
    if "csr_micro_campaign" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR micro-campaign stage")
    if "csr_transformer_ready" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR Transformer-ready stage")
    if "csr_transformer_ranker" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR Transformer ranker stage")
    if "csr_external_model_adapter" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR external model adapter stage")
    if "csr_transformer_model_replay" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR Transformer model replay stage")
    if "csr_transformer_quality_gate" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR Transformer quality gate stage")
    if "csr_policy_model_artifact" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR policy model artifact stage")
    if "csr_policy_model_acceptance" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR policy model acceptance stage")
    if "csr_policy_model_submission" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR policy model submission stage")
    if "csr_external_model_intake" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR external model intake stage")
    if "csr_transformer_training_entrypoint" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR Transformer training entrypoint stage")
    if "csr_transformer_reference_training_export" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR Transformer reference training export stage")
    if "csr_learned_runtime_guard" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR learned runtime guard stage")
    if "csr_guarded_auto_solve" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR guarded auto-solve stage")
    if "csr_guarded_promotion_readiness" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR guarded promotion readiness stage")
    if "csr_guarded_promotion_coverage_plan" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR guarded promotion coverage stage")
    if "csr_guarded_promotion_coverage_exec" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR guarded promotion coverage exec stage")
    if "csr_non_success_fallback_probe" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR non-success fallback probe stage")
    if "csr_guarded_non_success_fallback_integration" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR guarded non-success fallback integration stage")
    if "csr_ilu0_guarded_integration" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR ILU0 guarded integration stage")
    if "csr_ilu0_coverage_expansion" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR ILU0 coverage expansion stage")
    if "csr_unresolved_fallback_coverage_plan" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR unresolved fallback coverage plan stage")
    if "csr_unresolved_fallback_cpu_screen" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR unresolved fallback CPU screen stage")
    if "csr_unresolved_matrix_diagnostics" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing CSR unresolved matrix diagnostics stage")
    if "public_release_hygiene" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing public-release hygiene stage")
    if "taichi_csr_row_column_equilibration" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing Taichi CSR row/column equilibration stage")
    if "taichi_csr_ilu0" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing Taichi CSR ILU0 stage")
    if "taichi_csr_symmetric_equilibration" not in {stage["stage_id"] for stage in summary["stages"]}:
        raise SystemExit("campaign summary missing Taichi CSR symmetric equilibration stage")
    if any(stage["status"] != "passed" for stage in summary["stages"]):
        raise SystemExit("campaign summary contains failed stage")


if __name__ == "__main__":
    main()
