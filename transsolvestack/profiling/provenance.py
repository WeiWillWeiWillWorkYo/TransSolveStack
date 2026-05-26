"""Artifact provenance manifests and stale-artifact checks."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class FileHash:
    path: str
    sha256: str


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_kind: str
    created_at_utc: str
    command: str
    python_version: str
    file_hashes: tuple[FileHash, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def hash_file(path: str | Path) -> FileHash:
    file_path = Path(path)
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return FileHash(path=str(file_path), sha256=digest)


def build_artifact_manifest(
    artifact_kind: str,
    command: str,
    tracked_files: Iterable[str | Path],
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    return ArtifactManifest(
        artifact_kind=artifact_kind,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        command=command,
        python_version=platform.python_version(),
        file_hashes=tuple(hash_file(path) for path in tracked_files),
        metadata=metadata or {},
    )


def write_manifest(manifest: ArtifactManifest, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return output


def read_manifest(path: str | Path) -> ArtifactManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ArtifactManifest(
        artifact_kind=data["artifact_kind"],
        created_at_utc=data["created_at_utc"],
        command=data["command"],
        python_version=data["python_version"],
        file_hashes=tuple(FileHash(**item) for item in data["file_hashes"]),
        metadata=dict(data.get("metadata", {})),
    )


def verify_manifest_hashes(manifest: ArtifactManifest) -> tuple[str, ...]:
    """Return paths whose current SHA does not match the manifest."""

    stale: list[str] = []
    for item in manifest.file_hashes:
        path = Path(item.path)
        if not path.exists() or hash_file(path).sha256 != item.sha256:
            stale.append(item.path)
    return tuple(stale)


def git_commit_or_unknown() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


CORE_BENCHMARK_PROVENANCE_FILES = (
    "configs/workloads/phase1_smoke.yaml",
    "configs/candidates/phase1_taichi_gpu.yaml",
    "configs/runtime/resource_limits.yaml",
    "transsolvestack/operators/synthetic.py",
    "transsolvestack/operators/taichi_diffusion2d.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "transsolvestack/runtime/plan_resolver.py",
    "transsolvestack/profiling/artifacts.py",
    "transsolvestack/profiling/taichi_benchmark.py",
    "transsolvestack/profiling/protocol.py",
)


CORE_REGRESSION_PROVENANCE_FILES = (
    *CORE_BENCHMARK_PROVENANCE_FILES,
    "transsolvestack/profiling/numerical_regression.py",
)


CORE_SEQUENCE_PROVENANCE_FILES = (
    "configs/workloads/phase1_sequence.yaml",
    "configs/candidates/phase1_taichi_gpu.yaml",
    "configs/runtime/resource_limits.yaml",
    "transsolvestack/operators/synthetic.py",
    "transsolvestack/operators/taichi_diffusion2d.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "transsolvestack/runtime/plan_resolver.py",
    "transsolvestack/benchmarks/sequence_config.py",
    "transsolvestack/profiling/sequence_benchmark.py",
)


CORE_POLICY_SELECTION_PROVENANCE_FILES = (
    "configs/candidates/phase1_taichi_gpu.yaml",
    "runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
    "transsolvestack/policies/artifact_selector.py",
    "transsolvestack/policies/selection_artifacts.py",
    "scripts/tss_select_policies_from_artifact.py",
)


CORE_POLICY_SOLVE_PROVENANCE_FILES = (
    "configs/workloads/phase1_smoke.yaml",
    "configs/candidates/phase1_taichi_gpu.yaml",
    "configs/runtime/resource_limits.yaml",
    "runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
    "transsolvestack/operators/synthetic.py",
    "transsolvestack/operators/taichi_diffusion2d.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/policies/artifact_selector.py",
    "transsolvestack/runtime/engine.py",
    "transsolvestack/runtime/policy_solve.py",
    "scripts/tss_policy_solve_smoke.py",
)


CORE_DATASET_PLAN_PROVENANCE_FILES = (
    "configs/datasets/phase1_external_small.yaml",
    "configs/runtime/resource_limits.yaml",
    "transsolvestack/datasets/catalog.py",
    "scripts/tss_plan_dataset_downloads.py",
)


CORE_MATRIX_MARKET_PROBE_PROVENANCE_FILES = (
    "tests/fixtures/tiny_spd.mtx",
    "transsolvestack/datasets/matrix_market.py",
    "scripts/tss_inspect_matrix_market.py",
)


CORE_SUITESPARSE_INDEX_PROVENANCE_FILES = (
    "configs/runtime/storage_paths.yaml",
    "transsolvestack/datasets/suitesparse_index.py",
    "scripts/tss_build_suitesparse_index.py",
)


CORE_SUITESPARSE_SELECTION_PROVENANCE_FILES = (
    "configs/runtime/storage_paths.yaml",
    "transsolvestack/datasets/suitesparse_index.py",
    "transsolvestack/datasets/selection.py",
    "scripts/tss_select_suitesparse_subset.py",
)


CORE_SUITESPARSE_HEADER_PROBE_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_selection/selected_matrices.jsonl",
    "transsolvestack/datasets/matrix_market.py",
    "transsolvestack/datasets/archive_probe.py",
    "scripts/tss_probe_suitesparse_headers.py",
)


CORE_SUITESPARSE_CSR_IMPORT_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_selection/selected_matrices.jsonl",
    "runs/phase1_suitesparse_header_probe/archive_header_probe.jsonl",
    "transsolvestack/datasets/csr.py",
    "scripts/tss_import_suitesparse_csr.py",
)


CORE_TAICHI_CSR_MATVEC_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/operators/taichi_csr.py",
    "scripts/tss_taichi_csr_matvec_smoke.py",
)


CORE_TAICHI_CSR_PRIMITIVES_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/operators/taichi_csr.py",
    "scripts/tss_taichi_csr_primitives_smoke.py",
)


CORE_TAICHI_CSR_SOLVE_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "transsolvestack/profiling/repeats.py",
    "scripts/tss_taichi_csr_solve_smoke.py",
)


CORE_TAICHI_CSR_BICGSTAB_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "transsolvestack/profiling/repeats.py",
    "scripts/tss_taichi_csr_bicgstab_smoke.py",
)


CORE_TAICHI_CSR_GMRES_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "transsolvestack/profiling/repeats.py",
    "scripts/tss_taichi_csr_gmres_smoke.py",
)


CORE_TAICHI_CSR_RICHARDSON_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "transsolvestack/profiling/repeats.py",
    "scripts/tss_taichi_csr_richardson_smoke.py",
)


CORE_TAICHI_CSR_CHEBYSHEV_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "transsolvestack/profiling/repeats.py",
    "scripts/tss_taichi_csr_chebyshev_smoke.py",
)


CORE_TAICHI_CSR_SYMMETRIC_EQUILIBRATION_PROVENANCE_FILES = (
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/preconditioners/registry.py",
    "transsolvestack/preconditioners/taichi_jacobi.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_taichi_csr_symmetric_equilibration_smoke.py",
)


CORE_TAICHI_CSR_ROW_COLUMN_EQUILIBRATION_PROVENANCE_FILES = (
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/preconditioners/registry.py",
    "transsolvestack/preconditioners/taichi_jacobi.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_taichi_csr_row_column_equilibration_smoke.py",
)


CORE_TAICHI_CSR_ILU0_PROVENANCE_FILES = (
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/preconditioners/registry.py",
    "transsolvestack/preconditioners/taichi_jacobi.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_taichi_csr_ilu0_smoke.py",
)


CORE_CSR_PUBLIC_API_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/datasets/diagnostics.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_csr_public_api_smoke.py",
)


CORE_CSR_SELECTOR_READINESS_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_taichi_csr_solve/csr_solve_results.jsonl",
    "runs/phase1_taichi_csr_solve/csr_solve_summary.json",
    "runs/phase1_taichi_csr_bicgstab/csr_bicgstab_results.jsonl",
    "runs/phase1_taichi_csr_bicgstab/csr_bicgstab_summary.json",
    "runs/phase1_taichi_csr_gmres/csr_gmres_results.jsonl",
    "runs/phase1_taichi_csr_gmres/csr_gmres_summary.json",
    "runs/phase1_taichi_csr_richardson/csr_richardson_results.jsonl",
    "runs/phase1_taichi_csr_richardson/csr_richardson_summary.json",
    "runs/phase1_taichi_csr_chebyshev/csr_chebyshev_results.jsonl",
    "runs/phase1_taichi_csr_chebyshev/csr_chebyshev_summary.json",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/datasets/diagnostics.py",
    "transsolvestack/policies/csr_selector_data.py",
    "scripts/tss_export_csr_selector_rows.py",
)


CORE_CSR_SELECTOR_POLICY_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/policies/csr_artifact_selector.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_csr_selector_policy_smoke.py",
)


CORE_CSR_AUTO_SOLVE_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/policies/csr_artifact_selector.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_csr_auto_solve_smoke.py",
)


CORE_CSR_LEARNING_READINESS_PROVENANCE_FILES = (
    "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/policies/csr_learning.py",
    "scripts/tss_csr_learning_readiness.py",
)


CORE_CSR_MODEL_CONTRACT_PROVENANCE_FILES = (
    "runs/phase1_csr_learning_readiness/csr_learning_rows.jsonl",
    "runs/phase1_csr_learning_readiness/csr_baseline_predictions.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/policies/csr_model_contract.py",
    "scripts/tss_csr_model_contract.py",
)


CORE_CSR_TRAINING_TENSOR_PROVENANCE_FILES = (
    "runs/phase1_csr_model_contract/csr_model_requests.jsonl",
    "runs/phase1_csr_model_contract/csr_model_targets.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/policies/csr_tensor_export.py",
    "scripts/tss_csr_training_tensors.py",
)


CORE_CSR_LINEAR_RANKER_PROVENANCE_FILES = (
    "runs/phase1_csr_training_tensors/csr_training_tensors.json",
    "runs/phase1_csr_training_tensors/csr_training_request_index.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/policies/csr_linear_ranker.py",
    "scripts/tss_csr_linear_ranker.py",
)


CORE_CSR_SELECTOR_MODEL_EVAL_PROVENANCE_FILES = (
    "runs/phase1_csr_learning_readiness/csr_learning_summary.json",
    "runs/phase1_csr_learning_readiness/csr_baseline_predictions.jsonl",
    "runs/phase1_csr_linear_ranker/csr_linear_ranker_summary.json",
    "runs/phase1_csr_linear_ranker/csr_linear_ranker_predictions.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/policies/csr_selector_model_eval.py",
    "scripts/tss_csr_selector_model_eval.py",
)


CORE_CSR_BENCHMARK_EXPANSION_PROVENANCE_FILES = (
    "configs/runtime/resource_limits.yaml",
    "runs/phase1_suitesparse_selection/selected_matrices.jsonl",
    "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    "transsolvestack/profiling/csr_benchmark_expansion.py",
    "scripts/tss_csr_benchmark_expansion_plan.py",
)


CORE_CSR_MICRO_CAMPAIGN_PROVENANCE_FILES = (
    "runs/phase1_csr_benchmark_expansion_plan/csr_benchmark_matrix_queue.jsonl",
    "runs/phase1_csr_benchmark_expansion_plan/csr_benchmark_candidate_queue.jsonl",
    "transsolvestack/profiling/csr_micro_campaign.py",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/datasets/diagnostics.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "transsolvestack/policies/csr_selector_data.py",
    "transsolvestack/profiling/repeats.py",
    "scripts/tss_csr_micro_campaign.py",
)


CORE_CSR_TRANSFORMER_READY_PROVENANCE_FILES = (
    "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    "runs/phase1_csr_micro_campaign/csr_micro_selector_rows.jsonl",
    "transsolvestack/policies/csr_transformer_ready.py",
    "transsolvestack/policies/csr_learning.py",
    "transsolvestack/policies/csr_model_contract.py",
    "transsolvestack/policies/csr_tensor_export.py",
    "scripts/tss_csr_transformer_ready.py",
)


CORE_CSR_TRANSFORMER_RANKER_PROVENANCE_FILES = (
    "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json",
    "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/policies/csr_transformer_ranker.py",
    "scripts/tss_csr_transformer_ranker.py",
)


CORE_CSR_TRANSFORMER_QUALITY_GATE_PROVENANCE_FILES = (
    "runs/phase1_csr_transformer_ready/combined_csr_learning_summary.json",
    "runs/phase1_csr_transformer_ready/combined_csr_baseline_predictions.jsonl",
    "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_summary.json",
    "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl",
    "transsolvestack/policies/csr_selector_model_eval.py",
    "scripts/tss_csr_transformer_quality_gate.py",
)


CORE_CSR_TRANSFORMER_TRAINING_ENTRYPOINT_PROVENANCE_FILES = (
    "runs/phase1_csr_transformer_ready/csr_transformer_ready_summary.json",
    "runs/phase1_csr_transformer_ready/csr_transformer_ready_schema.json",
    "runs/phase1_csr_transformer_ready/csr_transformer_tensor_summary.json",
    "runs/phase1_csr_transformer_ready/csr_transformer_model_contract_summary.json",
    "runs/phase1_csr_transformer_ready/csr_transformer_model_requests.jsonl",
    "runs/phase1_csr_transformer_ready/csr_transformer_model_targets.jsonl",
    "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json",
    "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl",
    "runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/policies/csr_transformer_training_entrypoint.py",
    "scripts/tss_csr_transformer_training_entrypoint.py",
)


CORE_CSR_LEARNED_GUARD_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl",
    "runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/policies/csr_artifact_selector.py",
    "transsolvestack/policies/csr_learned_guard.py",
    "transsolvestack/runtime/guarded_fallback.py",
    "scripts/tss_csr_learned_guard_smoke.py",
)


CORE_CSR_GUARDED_AUTO_SOLVE_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl",
    "runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/policies/csr_artifact_selector.py",
    "transsolvestack/policies/csr_learned_guard.py",
    "transsolvestack/runtime/guarded_fallback.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_csr_guarded_auto_solve_smoke.py",
)


CORE_CSR_GUARDED_PROMOTION_READINESS_PROVENANCE_FILES = (
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl",
    "runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/policies/csr_artifact_selector.py",
    "transsolvestack/policies/csr_learned_guard.py",
    "transsolvestack/runtime/guarded_fallback.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_csr_guarded_promotion_readiness.py",
)


CORE_CSR_GUARDED_PROMOTION_COVERAGE_PROVENANCE_FILES = (
    "runs/phase1_csr_transformer_ready/combined_csr_selector_rows.jsonl",
    "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl",
    "runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "transsolvestack/profiling/csr_guarded_promotion_coverage.py",
    "scripts/tss_csr_guarded_promotion_coverage_plan.py",
)


CORE_CSR_GUARDED_PROMOTION_COVERAGE_EXEC_PROVENANCE_FILES = (
    "runs/phase1_csr_guarded_promotion_coverage_plan/csr_guarded_promotion_coverage_scenarios.jsonl",
    "runs/phase1_csr_transformer_ready/combined_csr_selector_rows.jsonl",
    "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl",
    "runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/policies/csr_artifact_selector.py",
    "transsolvestack/policies/csr_learned_guard.py",
    "transsolvestack/runtime/guarded_fallback.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_csr_guarded_promotion_coverage_exec.py",
)


CORE_CSR_NON_SUCCESS_FALLBACK_PROBE_PROVENANCE_FILES = (
    "runs/phase1_csr_guarded_promotion_coverage_plan/csr_guarded_promotion_coverage_scenarios.jsonl",
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/policies/csr_selector_data.py",
    "transsolvestack/profiling/csr_micro_campaign.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_csr_non_success_fallback_probe.py",
)


CORE_CSR_GUARDED_NON_SUCCESS_FALLBACK_INTEGRATION_PROVENANCE_FILES = (
    "runs/phase1_csr_guarded_promotion_coverage_plan/csr_guarded_promotion_coverage_scenarios.jsonl",
    "runs/phase1_csr_transformer_ready/combined_csr_selector_rows.jsonl",
    "runs/phase1_csr_non_success_fallback_probe/csr_non_success_fallback_selector_rows.jsonl",
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/policies/csr_artifact_selector.py",
    "transsolvestack/policies/csr_learned_guard.py",
    "transsolvestack/runtime/guarded_fallback.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_csr_guarded_non_success_fallback_integration.py",
)


CORE_CSR_ILU0_GUARDED_INTEGRATION_PROVENANCE_FILES = (
    "runs/phase1_taichi_csr_ilu0/csr_ilu0_results.jsonl",
    "runs/phase1_csr_transformer_ready/combined_csr_selector_rows.jsonl",
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/policies/csr_artifact_selector.py",
    "transsolvestack/policies/csr_learned_guard.py",
    "transsolvestack/runtime/guarded_fallback.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_csr_ilu0_guarded_integration.py",
)


CORE_CSR_ILU0_COVERAGE_EXPANSION_PROVENANCE_FILES = (
    "runs/phase1_taichi_csr_ilu0/csr_ilu0_results.jsonl",
    "runs/phase1_csr_transformer_ready/combined_csr_selector_rows.jsonl",
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/operators/taichi_csr.py",
    "transsolvestack/operators/taichi_diffusion2d.py",
    "transsolvestack/runtime/engine.py",
    "scripts/tss_taichi_csr_ilu0_smoke.py",
    "scripts/tss_csr_ilu0_coverage_expansion.py",
)


CORE_CSR_UNRESOLVED_FALLBACK_COVERAGE_PLAN_PROVENANCE_FILES = (
    "runs/phase1_csr_non_success_fallback_probe/csr_non_success_fallback_probe_results.jsonl",
    "runs/phase1_csr_guarded_non_success_fallback_integration/csr_guarded_non_success_fallback_integration_results.jsonl",
    "runs/phase1_csr_guarded_non_success_fallback_integration/csr_guarded_non_success_fallback_integration_summary.json",
    "scripts/tss_csr_unresolved_fallback_coverage_plan.py",
)


CORE_CSR_UNRESOLVED_FALLBACK_CPU_SCREEN_PROVENANCE_FILES = (
    "runs/phase1_csr_unresolved_fallback_coverage_plan/csr_unresolved_fallback_candidate_plan.jsonl",
    "runs/phase1_csr_unresolved_fallback_coverage_plan/csr_unresolved_fallback_coverage_plan_summary.json",
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/profiling/csr_micro_campaign.py",
    "scripts/tss_csr_unresolved_fallback_cpu_screen.py",
)


CORE_CSR_UNRESOLVED_MATRIX_DIAGNOSTICS_PROVENANCE_FILES = (
    "runs/phase1_csr_unresolved_fallback_cpu_screen/csr_unresolved_fallback_cpu_screen_results.jsonl",
    "runs/phase1_csr_unresolved_fallback_cpu_screen/csr_unresolved_fallback_cpu_screen_summary.json",
    "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
    "transsolvestack/datasets/csr.py",
    "transsolvestack/datasets/csr_diagnostics.py",
    "scripts/tss_csr_unresolved_matrix_diagnostics.py",
)


CORE_PUBLIC_RELEASE_HYGIENE_PROVENANCE_FILES = (
    ".gitignore",
    "LICENSE",
    "README.md",
    "docs/PROJECT_OVERVIEW.md",
    "LICENSE_POLICY.md",
    "COMMERCIAL_USE.md",
    "MODEL_CONTRIBUTION_TERMS.md",
    "CONTRIBUTOR_LICENSE_AGREEMENT.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "transsolvestack/profiling/public_release_hygiene.py",
    "scripts/tss_public_release_hygiene.py",
)


CORE_CAMPAIGN_PROVENANCE_FILES = (
    "runs/phase1_smoke_taichi/artifact_manifest.json",
    "runs/phase1_smoke_expanded_solvers/artifact_manifest.json",
    "runs/phase1_numerical_regression/artifact_manifest.json",
    "runs/phase1_sequence_taichi/artifact_manifest.json",
    "runs/phase1_policy_selection/artifact_manifest.json",
    "runs/phase1_policy_solve/artifact_manifest.json",
    "runs/phase1_dataset_plan/artifact_manifest.json",
    "runs/phase1_matrix_market_probe/artifact_manifest.json",
    "runs/phase1_suitesparse_selection/artifact_manifest.json",
    "runs/phase1_suitesparse_header_probe/artifact_manifest.json",
    "runs/phase1_suitesparse_csr_import/artifact_manifest.json",
    "runs/phase1_taichi_csr_matvec/artifact_manifest.json",
    "runs/phase1_taichi_csr_primitives/artifact_manifest.json",
    "runs/phase1_taichi_csr_solve/artifact_manifest.json",
    "runs/phase1_taichi_csr_bicgstab/artifact_manifest.json",
    "runs/phase1_taichi_csr_gmres/artifact_manifest.json",
    "runs/phase1_taichi_csr_richardson/artifact_manifest.json",
    "runs/phase1_taichi_csr_chebyshev/artifact_manifest.json",
    "runs/phase1_taichi_csr_symmetric_equilibration/artifact_manifest.json",
    "runs/phase1_taichi_csr_row_column_equilibration/artifact_manifest.json",
    "runs/phase1_taichi_csr_ilu0/artifact_manifest.json",
    "runs/phase1_csr_selector_readiness/artifact_manifest.json",
    "runs/phase1_csr_selector_policy/artifact_manifest.json",
    "runs/phase1_csr_auto_solve/artifact_manifest.json",
    "runs/phase1_csr_learning_readiness/artifact_manifest.json",
    "runs/phase1_csr_model_contract/artifact_manifest.json",
    "runs/phase1_csr_training_tensors/artifact_manifest.json",
    "runs/phase1_csr_linear_ranker/artifact_manifest.json",
    "runs/phase1_csr_selector_model_eval/artifact_manifest.json",
    "runs/phase1_csr_benchmark_expansion_plan/artifact_manifest.json",
    "runs/phase1_csr_micro_campaign/artifact_manifest.json",
    "runs/phase1_csr_transformer_ready/artifact_manifest.json",
    "runs/phase1_csr_transformer_ranker/artifact_manifest.json",
    "runs/phase1_csr_transformer_quality_gate/artifact_manifest.json",
    "runs/phase1_csr_transformer_training_entrypoint/artifact_manifest.json",
    "runs/phase1_csr_learned_guard/artifact_manifest.json",
    "runs/phase1_csr_guarded_auto_solve/artifact_manifest.json",
    "runs/phase1_csr_guarded_promotion_readiness/artifact_manifest.json",
    "runs/phase1_csr_guarded_promotion_coverage_plan/artifact_manifest.json",
    "runs/phase1_csr_guarded_promotion_coverage_exec/artifact_manifest.json",
    "runs/phase1_csr_non_success_fallback_probe/artifact_manifest.json",
    "runs/phase1_csr_guarded_non_success_fallback_integration/artifact_manifest.json",
    "runs/phase1_csr_ilu0_guarded_integration/artifact_manifest.json",
    "runs/phase1_csr_ilu0_coverage_expansion/artifact_manifest.json",
    "runs/phase1_csr_unresolved_fallback_coverage_plan/artifact_manifest.json",
    "runs/phase1_csr_unresolved_fallback_cpu_screen/artifact_manifest.json",
    "runs/phase1_csr_unresolved_matrix_diagnostics/artifact_manifest.json",
    "runs/phase1_public_release_hygiene/artifact_manifest.json",
    "runs/phase1_transformer_readiness/artifact_manifest.json",
    "runs/phase1_public_api_smoke/artifact_manifest.json",
    "runs/phase1_csr_public_api/artifact_manifest.json",
    "runs/phase1_solver_functional/artifact_manifest.json",
    "transsolvestack/profiling/campaign.py",
    "scripts/tss_phase1_campaign.py",
)


CORE_TRANSFORMER_READINESS_PROVENANCE_FILES = (
    "configs/workloads/phase1_smoke.yaml",
    "runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
    "transsolvestack/benchmarks/expand.py",
    "transsolvestack/policies/training_data.py",
    "transsolvestack/policies/transformer_contract.py",
    "scripts/tss_export_policy_training_rows.py",
)


CORE_PUBLIC_API_SMOKE_PROVENANCE_FILES = (
    "configs/candidates/phase1_taichi_gpu.yaml",
    "configs/datasets/phase1_external_small.yaml",
    "configs/workloads/phase1_smoke.yaml",
    "runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
    "transsolvestack/__init__.py",
    "transsolvestack/api.py",
    "transsolvestack/operators/synthetic.py",
    "transsolvestack/operators/taichi_diffusion2d.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/backends/registry.py",
    "scripts/tss_public_api_smoke.py",
)


CORE_SOLVER_FUNCTIONAL_PROVENANCE_FILES = (
    "configs/workloads/phase1_solver_functional.yaml",
    "configs/candidates/phase1_solver_expansion.yaml",
    "configs/runtime/resource_limits.yaml",
    "transsolvestack/backends/registry.py",
    "transsolvestack/operators/synthetic.py",
    "transsolvestack/operators/taichi_diffusion2d.py",
    "transsolvestack/solvers/registry.py",
    "transsolvestack/solvers/taichi_cg.py",
    "transsolvestack/runtime/engine.py",
    "transsolvestack/runtime/plan_resolver.py",
    "transsolvestack/profiling/solver_functional.py",
    "scripts/tss_solver_functional_smoke.py",
)
