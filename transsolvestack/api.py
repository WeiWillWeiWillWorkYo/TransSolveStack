"""Public API entry points for the Phase 1 framework."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

from transsolvestack.benchmarks.candidates import CandidateSet, load_candidate_set
from transsolvestack.benchmarks.config import WorkloadConfig, load_workload_config
from transsolvestack.core.result import PolicyPlan
from transsolvestack.core.types import LinearSystem, SolveContext
from transsolvestack.datasets.catalog import (
    DownloadPlan,
    build_download_plan,
    load_matrix_catalog,
)
from transsolvestack.datasets.suitesparse_index import (
    SuiteSparseIndex,
    build_suitesparse_index as build_suitesparse_index_record,
)
from transsolvestack.datasets.selection import (
    MatrixSelection,
    MatrixSelectionCriteria,
    select_matrix_subset_from_index,
)
from transsolvestack.datasets.archive_probe import (
    ArchiveHeaderProbe,
    probe_selected_matrix_archives_from_file,
)
from transsolvestack.datasets.csr import (
    CsrMatrix,
    CsrImportBatch,
    csr_matrix_from_record,
    import_selected_archives_to_csr_from_files,
)
from transsolvestack.datasets.diagnostics import (
    CsrMatrixDiagnostic,
    diagnose_csr_matrix as diagnose_csr_matrix_record,
)
from transsolvestack.policies.heuristic import TaichiFirstHeuristicPlanner
from transsolvestack.policies.artifact_selector import BenchmarkArtifactPolicySelector
from transsolvestack.policies.training_data import (
    PolicyTrainingRow,
    build_policy_training_rows,
)
from transsolvestack.policies.csr_selector_data import (
    CsrSelectorExport,
    build_csr_selector_export_from_files,
)
from transsolvestack.policies.csr_learning import (
    CsrLearningExport,
    build_csr_learning_export_from_selector_rows,
)
from transsolvestack.policies.csr_model_contract import (
    CsrSelectorModelContractExport,
    build_csr_model_contract_export,
)
from transsolvestack.policies.csr_tensor_export import (
    CsrTensorExport,
    build_csr_tensor_export,
)
from transsolvestack.policies.csr_linear_ranker import (
    CsrLinearRankerExport,
    train_csr_linear_ranker_from_tensor_file,
)
from transsolvestack.policies.csr_selector_model_eval import (
    CsrSelectorModelEvaluationExport,
    evaluate_csr_selector_models_from_files,
)
from transsolvestack.policies.csr_transformer_ready import (
    build_csr_transformer_ready_bundle_from_files,
)
from transsolvestack.policies.csr_transformer_ranker import (
    CsrTransformerRankerExport,
    train_csr_transformer_ranker_from_tensor_file,
)
from transsolvestack.policies.csr_external_model_adapter import (
    adapt_csr_external_ranker_checkpoint_from_files,
    build_reference_csr_external_ranker_checkpoint_from_files,
)
from transsolvestack.policies.csr_external_model_intake import (
    run_csr_external_model_intake_from_files,
)
from transsolvestack.policies.csr_policy_model_artifact import (
    build_csr_policy_model_artifact_from_files,
)
from transsolvestack.policies.csr_policy_model_acceptance import (
    build_csr_policy_model_acceptance_from_files,
)
from transsolvestack.policies.csr_policy_model_submission import (
    build_csr_policy_model_submission_from_files,
)
from transsolvestack.policies.csr_transformer_training_entrypoint import (
    build_csr_transformer_training_entrypoint_from_files,
)
from transsolvestack.policies.csr_transformer_reference_training_export import (
    build_csr_transformer_reference_training_export_from_files,
)
from transsolvestack.profiling.csr_benchmark_expansion import (
    CsrBenchmarkExpansionPlan,
    build_csr_benchmark_expansion_plan_from_files,
)
from transsolvestack.profiling.csr_full_dataset_queue import (
    CsrFullDatasetQueue,
    build_csr_full_dataset_queue_from_files,
)
from transsolvestack.policies.csr_artifact_selector import (
    CsrCandidateSelection,
    CsrArtifactPolicySelector,
)
from transsolvestack.policies.csr_learned_guard import (
    CsrLearnedGuardDecision,
    plan_csr_with_learned_guard as plan_csr_with_learned_guard_record,
)
from transsolvestack.runtime.config import load_resource_budget
from transsolvestack.runtime.engine import TaichiExecutionEngine
from transsolvestack.runtime.guarded_fallback import (
    RuntimeGuardConfig,
    execute_with_fallback_guard,
)


def plan(
    system: LinearSystem,
    context: SolveContext,
    policy_mode: str = "heuristic",
    candidate_set: CandidateSet | str | Path | None = None,
    evaluation_path: str | Path | None = None,
    fallback_candidate_id: str | None = None,
) -> PolicyPlan:
    """Build an auditable PolicyPlan without running a solve."""

    if policy_mode == "heuristic":
        return TaichiFirstHeuristicPlanner().plan(system, context)
    if policy_mode == "benchmark_artifact":
        if candidate_set is None:
            raise ValueError("benchmark_artifact policy requires candidate_set")
        if evaluation_path is None:
            raise ValueError("benchmark_artifact policy requires evaluation_path")
        loaded_candidate_set = _load_candidate_set(candidate_set)
        return BenchmarkArtifactPolicySelector.from_evaluation_artifact(
            candidate_set=loaded_candidate_set,
            evaluation_path=evaluation_path,
            fallback_candidate_id=fallback_candidate_id,
        ).select(system.system_id, context.context_id).plan
    raise ValueError(f"unsupported policy_mode: {policy_mode}")


def solve(
    system: LinearSystem,
    rhs: Any,
    context: SolveContext,
    policy_plan: PolicyPlan | None = None,
    *,
    device_memory_gb: float = 0.5,
) -> Any:
    """Execute a supported Taichi GPU solve.

    Phase 1 supports synthetic structured-stencil systems with the internal
    smoke-test RHS. General external RHS vectors are intentionally not accepted
    until the sparse matrix import path is wired into the GPU runtime.
    """

    if rhs is not None:
        raise NotImplementedError("public solve currently supports rhs=None only")
    active_plan = policy_plan or plan(system, context)
    if active_plan.backend != "taichi_gpu":
        raise ValueError(f"public solve requires taichi_gpu, got {active_plan.backend}")
    from transsolvestack.operators.taichi_diffusion2d import (
        diffusion_operator_from_system,
        ensure_taichi_cuda,
    )

    ensure_taichi_cuda(device_memory_gb=device_memory_gb)
    operator = diffusion_operator_from_system(system)
    return TaichiExecutionEngine().solve(
        operator=operator,
        rhs=None,
        context=context,
        plan=active_plan,
    )


def diagnose_csr_matrix(csr: CsrMatrix | dict[str, Any]) -> CsrMatrixDiagnostic:
    """Diagnose a CSR matrix for solver-selection and dispatch decisions."""

    return diagnose_csr_matrix_record(_coerce_csr_matrix(csr))


def solve_csr(
    csr: CsrMatrix | dict[str, Any],
    rhs: Any,
    *,
    context: SolveContext | None = None,
    solver: str = "cg",
    preconditioner: str | None = None,
    precision: str | None = None,
    solver_parameters: dict[str, Any] | None = None,
    policy_plan: PolicyPlan | None = None,
    device_memory_gb: float = 0.25,
) -> Any:
    """Execute a Taichi GPU solve on a CPU-side CSR fixture.

    This is the public assembled-sparse entry point. It accepts bounded CSR
    fixtures imported by `import_suitesparse_csr` and transfers them into a
    Taichi device-resident CSR operator before solving.
    """

    matrix = _coerce_csr_matrix(csr)
    if rhs is None:
        raise ValueError("solve_csr requires an explicit rhs vector")
    vector = tuple(float(value) for value in rhs)
    if len(vector) != matrix.n_rows:
        raise ValueError(f"solve_csr expected rhs length {matrix.n_rows}, got {len(vector)}")
    active_preconditioner = preconditioner or (
        "jacobi" if solver in {"pcg", "gmres", "richardson", "chebyshev"} else "none"
    )
    active_context = _csr_context(context, precision=precision)
    active_plan = policy_plan or PolicyPlan(
        plan_id=f"public_csr_{solver}:{matrix.matrix_id}",
        backend="taichi_gpu",
        solver={"name": solver, **(solver_parameters or {})},
        preconditioner={"name": active_preconditioner},
        audit={"policy": "public_csr_solve"},
    )
    if active_plan.backend != "taichi_gpu":
        raise ValueError(f"solve_csr requires taichi_gpu, got {active_plan.backend}")
    if solver in {"cg", "pcg"}:
        diagnostic = diagnose_csr_matrix_record(matrix)
        if not diagnostic.is_square:
            raise ValueError(f"solve_csr {solver} requires a square matrix")
        if not diagnostic.actual_symmetric:
            raise ValueError(f"solve_csr {solver} requires an actually symmetric matrix")
    from transsolvestack.operators.taichi_csr import TaichiCsrMatrixOperator
    from transsolvestack.operators.taichi_diffusion2d import ensure_taichi_cuda

    ensure_taichi_cuda(device_memory_gb=device_memory_gb)
    operator = TaichiCsrMatrixOperator.from_csr_matrix(
        matrix,
        dtype=active_context.precision,
    )
    return TaichiExecutionEngine().solve(
        operator=operator,
        rhs=vector,
        context=active_context,
        plan=active_plan,
    )


def plan_csr(
    csr: CsrMatrix | dict[str, Any],
    *,
    selector_path: str | Path = "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    context: SolveContext | None = None,
    objective: str = "min_solve_time_success",
    fallback_candidate_id: str | None = None,
) -> CsrCandidateSelection:
    """Select an auditable CSR PolicyPlan from real profiling artifacts."""

    matrix = _coerce_csr_matrix(csr)
    active_context = _csr_selector_context(context)
    return CsrArtifactPolicySelector.from_selector_rows(
        selector_path,
        fallback_candidate_id=fallback_candidate_id,
    ).select(
        matrix,
        context_id=active_context.context_id,
        objective=objective,
    )


def auto_solve_csr(
    csr: CsrMatrix | dict[str, Any],
    rhs: Any,
    *,
    selector_path: str | Path = "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    context: SolveContext | None = None,
    objective: str = "min_solve_time_success",
    fallback_candidate_id: str | None = None,
    device_memory_gb: float = 0.25,
) -> Any:
    """Select and execute an auditable Taichi GPU CSR solve."""

    matrix = _coerce_csr_matrix(csr)
    active_context = _csr_selector_context(context)
    selection = plan_csr(
        matrix,
        selector_path=selector_path,
        context=active_context,
        objective=objective,
        fallback_candidate_id=fallback_candidate_id,
    )
    solver_name = str(selection.plan.solver["name"])
    preconditioner_name = selection.plan.preconditioner.get("name")
    result = solve_csr(
        matrix,
        rhs,
        context=active_context,
        solver=solver_name,
        preconditioner=preconditioner_name,
        precision=selection.plan.solver.get("precision", active_context.precision),
        policy_plan=selection.plan,
        device_memory_gb=device_memory_gb,
    )
    selection_metadata = {
        "selector": "csr_artifact",
        "selector_path": str(selector_path),
        "objective": objective,
        "candidate_id": selection.candidate_id,
        "reason": selection.reason,
        "fallback_candidate_ids": list(selection.fallback_candidate_ids),
        "plan": asdict(selection.plan),
        "profile": (
            asdict(selection.profile) if selection.profile is not None else None
        ),
    }
    result.trace.metadata["auto_solve_csr"] = selection_metadata
    result.metadata["auto_solve_csr"] = selection_metadata
    return result


def plan_csr_with_learned_guard(
    csr: CsrMatrix | dict[str, Any],
    *,
    selector_path: str | Path = "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    learned_predictions_path: str | Path | None = (
        "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl"
    ),
    learned_model_artifact_path: str | Path | None = None,
    learned_model_path: str | Path | None = None,
    learned_tensor_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json"
    ),
    learned_request_index_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl"
    ),
    quality_gate_summary_path: str | Path = "runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    context: SolveContext | None = None,
    objective: str = "min_solve_time_success",
    fallback_candidate_id: str | None = None,
    mode: str = "shadow",
    min_confidence: float = 0.75,
) -> CsrLearnedGuardDecision:
    """Plan CSR execution with learned-policy shadowing and runtime guards."""

    matrix = _coerce_csr_matrix(csr)
    active_context = _csr_selector_context(context)
    return plan_csr_with_learned_guard_record(
        matrix,
        selector_path=selector_path,
        learned_predictions_path=learned_predictions_path,
        learned_model_artifact_path=learned_model_artifact_path,
        learned_model_path=learned_model_path,
        learned_tensor_path=learned_tensor_path,
        learned_request_index_path=learned_request_index_path,
        quality_gate_summary_path=quality_gate_summary_path,
        context_id=active_context.context_id,
        objective=objective,
        fallback_candidate_id=fallback_candidate_id,
        mode=mode,
        min_confidence=min_confidence,
    )


def auto_solve_csr_guarded(
    csr: CsrMatrix | dict[str, Any],
    rhs: Any,
    *,
    selector_path: str | Path = "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    learned_predictions_path: str | Path | None = (
        "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl"
    ),
    learned_model_artifact_path: str | Path | None = None,
    learned_model_path: str | Path | None = None,
    learned_tensor_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json"
    ),
    learned_request_index_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl"
    ),
    quality_gate_summary_path: str | Path = "runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    context: SolveContext | None = None,
    objective: str = "min_solve_time_success",
    fallback_candidate_id: str | None = None,
    learned_policy_mode: str = "shadow",
    min_learned_confidence: float = 0.75,
    max_attempt_wall_time_ms: float | None = None,
    max_total_wall_time_ms: float | None = None,
    device_memory_gb: float = 0.25,
) -> Any:
    """Select, guard, and execute CSR solve with learned-policy shadowing."""

    matrix = _coerce_csr_matrix(csr)
    active_context = _csr_selector_context(context)
    decision = plan_csr_with_learned_guard(
        matrix,
        selector_path=selector_path,
        learned_predictions_path=learned_predictions_path,
        learned_model_artifact_path=learned_model_artifact_path,
        learned_model_path=learned_model_path,
        learned_tensor_path=learned_tensor_path,
        learned_request_index_path=learned_request_index_path,
        quality_gate_summary_path=quality_gate_summary_path,
        context=active_context,
        objective=objective,
        fallback_candidate_id=fallback_candidate_id,
        mode=learned_policy_mode,
        min_confidence=min_learned_confidence,
    )
    plans = (decision.selection.plan, *_fallback_plans(decision.selection.plan))

    def execute(plan: PolicyPlan) -> Any:
        return solve_csr(
            matrix,
            rhs,
            context=active_context,
            solver=str(plan.solver["name"]),
            preconditioner=plan.preconditioner.get("name"),
            precision=plan.solver.get("precision", active_context.precision),
            solver_parameters={
                key: value
                for key, value in plan.solver.items()
                if key not in {"name", "precision"}
            },
            policy_plan=plan,
            device_memory_gb=device_memory_gb,
        )

    outcome = execute_with_fallback_guard(
        plans,
        execute,
        config=RuntimeGuardConfig(
            max_attempt_wall_time_ms=max_attempt_wall_time_ms,
            max_total_wall_time_ms=max_total_wall_time_ms,
        ),
    )
    guard_metadata = {
        "schema_version": decision.schema_version,
        "guard_id": decision.guard_id,
        "mode": decision.mode,
        "guard_status": decision.guard_status,
        "guard_reasons": list(decision.guard_reasons),
        "runtime_selector_changed": decision.runtime_selector_changed,
        "runtime_candidate_id": decision.runtime_candidate_id,
        "runtime_selection_source": decision.runtime_selection_source,
        "artifact_candidate_id": decision.artifact_candidate_id,
        "fallback_candidate_ids": list(decision.fallback_candidate_ids),
        "fallback_chain_enforced": decision.fallback_chain_enforced,
        "min_confidence": decision.min_confidence,
        "learned_policy_source": asdict(decision.learned_policy_source),
        "learned_prediction": (
            None
            if decision.learned_prediction is None
            else asdict(decision.learned_prediction)
        ),
        "quality_gate_summary": dict(decision.quality_gate_summary),
    }
    outcome.result.trace.metadata["learned_policy_guard"] = guard_metadata
    outcome.result.metadata["learned_policy_guard"] = guard_metadata
    return outcome.result


def auto_solve(
    system: LinearSystem,
    rhs: Any,
    context: SolveContext,
    *,
    policy_mode: str = "heuristic",
    candidate_set: CandidateSet | str | Path | None = None,
    evaluation_path: str | Path | None = None,
    fallback_candidate_id: str | None = None,
    device_memory_gb: float = 0.5,
) -> Any:
    """Plan and execute a supported Taichi GPU solve."""

    active_plan = plan(
        system=system,
        context=context,
        policy_mode=policy_mode,
        candidate_set=candidate_set,
        evaluation_path=evaluation_path,
        fallback_candidate_id=fallback_candidate_id,
    )
    return solve(
        system=system,
        rhs=rhs,
        context=context,
        policy_plan=active_plan,
        device_memory_gb=device_memory_gb,
    )


def plan_dataset_downloads(
    catalog: str | Path,
    *,
    data_root: str | Path | None = None,
) -> DownloadPlan:
    """Build a resource-checked external matrix download plan without fetching."""

    loaded_catalog = load_matrix_catalog(catalog)
    if loaded_catalog.resource_limits_path is None:
        raise ValueError("matrix catalog does not define resource_limits")
    budget = load_resource_budget(loaded_catalog.resource_limits_path)
    return build_download_plan(loaded_catalog, budget, data_root=data_root)


def build_suitesparse_index(
    stats_path: str | Path,
    urls_path: str | Path,
    mm_root: str | Path,
) -> SuiteSparseIndex:
    """Build a metadata-only SuiteSparse archive index."""

    return build_suitesparse_index_record(
        stats_path=stats_path,
        urls_path=urls_path,
        mm_root=mm_root,
    )


def select_suitesparse_subset(
    index_path: str | Path,
    criteria: MatrixSelectionCriteria | None = None,
) -> MatrixSelection:
    """Select a deterministic metadata-only real-matrix subset."""

    return select_matrix_subset_from_index(index_path, criteria=criteria)


def probe_suitesparse_headers(selection_path: str | Path) -> ArchiveHeaderProbe:
    """Probe Matrix Market headers inside selected SuiteSparse archives."""

    return probe_selected_matrix_archives_from_file(selection_path)


def import_suitesparse_csr(
    selection_path: str | Path,
    *,
    header_probe_path: str | Path | None = None,
    max_matrices: int = 12,
    max_rows: int = 10_000,
    max_cols: int = 10_000,
    max_stored_entries: int = 100_000,
    max_archive_size_bytes: int = 8_000_000,
) -> CsrImportBatch:
    """Import a bounded selected SuiteSparse subset into CPU-side CSR fixtures."""

    return import_selected_archives_to_csr_from_files(
        selection_path,
        header_probe_path=header_probe_path,
        max_matrices=max_matrices,
        max_rows=max_rows,
        max_cols=max_cols,
        max_stored_entries=max_stored_entries,
        max_archive_size_bytes=max_archive_size_bytes,
    )


def export_policy_training_rows(
    workload: WorkloadConfig | str | Path,
    evaluation_path: str | Path,
) -> tuple[PolicyTrainingRow, ...]:
    """Build policy ranking rows for a future learned/Transformer selector."""

    loaded_workload = (
        load_workload_config(workload)
        if isinstance(workload, str | Path)
        else workload
    )
    return build_policy_training_rows(loaded_workload, evaluation_path)


def export_csr_selector_rows(
    csr_path: str | Path,
    solve_results_path: str | Path | Iterable[str | Path],
    *,
    screened_results_path: str | Path | Iterable[str | Path] | None = None,
    context_id: str = "phase1_csr_selector",
) -> CsrSelectorExport:
    """Build real-CSR selector diagnostics and ranking rows from artifacts."""

    return build_csr_selector_export_from_files(
        csr_path,
        solve_results_path,
        screened_results_path=screened_results_path,
        context_id=context_id,
    )


def export_csr_learning_rows(
    selector_rows_path: str | Path,
    *,
    eval_fraction: float = 0.25,
    min_eval_matrices: int = 2,
) -> CsrLearningExport:
    """Build CSR selector learning rows and a baseline split/evaluation."""

    return build_csr_learning_export_from_selector_rows(
        selector_rows_path,
        eval_fraction=eval_fraction,
        min_eval_matrices=min_eval_matrices,
    )


def export_csr_model_contract(
    learning_rows_path: str | Path,
    baseline_predictions_path: str | Path,
) -> CsrSelectorModelContractExport:
    """Build label-free CSR model requests, offline targets, and prediction checks."""

    return build_csr_model_contract_export(
        learning_rows_path,
        baseline_predictions_path,
    )


def export_csr_training_tensors(
    requests_path: str | Path,
    targets_path: str | Path,
) -> CsrTensorExport:
    """Build numeric CSR selector arrays from model contract requests/targets."""

    return build_csr_tensor_export(requests_path, targets_path)


def train_csr_linear_ranker(
    tensor_path: str | Path,
    request_index_path: str | Path,
    *,
    epochs: int = 80,
    learning_rate: float = 0.05,
    l2_regularization: float = 1.0e-4,
) -> CsrLinearRankerExport:
    """Train and evaluate an offline CSR pairwise linear ranker baseline."""

    return train_csr_linear_ranker_from_tensor_file(
        tensor_path,
        request_index_path,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
    )


def train_csr_transformer_ranker(
    tensor_path: str | Path = "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json",
    request_index_path: str | Path = "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl",
    *,
    epochs: int = 160,
    learning_rate: float = 0.03,
    l2_regularization: float = 1.0e-4,
    d_model: int = 24,
    num_attention_heads: int = 4,
    feedforward_dim: int = 48,
    seed: int = 18,
) -> CsrTransformerRankerExport:
    """Train the offline CSR masked self-attention ranker artifact."""

    return train_csr_transformer_ranker_from_tensor_file(
        tensor_path,
        request_index_path,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        feedforward_dim=feedforward_dim,
        seed=seed,
    )


def build_reference_csr_external_ranker_checkpoint(
    model_path: str | Path = "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_model.json",
    tensor_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json"
    ),
    request_index_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl"
    ),
    *,
    contributor_id: str = "wei_cui_reference",
    contributor_name: str = "Wei CUI",
    training_statement: str = (
        "Reference checkpoint exported from the Phase 1 CSR Transformer ranker."
    ),
) -> dict[str, Any]:
    """Build a reference external-checkpoint fixture from a TSS saved ranker."""

    return build_reference_csr_external_ranker_checkpoint_from_files(
        model_path,
        tensor_path,
        request_index_path,
        contributor_id=contributor_id,
        contributor_name=contributor_name,
        training_statement=training_statement,
    )


def adapt_csr_external_ranker_checkpoint(
    checkpoint_path: str | Path,
    tensor_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json"
    ),
    request_index_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl"
    ),
) -> dict[str, Any]:
    """Adapt an external CSR ranker checkpoint into TSS model artifacts."""

    return adapt_csr_external_ranker_checkpoint_from_files(
        checkpoint_path,
        tensor_path,
        request_index_path,
    )


def run_csr_external_model_intake(
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path = "runs/phase1_csr_external_model_intake",
    *,
    source_model_path: str | Path = (
        "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_model.json"
    ),
    tensor_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json"
    ),
    request_index_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl"
    ),
    baseline_summary_path: str | Path = (
        "runs/phase1_csr_transformer_ready/combined_csr_learning_summary.json"
    ),
    baseline_predictions_path: str | Path = (
        "runs/phase1_csr_transformer_ready/combined_csr_baseline_predictions.jsonl"
    ),
    csr_path: str | Path = "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    selector_path: str | Path = "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    guarded_auto_solve_summary_path: str | Path = (
        "runs/phase1_csr_guarded_auto_solve/csr_guarded_auto_solve_summary.json"
    ),
    guarded_auto_solve_results_path: str | Path = (
        "runs/phase1_csr_guarded_auto_solve/csr_guarded_auto_solve_results.jsonl"
    ),
    contributor_id: str = "wei_cui_reference",
    contributor_name: str = "Wei CUI",
    contribution_name: str = "csr_external_ranker_reference_intake",
    contribution_version: str = "phase1-reference",
) -> dict[str, Any]:
    """Run adapter, replay, quality gate, acceptance, and submission in one pass."""

    return run_csr_external_model_intake_from_files(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        source_model_path=source_model_path,
        tensor_path=tensor_path,
        request_index_path=request_index_path,
        baseline_summary_path=baseline_summary_path,
        baseline_predictions_path=baseline_predictions_path,
        csr_path=csr_path,
        selector_path=selector_path,
        guarded_auto_solve_summary_path=guarded_auto_solve_summary_path,
        guarded_auto_solve_results_path=guarded_auto_solve_results_path,
        contributor_id=contributor_id,
        contributor_name=contributor_name,
        contribution_name=contribution_name,
        contribution_version=contribution_version,
    )


def evaluate_csr_selector_models(
    baseline_summary_path: str | Path,
    baseline_predictions_path: str | Path,
    ranker_summary_path: str | Path,
    ranker_predictions_path: str | Path,
    *,
    min_eval_oracle_requests: int = 10,
    min_runtime_oracle_top1_accuracy: float = 0.5,
    min_runtime_profiled_success_rate: float = 2.0 / 3.0,
) -> CsrSelectorModelEvaluationExport:
    """Evaluate offline CSR selector models before any runtime integration."""

    return evaluate_csr_selector_models_from_files(
        baseline_summary_path,
        baseline_predictions_path,
        ranker_summary_path,
        ranker_predictions_path,
        min_eval_oracle_requests=min_eval_oracle_requests,
        min_runtime_oracle_top1_accuracy=min_runtime_oracle_top1_accuracy,
        min_runtime_profiled_success_rate=min_runtime_profiled_success_rate,
    )


def plan_csr_benchmark_expansion(
    selection_path: str | Path = "runs/phase1_suitesparse_selection/selected_matrices.jsonl",
    selector_rows_path: str | Path = "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    resource_limits_path: str | Path = "configs/runtime/resource_limits.yaml",
    *,
    max_new_matrices: int = 8,
    max_planned_gpu_solves: int = 24,
    measurement_repeats: int = 1,
    max_rows: int = 10_000,
    max_cols: int = 10_000,
    max_nnz: int = 100_000,
    max_archive_size_bytes: int = 8_000_000,
    max_iter: int = 256,
    tolerance_rel: float = 1.0e-5,
) -> CsrBenchmarkExpansionPlan:
    """Plan a bounded real-CSR benchmark expansion without launching GPU work."""

    return build_csr_benchmark_expansion_plan_from_files(
        selection_path,
        selector_rows_path,
        resource_limits_path,
        max_new_matrices=max_new_matrices,
        max_planned_gpu_solves=max_planned_gpu_solves,
        measurement_repeats=measurement_repeats,
        max_rows=max_rows,
        max_cols=max_cols,
        max_nnz=max_nnz,
        max_archive_size_bytes=max_archive_size_bytes,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
    )


def plan_csr_full_dataset_queue(
    index_path: str | Path = (
        "/mnt/tss_external/TransSolveStack/datasets/"
        "suitesparse_full/index/matrix_manifest.jsonl"
    ),
    selector_rows_path: str | Path = (
        "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl"
    ),
    resource_limits_path: str | Path = "configs/runtime/resource_limits.yaml",
    storage_paths_path: str | Path = "configs/runtime/storage_paths.yaml",
    *,
    queue_id: str = "phase1_csr_full_dataset_queue_m83",
    batch_matrix_count: int = 8,
    measurement_repeats: int = 1,
    max_rows: int | None = None,
    max_cols: int | None = None,
    max_nnz: int | None = None,
    max_archive_size_bytes: int = 256_000_000,
    max_iter: int | None = None,
    tolerance_rel: float = 1.0e-5,
    max_queue_matrices: int | None = None,
) -> CsrFullDatasetQueue:
    """Plan a resumable full SuiteSparse CSR benchmark/training queue."""

    return build_csr_full_dataset_queue_from_files(
        index_path,
        selector_rows_path,
        resource_limits_path,
        storage_paths_path,
        queue_id=queue_id,
        batch_matrix_count=batch_matrix_count,
        measurement_repeats=measurement_repeats,
        max_rows=max_rows,
        max_cols=max_cols,
        max_nnz=max_nnz,
        max_archive_size_bytes=max_archive_size_bytes,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
        max_queue_matrices=max_queue_matrices,
    )


def build_csr_transformer_ready_bundle(
    selector_paths: Iterable[str | Path] = (
        "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
        "runs/phase1_csr_micro_campaign/csr_micro_selector_rows.jsonl",
    ),
    output_dir: str | Path = "runs/phase1_csr_transformer_ready",
    *,
    eval_fraction: float = 0.25,
    min_eval_matrices: int = 4,
) -> dict[str, Any]:
    """Build label-free CSR requests, offline targets, and tensors for a ranker."""

    return build_csr_transformer_ready_bundle_from_files(
        selector_paths,
        output_dir,
        eval_fraction=eval_fraction,
        min_eval_matrices=min_eval_matrices,
    )


def build_csr_transformer_training_entrypoint(
    ready_dir: str | Path = "runs/phase1_csr_transformer_ready",
    quality_gate_summary_path: str | Path = (
        "runs/phase1_csr_transformer_quality_gate/"
        "csr_transformer_quality_gate_summary.json"
    ),
    output_dir: str | Path = "runs/phase1_csr_transformer_training_entrypoint",
    *,
    model_family: str = "csr_transformer_policy_v1",
) -> dict[str, Any]:
    """Build the formal training contract for a future CSR Transformer policy."""

    return build_csr_transformer_training_entrypoint_from_files(
        ready_dir,
        quality_gate_summary_path,
        output_dir,
        model_family=model_family,
    )


def build_csr_transformer_reference_training_export(
    tensor_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json"
    ),
    request_index_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl"
    ),
    training_entrypoint_summary_path: str | Path = (
        "runs/phase1_csr_transformer_training_entrypoint/"
        "csr_transformer_training_entrypoint_summary.json"
    ),
    output_dir: str | Path = "runs/phase1_csr_transformer_reference_training_export",
    *,
    contributor_id: str = "wei_cui_reference",
    contributor_name: str = "Wei CUI",
    epochs: int = 160,
    learning_rate: float = 0.03,
    l2_regularization: float = 1.0e-4,
    d_model: int = 24,
    num_attention_heads: int = 4,
    feedforward_dim: int = 48,
    seed: int = 18,
) -> dict[str, Any]:
    """Train the reference CSR Transformer ranker and export an intake checkpoint."""

    return build_csr_transformer_reference_training_export_from_files(
        tensor_path,
        request_index_path,
        training_entrypoint_summary_path,
        output_dir,
        contributor_id=contributor_id,
        contributor_name=contributor_name,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        d_model=d_model,
        num_attention_heads=num_attention_heads,
        feedforward_dim=feedforward_dim,
        seed=seed,
    )


def build_csr_policy_model_artifact(
    model_path: str | Path = "runs/phase1_csr_transformer_ranker/csr_transformer_ranker_model.json",
    tensor_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_training_tensors.json"
    ),
    request_index_path: str | Path = (
        "runs/phase1_csr_transformer_ready/csr_transformer_request_index.jsonl"
    ),
    quality_gate_summary_path: str | Path = (
        "runs/phase1_csr_transformer_quality_gate/"
        "csr_transformer_quality_gate_summary.json"
    ),
    replay_summary_path: str | Path = (
        "runs/phase1_csr_transformer_model_replay/"
        "csr_transformer_model_replay_summary.json"
    ),
) -> dict[str, Any]:
    """Build the runtime-loadable CSR policy model artifact contract."""

    return build_csr_policy_model_artifact_from_files(
        model_path,
        tensor_path,
        request_index_path,
        quality_gate_summary_path,
        replay_summary_path,
    )


def accept_csr_policy_model_artifact(
    model_artifact_path: str | Path = (
        "runs/phase1_csr_policy_model_artifact/csr_policy_model_artifact.json"
    ),
    csr_path: str | Path = "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
    selector_path: str | Path = "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl",
    learned_guard_summary_path: str | Path = (
        "runs/phase1_csr_learned_guard/csr_learned_guard_summary.json"
    ),
    guarded_auto_solve_summary_path: str | Path = (
        "runs/phase1_csr_guarded_auto_solve/csr_guarded_auto_solve_summary.json"
    ),
    guarded_auto_solve_results_path: str | Path = (
        "runs/phase1_csr_guarded_auto_solve/csr_guarded_auto_solve_results.jsonl"
    ),
) -> dict[str, Any]:
    """Run the guarded acceptance gate for a CSR policy model artifact."""

    return build_csr_policy_model_acceptance_from_files(
        model_artifact_path,
        csr_path,
        selector_path,
        learned_guard_summary_path,
        guarded_auto_solve_summary_path,
        guarded_auto_solve_results_path,
    )


def prepare_csr_policy_model_submission(
    model_artifact_path: str | Path = (
        "runs/phase1_csr_policy_model_artifact/csr_policy_model_artifact.json"
    ),
    acceptance_summary_path: str | Path = (
        "runs/phase1_csr_policy_model_acceptance/"
        "csr_policy_model_acceptance_summary.json"
    ),
    acceptance_rows_path: str | Path = (
        "runs/phase1_csr_policy_model_acceptance/"
        "csr_policy_model_acceptance_rows.jsonl"
    ),
    *,
    contributor_id: str = "wei_cui_reference",
    contributor_name: str = "Wei CUI",
    contribution_name: str = "csr_masked_self_attention_ranker_v1_reference_submission",
    contribution_version: str = "phase1-reference",
    training_statement: str = (
        "Reference CSR Transformer ranker trained on the current Phase 1 "
        "Transformer-ready SuiteSparse subset."
    ),
    contributor_terms_acknowledged: bool = True,
) -> dict[str, Any]:
    """Prepare a reviewable submission package for an external policy model."""

    return build_csr_policy_model_submission_from_files(
        model_artifact_path,
        acceptance_summary_path,
        acceptance_rows_path,
        contributor_id=contributor_id,
        contributor_name=contributor_name,
        contribution_name=contribution_name,
        contribution_version=contribution_version,
        training_statement=training_statement,
        contributor_terms_acknowledged=contributor_terms_acknowledged,
    )


def _load_candidate_set(candidate_set: CandidateSet | str | Path) -> CandidateSet:
    if isinstance(candidate_set, CandidateSet):
        return candidate_set
    return load_candidate_set(candidate_set)


def _coerce_csr_matrix(csr: CsrMatrix | dict[str, Any]) -> CsrMatrix:
    if isinstance(csr, CsrMatrix):
        return csr
    return csr_matrix_from_record(csr)


def _csr_context(
    context: SolveContext | None,
    *,
    precision: str | None,
) -> SolveContext:
    if context is None:
        return SolveContext(
            context_id="csr_public_api",
            tolerance_abs=1.0e-7,
            tolerance_rel=1.0e-5,
            max_iter=256,
            precision=precision or "float64",
            required_backend="taichi_gpu",
        )
    if precision is not None and precision != context.precision:
        return replace(context, precision=precision)
    return context


def _csr_selector_context(context: SolveContext | None) -> SolveContext:
    return context or SolveContext(
        context_id="phase1_csr_selector",
        tolerance_abs=1.0e-7,
        tolerance_rel=1.0e-5,
        max_iter=512,
        precision="float64",
        required_backend="taichi_gpu",
    )


def _fallback_plans(plan: PolicyPlan) -> tuple[PolicyPlan, ...]:
    plans = []
    for index, entry in enumerate(plan.fallback_chain, start=1):
        solver = dict(entry["solver"])
        candidate_id = str(entry["candidate_id"])
        plans.append(
            PolicyPlan(
                plan_id=f"{plan.plan_id}:fallback:{index}:{candidate_id}",
                backend=plan.backend,
                solver=solver,
                preconditioner=dict(entry.get("preconditioner", {})),
                reuse=dict(entry.get("reuse", {})),
                coarsening=dict(plan.coarsening),
                correction=dict(plan.correction),
                fallback_chain=[],
                budget=dict(plan.budget),
                audit={
                    **dict(plan.audit),
                    "candidate_id": candidate_id,
                    "selection_reason": "runtime_guard_fallback",
                    "fallback_index": index,
                    "fallback_source_plan_id": plan.plan_id,
                },
            )
        )
    return tuple(plans)
