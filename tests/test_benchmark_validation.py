import pytest

from transsolvestack.benchmarks.candidates import (
    CandidateConfig,
    CandidateSet,
    load_candidate_set,
)
from transsolvestack.benchmarks.config import (
    WorkloadConfig,
    WorkloadContext,
    WorkloadFamily,
    load_workload_config,
)
from transsolvestack.benchmarks.validation import validate_workload_for_budget
from transsolvestack.runtime.config import load_resource_budget
from transsolvestack.runtime.resource_budget import (
    ResourceBudget,
    ResourceBudgetExceeded,
)


def test_phase1_smoke_workload_passes_budget_dry_run():
    workload = load_workload_config("configs/workloads/phase1_smoke.yaml")
    candidate_set = load_candidate_set(workload.candidate_set_path)
    budget = load_resource_budget(workload.resource_limits_path)
    report = validate_workload_for_budget(workload, candidate_set, budget)
    assert report.workload_id == "phase1_smoke"
    assert report.backend == "taichi_gpu"
    assert report.num_candidates == 3
    assert report.num_systems == 7
    assert report.max_problem_n == 4096
    assert report.max_effective_nnz == 28672
    assert report.max_candidate_iter == 300


def test_dry_run_rejects_backend_mismatch():
    workload = WorkloadConfig(
        workload_id="w",
        backend="taichi_gpu",
        families=(
            WorkloadFamily(
                family_id="f",
                operator_kind="structured_stencil",
                sizes=((16, 16),),
            ),
        ),
        contexts=(
            WorkloadContext(
                context_id="c",
                tolerance_abs=1.0e-8,
                tolerance_rel=1.0e-6,
                max_iter=100,
                precision="float32",
            ),
        ),
        candidate_set_path="unused",
    )
    candidate_set = CandidateSet(
        candidate_set_id="candidates",
        backend="cpu_reference",
        candidates=(
            CandidateConfig(candidate_id="cpu", solver={"name": "reference_direct"}),
        ),
    )
    with pytest.raises(ValueError):
        validate_workload_for_budget(workload, candidate_set, ResourceBudget())


def test_dry_run_rejects_large_workload_before_gpu_execution():
    workload = WorkloadConfig(
        workload_id="too_large",
        backend="taichi_gpu",
        families=(
            WorkloadFamily(
                family_id="poisson_2d_stencil",
                operator_kind="structured_stencil",
                sizes=((1024, 1024),),
            ),
        ),
        contexts=(
            WorkloadContext(
                context_id="c",
                tolerance_abs=1.0e-8,
                tolerance_rel=1.0e-6,
                max_iter=100,
                precision="float32",
            ),
        ),
        candidate_set_path="unused",
    )
    candidate_set = CandidateSet(
        candidate_set_id="candidates",
        backend="taichi_gpu",
        candidates=(CandidateConfig(candidate_id="pcg", solver={"name": "pcg"}),),
    )
    with pytest.raises(ResourceBudgetExceeded):
        validate_workload_for_budget(workload, candidate_set, ResourceBudget())


def test_dry_run_rejects_candidate_iteration_over_budget():
    workload = WorkloadConfig(
        workload_id="w",
        backend="taichi_gpu",
        families=(
            WorkloadFamily(
                family_id="poisson_2d_stencil",
                operator_kind="structured_stencil",
                sizes=((16, 16),),
            ),
        ),
        contexts=(
            WorkloadContext(
                context_id="c",
                tolerance_abs=1.0e-8,
                tolerance_rel=1.0e-6,
                max_iter=100,
                precision="float32",
            ),
        ),
        candidate_set_path="unused",
    )
    candidate_set = CandidateSet(
        candidate_set_id="candidates",
        backend="taichi_gpu",
        candidates=(
            CandidateConfig(candidate_id="pcg", solver={"name": "pcg", "max_iter": 301}),
        ),
    )
    with pytest.raises(ResourceBudgetExceeded):
        validate_workload_for_budget(workload, candidate_set, ResourceBudget())
