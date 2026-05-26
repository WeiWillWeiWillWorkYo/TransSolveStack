import json

import pytest

from transsolvestack.benchmarks.candidates import (
    CandidateConfig,
    CandidateSet,
    load_candidate_set,
)
from transsolvestack.benchmarks.config import load_workload_config
from transsolvestack.benchmarks.expand import expand_workload_systems
from transsolvestack.profiling.runner import (
    ProfileRunner,
    policy_plan_from_candidate,
    solve_context_from_workload,
)
from transsolvestack.runtime.config import load_resource_budget


def _phase1_inputs():
    workload = load_workload_config("configs/workloads/phase1_smoke.yaml")
    candidate_set = load_candidate_set(workload.candidate_set_path)
    budget = load_resource_budget(workload.resource_limits_path)
    return workload, candidate_set, budget


def test_profile_runner_builds_expected_phase1_run_plans():
    workload, candidate_set, _budget = _phase1_inputs()
    run_plans = ProfileRunner().build_run_plans(workload, candidate_set)
    assert len(run_plans) == 21
    assert run_plans[0].policy_plan.backend == "taichi_gpu"
    assert run_plans[0].policy_plan.solver["name"] == "pcg"
    assert run_plans[0].context_id == "default_f32"


def test_profile_runner_dry_run_generates_traces_and_performance():
    workload, candidate_set, budget = _phase1_inputs()
    result = ProfileRunner().dry_run(workload, candidate_set, budget)
    assert result.validation.num_systems == 7
    assert len(result.traces) == 21
    assert len(result.performance) == 21
    assert result.traces[0].status == "dry_run"
    assert result.traces[0].metadata["executed"] is False
    assert result.performance[0].status == "dry_run"


def test_profile_runner_writes_dry_run_artifacts(tmp_path):
    workload, candidate_set, budget = _phase1_inputs()
    runner = ProfileRunner()
    result = runner.dry_run(workload, candidate_set, budget)
    paths = runner.write_dry_run_artifacts(result, tmp_path)
    assert set(paths) == {
        "run_plans",
        "run_traces",
        "candidate_performance",
        "report",
    }
    trace_rows = [
        json.loads(line)
        for line in paths["run_traces"].read_text(encoding="utf-8").splitlines()
    ]
    perf_rows = [
        json.loads(line)
        for line in paths["candidate_performance"]
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(trace_rows) == 21
    assert len(perf_rows) == 21
    assert "Benchmark Dry Run" in paths["report"].read_text(encoding="utf-8")


def test_profile_runner_rejects_unsupported_candidate_before_execution():
    workload, _candidate_set, budget = _phase1_inputs()
    bad_candidates = CandidateSet(
        candidate_set_id="bad",
        backend="taichi_gpu",
        candidates=(
            CandidateConfig(
                candidate_id="direct",
                solver={"name": "direct_lu", "max_iter": 1},
            ),
        ),
    )
    with pytest.raises(ValueError):
        ProfileRunner().dry_run(workload, bad_candidates, budget)


def test_policy_plan_from_candidate_preserves_candidate_config():
    workload, candidate_set, _budget = _phase1_inputs()
    context = solve_context_from_workload(workload.contexts[0], backend=workload.backend)
    system = expand_workload_systems(workload)[0].system
    candidate = candidate_set.candidates[0]
    plan = policy_plan_from_candidate(
        system=system,
        context=context,
        candidate=candidate,
        backend=candidate_set.backend,
    )
    assert plan.solver["name"] == "pcg"
    assert plan.audit["system_id"] == system.system_id
