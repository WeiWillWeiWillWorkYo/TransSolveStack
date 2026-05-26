from transsolvestack.benchmarks.candidates import load_candidate_set
from transsolvestack.benchmarks.config import load_workload_config
from transsolvestack.runtime.config import load_resource_budget


def test_load_phase1_candidate_set():
    candidate_set = load_candidate_set("configs/candidates/phase1_taichi_gpu.yaml")
    assert candidate_set.candidate_set_id == "phase1_taichi_gpu"
    assert candidate_set.backend == "taichi_gpu"
    assert len(candidate_set.candidates) == 3
    assert candidate_set.candidates[0].max_iter == 300


def test_load_phase1_workload_config():
    workload = load_workload_config("configs/workloads/phase1_smoke.yaml")
    assert workload.workload_id == "phase1_smoke"
    assert workload.backend == "taichi_gpu"
    assert workload.resource_limits_path == "configs/runtime/resource_limits.yaml"
    assert workload.benchmark_protocol["warmup_runs"] == 1
    assert workload.benchmark_protocol["measurement_repeats"] == 3
    assert workload.families[0].problem_sizes() == (1024, 4096)


def test_load_resource_budget_config():
    budget = load_resource_budget("configs/runtime/resource_limits.yaml")
    assert budget.mode == "coexistence"
    assert budget.max_problem_n == 65536
    assert budget.max_default_iterations == 300
