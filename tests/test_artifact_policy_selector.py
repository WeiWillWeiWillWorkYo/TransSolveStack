from transsolvestack.benchmarks.candidates import (
    CandidateConfig,
    CandidateSet,
    load_candidate_set,
)
from transsolvestack.policies.artifact_selector import BenchmarkArtifactPolicySelector
from transsolvestack.profiling.artifacts import write_jsonl


def test_artifact_policy_selector_uses_fastest_success_profile(tmp_path):
    candidate_set = _candidate_set()
    evaluation_path = write_jsonl(
        (
            {
                "system_id": "s1",
                "context_id": "ctx",
                "candidate_id": "slow",
                "status": "success",
                "is_oracle": False,
                "total_time_ms": 3.0,
                "regret_vs_oracle": 0.5,
                "num_iterations": 20,
                "final_residual_norm": 1.0e-7,
                "relative_error_to_true": 2.0e-7,
            },
            {
                "system_id": "s1",
                "context_id": "ctx",
                "candidate_id": "fast",
                "status": "success",
                "is_oracle": True,
                "total_time_ms": 2.0,
                "regret_vs_oracle": 0.0,
                "num_iterations": 12,
                "final_residual_norm": 8.0e-7,
                "relative_error_to_true": 9.0e-7,
            },
        ),
        tmp_path / "candidate_evaluation.jsonl",
    )

    selection = BenchmarkArtifactPolicySelector.from_evaluation_artifact(
        candidate_set, evaluation_path
    ).select("s1", "ctx")

    assert selection.candidate_id == "fast"
    assert selection.reason == "profiled_success"
    assert selection.fallback_candidate_ids == ("slow",)
    assert selection.plan.backend == "taichi_gpu"
    assert selection.plan.solver["name"] == "cg"
    assert selection.plan.preconditioner["name"] == "none"
    assert selection.plan.budget == {"max_iter": 300}
    assert selection.plan.audit["selector"] == "benchmark_artifact"
    assert selection.plan.audit["is_oracle"] is True
    assert selection.plan.audit["total_time_ms"] == 2.0
    assert selection.plan.audit["relative_error_to_true"] == 9.0e-7
    assert selection.plan.fallback_chain[0]["candidate_id"] == "slow"


def test_artifact_policy_selector_uses_explicit_fallback_when_unprofiled(tmp_path):
    candidate_set = _candidate_set()
    evaluation_path = write_jsonl(
        (
            {
                "system_id": "profiled",
                "context_id": "ctx",
                "candidate_id": "fast",
                "status": "success",
                "is_oracle": True,
                "total_time_ms": 1.0,
            },
        ),
        tmp_path / "candidate_evaluation.jsonl",
    )

    selection = BenchmarkArtifactPolicySelector.from_evaluation_artifact(
        candidate_set,
        evaluation_path,
        fallback_candidate_id="slow",
    ).select("unprofiled", "ctx")

    assert selection.candidate_id == "slow"
    assert selection.reason == "fallback_no_profile"
    assert selection.profile is None
    assert selection.plan.solver["name"] == "pcg"
    assert selection.plan.preconditioner["name"] == "jacobi"
    assert selection.plan.audit["is_oracle"] is False
    assert selection.plan.audit["profile_source"] is None
    assert selection.plan.audit["selection_reason"] == "fallback_no_profile"
    assert selection.plan.fallback_chain[0]["candidate_id"] == "fast"


def test_artifact_policy_selector_reads_current_benchmark_artifact():
    candidate_set = load_candidate_set("configs/candidates/phase1_taichi_gpu.yaml")
    selector = BenchmarkArtifactPolicySelector.from_evaluation_artifact(
        candidate_set,
        "runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
    )

    selection = selector.select(
        "synthetic:poisson_2d_stencil:32x32:float32",
        "default_f32",
    )

    assert selection.reason == "profiled_success"
    assert selection.profile is not None
    assert selection.profile.is_success is True
    assert selection.plan.backend == "taichi_gpu"
    assert selection.plan.audit["profile_source"].endswith(
        "runs/phase1_smoke_taichi/candidate_evaluation.jsonl"
    )
    assert len(selection.plan.fallback_chain) == 2


def _candidate_set() -> CandidateSet:
    return CandidateSet(
        candidate_set_id="test_candidates",
        backend="taichi_gpu",
        candidates=(
            CandidateConfig(
                candidate_id="fast",
                solver={"name": "cg", "max_iter": 300},
                preconditioner={"name": "none"},
                reuse={"device_buffers": True},
            ),
            CandidateConfig(
                candidate_id="slow",
                solver={"name": "pcg", "max_iter": 300},
                preconditioner={"name": "jacobi"},
                reuse={"device_buffers": True},
            ),
        ),
    )
