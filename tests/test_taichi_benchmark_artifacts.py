from pathlib import Path

from transsolvestack.profiling.artifacts import read_candidate_performance


def test_real_taichi_smoke_benchmark_artifacts_are_numerically_valid():
    profile_dir = Path("runs/phase1_smoke_taichi")
    perf_path = profile_dir / "candidate_performance.jsonl"
    oracle_path = profile_dir / "oracle_plan.jsonl"
    summary_path = profile_dir / "performance_summary.md"
    assert perf_path.exists()
    assert oracle_path.exists()
    assert summary_path.exists()

    records = read_candidate_performance(perf_path)
    assert len(records) == 21
    assert {record.status for record in records} == {"success"}
    assert all(record.backend == "taichi_gpu" for record in records)
    assert all(record.system_id.startswith("synthetic:") for record in records)
    assert all(record.final_residual_norm is not None for record in records)
    assert all(record.final_residual_norm <= 1.0e-6 for record in records)
    assert all(record.num_iterations is not None for record in records)
    assert all(record.num_iterations <= 300 for record in records)
    assert all(record.total_time_ms is not None for record in records)
    assert all(record.total_time_ms > 0.0 for record in records)
    assert all(record.warmup_runs == 1 for record in records)
    assert all(record.measurement_repeats == 3 for record in records)
    assert all(len(record.measurement_total_time_ms) == 3 for record in records)
    assert all(record.total_time_ms_min == record.total_time_ms for record in records)
    assert all(record.total_time_ms_median is not None for record in records)
    assert all(record.gpu_kernel_time_ms_std is not None for record in records)
    assert all(record.metadata["relative_error_to_true"] < 5.0e-3 for record in records)
