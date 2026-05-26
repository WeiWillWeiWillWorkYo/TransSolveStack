import json

from transsolvestack.benchmarks.oracle import build_fastest_success_oracle
from transsolvestack.benchmarks.report import write_dry_run_report
from transsolvestack.benchmarks.validation import BenchmarkDryRunReport
from transsolvestack.core.result import RunTrace
from transsolvestack.profiling.artifacts import (
    CandidatePerformanceRecord,
    performance_from_trace,
    write_candidate_performance,
    write_run_traces,
)


def test_run_trace_jsonl_writer(tmp_path):
    trace = RunTrace(
        run_id="run-1",
        plan_id="plan-1",
        backend="taichi_gpu",
        status="success",
        total_time_ms=2.0,
        gpu_kernel_time_ms=1.5,
    )
    output = write_run_traces([trace], tmp_path / "run_trace.jsonl")
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["backend"] == "taichi_gpu"
    assert rows[0]["gpu_kernel_time_ms"] == 1.5


def test_candidate_performance_record_from_trace_and_writer(tmp_path):
    trace = RunTrace(
        run_id="run-1",
        plan_id="plan-1",
        backend="taichi_gpu",
        status="success",
        total_time_ms=3.0,
        num_iterations=12,
    )
    record = performance_from_trace(
        trace,
        candidate_id="pcg_jacobi",
        context_id="ctx",
        run_trace_uri="run_trace.jsonl",
    )
    output = write_candidate_performance(
        [record],
        tmp_path / "candidate_performance.jsonl",
    )
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["candidate_id"] == "pcg_jacobi"
    assert row["context_id"] == "ctx"
    assert row["num_iterations"] == 12


def test_fastest_success_oracle_ignores_failed_records():
    records = [
        CandidatePerformanceRecord(
            run_id="failed",
            candidate_id="bad",
            context_id="ctx",
            backend="taichi_gpu",
            status="failed",
            total_time_ms=1.0,
        ),
        CandidatePerformanceRecord(
            run_id="slow",
            candidate_id="slow",
            context_id="ctx",
            backend="taichi_gpu",
            status="success",
            total_time_ms=5.0,
        ),
        CandidatePerformanceRecord(
            run_id="fast",
            candidate_id="fast",
            context_id="ctx",
            backend="taichi_gpu",
            status="success",
            total_time_ms=2.0,
        ),
    ]
    oracle = build_fastest_success_oracle("unknown", "ctx", records)
    assert oracle.best_candidate_id == "fast"
    assert oracle.best_total_time_ms == 2.0
    assert oracle.safe_candidate_id == "fast"
    assert oracle.system_id == "unknown"


def test_fastest_success_oracle_returns_empty_when_no_success():
    oracle = build_fastest_success_oracle(
        "unknown",
        "ctx",
        [
            CandidatePerformanceRecord(
                run_id="failed",
                candidate_id="bad",
                context_id="ctx",
                backend="taichi_gpu",
                status="failed",
            )
        ],
    )
    assert oracle.best_candidate_id is None
    assert oracle.best_total_time_ms is None


def test_dry_run_report_writer(tmp_path):
    report = BenchmarkDryRunReport(
        workload_id="phase1_smoke",
        candidate_set_id="phase1_taichi_gpu",
        backend="taichi_gpu",
        num_families=2,
        num_contexts=1,
        num_candidates=3,
        max_problem_n=4096,
        max_candidate_iter=300,
        budget_mode="coexistence",
        num_systems=4,
        max_effective_nnz=20480,
    )
    output = write_dry_run_report(report, tmp_path / "report.md")
    text = output.read_text(encoding="utf-8")
    assert "# Benchmark Dry Run: phase1_smoke" in text
    assert "candidate_set_id" in text
    assert "none" in text
