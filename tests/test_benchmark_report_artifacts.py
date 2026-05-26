from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl


def test_real_benchmark_evaluation_artifacts_are_valid():
    table_path = Path("runs/phase1_smoke_taichi/candidate_evaluation.jsonl")
    report_path = Path("runs/phase1_smoke_taichi/benchmark_report.md")
    assert table_path.exists()
    assert report_path.exists()
    rows = read_jsonl(table_path)
    assert len(rows) == 21
    assert sum(1 for row in rows if row["is_oracle"]) == 7
    assert all(row["status"] == "success" for row in rows)
    assert all(row["regret_vs_oracle"] is not None for row in rows)
    assert all(row["final_residual_norm"] <= 1.0e-6 for row in rows)
    assert "Per-System Candidate Table" in report_path.read_text(encoding="utf-8")
