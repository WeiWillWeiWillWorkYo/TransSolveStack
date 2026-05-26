import json
from pathlib import Path


def test_numerical_regression_artifacts_are_valid():
    records_path = Path("runs/phase1_numerical_regression/numerical_regression.jsonl")
    report_path = Path("runs/phase1_numerical_regression/numerical_regression.md")
    assert records_path.exists()
    assert report_path.exists()

    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 21
    assert all(record["passed"] for record in records)
    assert all(record["status"] == "success" for record in records)
    assert all(record["final_residual_norm"] <= 1.0e-6 for record in records)
    assert all(record["relative_error_to_true"] < 5.0e-3 for record in records)
    assert all(record["num_iterations"] <= record["max_iter"] for record in records)
    assert all(record["residual_drop"] <= 1.0e-3 for record in records)
