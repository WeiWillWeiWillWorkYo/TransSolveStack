import pytest

from transsolvestack.datasets.catalog import (
    build_download_plan,
    load_matrix_catalog,
)
from transsolvestack.runtime.resource_budget import (
    ResourceBudget,
    ResourceBudgetExceeded,
)


def test_load_phase1_external_catalog():
    catalog = load_matrix_catalog("configs/datasets/phase1_external_small.yaml")
    assert catalog.catalog_id == "phase1_external_small"
    assert len(catalog.matrices) == 3
    assert catalog.resource_limits_path == "configs/runtime/resource_limits.yaml"
    assert catalog.estimated_download_gb < 0.01
    assert all(entry.url.startswith("https://") for entry in catalog.matrices)


def test_dataset_download_plan_is_dry_run_and_budgeted(tmp_path):
    catalog = load_matrix_catalog("configs/datasets/phase1_external_small.yaml")
    plan = build_download_plan(catalog, ResourceBudget(), data_root=tmp_path)
    assert plan.dry_run is True
    assert len(plan.entries) == 3
    assert all(entry.download_required for entry in plan.entries)
    assert all(str(tmp_path) in entry.local_path for entry in plan.entries)
    assert plan.total_estimated_download_gb == catalog.estimated_download_gb


def test_dataset_download_plan_rejects_large_catalog():
    catalog = load_matrix_catalog("configs/datasets/phase1_external_small.yaml")
    tiny_budget = ResourceBudget(max_dataset_download_gb=0.0001)
    with pytest.raises(ResourceBudgetExceeded):
        build_download_plan(catalog, tiny_budget)
