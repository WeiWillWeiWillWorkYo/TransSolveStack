import pytest

from transsolvestack.runtime.resource_budget import (
    ResourceBudget,
    ResourceBudgetExceeded,
)


def test_resource_budget_accepts_smoke_problem():
    budget = ResourceBudget()
    budget.validate_problem_size(n=4096, effective_nnz=20000)
    budget.validate_iterations(max_iter=300)
    budget.validate_download_size(download_gb=1.5)


def test_resource_budget_rejects_large_problem():
    budget = ResourceBudget(max_problem_n=1024)
    with pytest.raises(ResourceBudgetExceeded):
        budget.validate_problem_size(n=2048)


def test_resource_budget_rejects_large_download():
    budget = ResourceBudget(max_dataset_download_gb=2)
    with pytest.raises(ResourceBudgetExceeded):
        budget.validate_download_size(download_gb=3)

