import pytest

from transsolvestack.benchmarks.config import load_workload_config
from transsolvestack.benchmarks.expand import expand_workload_systems
from transsolvestack.operators.synthetic import (
    build_synthetic_system,
    get_synthetic_family,
    known_synthetic_families,
)


def test_known_synthetic_families_include_phase1_workload_families():
    families = known_synthetic_families()
    assert "poisson_2d_stencil" in families
    assert "anisotropic_diffusion_2d" in families
    assert "strong_anisotropic_diffusion_2d" in families
    assert "poisson_3d_stencil" in families
    assert families["poisson_2d_stencil"].operator_kind == "structured_stencil"


def test_build_synthetic_system_is_metadata_only():
    spec = build_synthetic_system("poisson_2d_stencil", (32, 32))
    assert spec.estimated_unknowns == 1024
    assert spec.estimated_effective_nnz == 5120
    assert spec.system.operator.shape == (1024, 1024)
    assert spec.system.operator.device_resident is True
    assert spec.system.metadata["materialized"] is False


def test_build_block_synthetic_system_accounts_for_dofs_per_node():
    spec = build_synthetic_system("linear_elasticity_block_2d", (8, 8))
    assert spec.estimated_unknowns == 128
    assert spec.system.operator.kind == "block_sparse"
    assert spec.system.operator.metadata["dofs_per_node"] == 2


def test_synthetic_family_rejects_wrong_dimension():
    family = get_synthetic_family("poisson_3d_stencil")
    with pytest.raises(ValueError):
        family.unknowns((16, 16))


def test_build_synthetic_system_rejects_operator_kind_mismatch():
    with pytest.raises(ValueError):
        build_synthetic_system(
            "poisson_2d_stencil",
            (16, 16),
            expected_operator_kind="block_sparse",
        )


def test_expand_phase1_workload_into_system_metadata():
    workload = load_workload_config("configs/workloads/phase1_smoke.yaml")
    systems = expand_workload_systems(workload)
    assert len(systems) == 7
    assert systems[0].system.family_id == "poisson_2d_stencil"
    assert systems[-1].system.family_id == "poisson_3d_stencil"
    assert systems[-1].estimated_unknowns == 4096
