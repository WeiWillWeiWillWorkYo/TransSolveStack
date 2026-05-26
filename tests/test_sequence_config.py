from transsolvestack.benchmarks.sequence_config import load_sequence_workload_config


def test_load_phase1_sequence_workload():
    workload = load_sequence_workload_config("configs/workloads/phase1_sequence.yaml")
    assert workload.workload_id == "phase1_sequence"
    assert workload.sequence_id == "anisotropy_drift_2d_32"
    assert workload.size == (32, 32)
    assert len(workload.steps) == 4
    assert workload.candidate_ids == ("taichi_cg_none_f32", "taichi_pcg_jacobi_f32")

