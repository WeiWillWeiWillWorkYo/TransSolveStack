from transsolvestack.datasets.matrix_market import inspect_matrix_market_header


def test_matrix_market_header_probe_reads_fixture():
    header = inspect_matrix_market_header("tests/fixtures/tiny_spd.mtx")
    assert header.object_type == "matrix"
    assert header.storage_format == "coordinate"
    assert header.field == "real"
    assert header.symmetry == "symmetric"
    assert header.n_rows == 3
    assert header.n_cols == 3
    assert header.nnz == 5
    assert header.compressed is False
    assert header.is_sparse_coordinate is True
