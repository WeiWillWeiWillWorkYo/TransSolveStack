from transsolvestack.datasets.selection import (
    MatrixSelectionCriteria,
    build_external_matrix_systems,
    select_matrix_subset,
    write_external_matrix_systems,
    write_matrix_selection,
    write_matrix_selection_report,
    write_matrix_selection_summary,
)
from transsolvestack.datasets.suitesparse_index import SuiteSparseIndexEntry
from transsolvestack.profiling.artifacts import read_jsonl


def test_matrix_selection_filters_and_stratifies_by_kind():
    selection = select_matrix_subset(
        _entries(),
        MatrixSelectionCriteria(
            max_matrices=3,
            max_per_kind=1,
            max_rows=100_000,
            max_cols=100_000,
            max_nnz=1_000_000,
            max_archive_size_bytes=100_000_000,
        ),
        source_index_path="index.jsonl",
    )

    assert selection.summary.status == "ready"
    assert selection.summary.candidates_considered == 6
    assert selection.summary.candidates_after_filter == 4
    assert selection.summary.selected_matrices == 3
    assert {record.kind for record in selection.records} == {
        "2D/3D problem",
        "optimization problem",
        "power network problem",
    }
    assert all(record.n_rows == record.n_cols for record in selection.records)
    assert all(record.is_real for record in selection.records)


def test_matrix_selection_can_require_spd():
    selection = select_matrix_subset(
        _entries(),
        MatrixSelectionCriteria(
            max_matrices=4,
            max_per_kind=4,
            require_positive_definite=True,
        ),
    )

    assert selection.summary.selected_matrices == 2
    assert all(record.symmetry_class == "spd" for record in selection.records)
    assert {record.matrix_id for record in selection.records} == {
        "suitesparse:HB/1138_bus",
        "suitesparse:Fluorem/thermal1",
    }


def test_external_matrix_systems_are_metadata_only():
    selection = select_matrix_subset(
        _entries(),
        MatrixSelectionCriteria(max_matrices=2, max_per_kind=1),
    )
    systems = build_external_matrix_systems(selection.records)

    assert len(systems) == 2
    assert all(system.operator.kind == "assembled_sparse" for system in systems)
    assert all(system.operator.device_resident is False for system in systems)
    assert all(
        system.operator.metadata["import_status"] == "archive_indexed_not_loaded"
        for system in systems
    )


def test_matrix_selection_artifacts_round_trip(tmp_path):
    selection = select_matrix_subset(
        _entries(),
        MatrixSelectionCriteria(max_matrices=2, max_per_kind=1),
    )
    systems = build_external_matrix_systems(selection.records)
    selection_path = tmp_path / "selected_matrices.jsonl"
    systems_path = tmp_path / "selected_systems.jsonl"
    summary_path = tmp_path / "selection_summary.json"
    report_path = tmp_path / "selection_report.md"

    write_matrix_selection(selection, selection_path)
    write_external_matrix_systems(systems, systems_path)
    write_matrix_selection_summary(selection.summary, summary_path)
    write_matrix_selection_report(selection, report_path)

    selected_rows = read_jsonl(selection_path)
    system_rows = read_jsonl(systems_path)
    assert len(selected_rows) == 2
    assert len(system_rows) == 2
    assert system_rows[0]["operator"]["kind"] == "assembled_sparse"
    assert '"selected_matrices": 2' in summary_path.read_text(encoding="utf-8")
    assert "SuiteSparse Matrix Subset Selection" in report_path.read_text(
        encoding="utf-8"
    )


def _entries() -> tuple[SuiteSparseIndexEntry, ...]:
    return (
        _entry(
            group="HB",
            name="1138_bus",
            n_rows=1138,
            n_cols=1138,
            nnz=4054,
            kind="power network problem",
            archive_size_bytes=20_000,
            is_pos_def=True,
        ),
        _entry(
            group="HB",
            name="abb313",
            n_rows=313,
            n_cols=176,
            nnz=1557,
            kind="least squares problem",
            archive_size_bytes=30_000,
        ),
        _entry(
            group="Boeing",
            name="crystk01",
            n_rows=4875,
            n_cols=4875,
            nnz=105339,
            kind="2D/3D problem",
            archive_size_bytes=600_000,
        ),
        _entry(
            group="Gset",
            name="G10",
            n_rows=800,
            n_cols=800,
            nnz=19176,
            kind="optimization problem",
            archive_size_bytes=250_000,
        ),
        _entry(
            group="Fluorem",
            name="thermal1",
            n_rows=82_654,
            n_cols=82_654,
            nnz=574_458,
            kind="thermal problem",
            archive_size_bytes=10_000_000,
            is_pos_def=True,
        ),
        _entry(
            group="Large",
            name="too_big",
            n_rows=1_000_000,
            n_cols=1_000_000,
            nnz=30_000_000,
            kind="large problem",
            archive_size_bytes=900_000_000,
        ),
    )


def _entry(
    *,
    group: str,
    name: str,
    n_rows: int,
    n_cols: int,
    nnz: int,
    kind: str,
    archive_size_bytes: int,
    is_pos_def: bool = False,
) -> SuiteSparseIndexEntry:
    return SuiteSparseIndexEntry(
        matrix_id=f"suitesparse:{group}/{name}",
        source="suitesparse",
        group=group,
        name=name,
        url=f"https://sparse.tamu.edu/MM/{group}/{name}.tar.gz",
        file_format="matrix_market_tar_gz",
        n_rows=n_rows,
        n_cols=n_cols,
        nnz=nnz,
        is_real=True,
        is_binary=False,
        is_nd=False,
        is_pos_def=is_pos_def,
        pattern_symmetry=1.0 if n_rows == n_cols else 0.0,
        numerical_symmetry=1.0 if is_pos_def else 0.0,
        kind=kind,
        nnzdiag=min(n_rows, nnz),
        local_path=f"/tmp/MM/{group}/{name}.tar.gz",
        archive_size_bytes=archive_size_bytes,
        download_present=True,
        tags=("square", "real") if n_rows == n_cols else ("rectangular", "real"),
    )
