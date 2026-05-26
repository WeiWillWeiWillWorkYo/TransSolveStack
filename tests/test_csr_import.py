import io
import tarfile

import pytest

import transsolvestack as tss
from transsolvestack.datasets.csr import (
    import_archive_matrix_market_csr,
    import_selected_archives_to_csr,
    read_matrix_market_coordinate_csr,
    write_csr_import_records,
    write_csr_import_report,
    write_csr_import_summary,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


def test_matrix_market_symmetric_real_import_expands_to_csr():
    text = io.StringIO(
        "\n".join(
            [
                "%%MatrixMarket matrix coordinate real symmetric",
                "% comment",
                "3 3 5",
                "1 1 4.0",
                "2 1 -1.0",
                "2 2 4.0",
                "3 2 -1.0",
                "3 3 4.0",
                "",
            ]
        )
    )

    csr = read_matrix_market_coordinate_csr(
        text,
        matrix_id="fixture:tiny_spd",
        source_path="memory://tiny_spd.mtx",
    )

    assert csr.n_rows == 3
    assert csr.n_cols == 3
    assert csr.row_ptr == (0, 2, 5, 7)
    assert csr.col_ind == (0, 1, 0, 1, 2, 1, 2)
    assert csr.values == (4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0)
    assert csr.nnz == 7
    assert csr.metadata["expanded_entries_added"] == 2
    assert csr.matvec((1.0, 1.0, 1.0)) == (3.0, 2.0, 3.0)


def test_matrix_market_pattern_and_duplicate_entries_are_merged():
    text = io.StringIO(
        "\n".join(
            [
                "%%MatrixMarket matrix coordinate pattern general",
                "3 3 4",
                "1 1",
                "1 1",
                "2 3",
                "3 2",
                "",
            ]
        )
    )

    csr = read_matrix_market_coordinate_csr(
        text,
        matrix_id="fixture:pattern",
        source_path="memory://pattern.mtx",
    )

    assert csr.row_ptr == (0, 1, 2, 3)
    assert csr.col_ind == (0, 2, 1)
    assert csr.values == (2.0, 1.0, 1.0)
    assert csr.metadata["duplicate_entries_merged"] == 1


def test_matrix_market_skew_symmetric_import_negates_transpose():
    text = io.StringIO(
        "\n".join(
            [
                "%%MatrixMarket matrix coordinate real skew-symmetric",
                "3 3 2",
                "1 2 5.0",
                "2 3 -2.0",
                "",
            ]
        )
    )

    csr = read_matrix_market_coordinate_csr(
        text,
        matrix_id="fixture:skew",
        source_path="memory://skew.mtx",
    )

    assert csr.matvec((1.0, 1.0, 1.0)) == (5.0, -7.0, 2.0)
    assert csr.metadata["expanded_entries_added"] == 2


def test_matrix_market_complex_field_is_rejected():
    text = io.StringIO(
        "\n".join(
            [
                "%%MatrixMarket matrix coordinate complex general",
                "1 1 1",
                "1 1 1.0 0.0",
                "",
            ]
        )
    )

    with pytest.raises(ValueError, match="unsupported Matrix Market field"):
        read_matrix_market_coordinate_csr(
            text,
            matrix_id="fixture:complex",
            source_path="memory://complex.mtx",
        )


def test_archive_matrix_market_csr_import_reads_member(tmp_path):
    archive_path = _write_archive(
        tmp_path,
        matrix_name="tiny",
        content="\n".join(
            [
                "%%MatrixMarket matrix coordinate integer general",
                "2 2 2",
                "1 1 7",
                "2 1 -3",
                "",
            ]
        ),
    )

    member_path, csr = import_archive_matrix_market_csr(
        archive_path,
        matrix_id="suitesparse:Test/tiny",
        expected_name="tiny",
    )

    assert member_path == "tiny/tiny.mtx"
    assert csr.field == "integer"
    assert csr.row_ptr == (0, 1, 2)
    assert csr.values == (7.0, -3.0)


def test_selected_archive_csr_batch_import_filters_and_writes_artifacts(tmp_path):
    archive_small = _write_archive(
        tmp_path,
        matrix_name="small",
        content="\n".join(
            [
                "%%MatrixMarket matrix coordinate real symmetric",
                "2 2 3",
                "1 1 2.0",
                "2 1 -1.0",
                "2 2 2.0",
                "",
            ]
        ),
    )
    archive_large = _write_archive(
        tmp_path,
        matrix_name="large",
        content="\n".join(
            [
                "%%MatrixMarket matrix coordinate real general",
                "20000 20000 1",
                "1 1 1.0",
                "",
            ]
        ),
    )
    selection_rows = (
        _selected_row(archive_small, name="small", n_rows=2, n_cols=2, nnz=4, rank=1),
        _selected_row(
            archive_large,
            name="large",
            n_rows=20_000,
            n_cols=20_000,
            nnz=1,
            rank=2,
        ),
    )
    header_rows = (
        _header_row("suitesparse:Test/small", n_rows=2, n_cols=2, stored_nnz=3),
        _header_row("suitesparse:Test/large", n_rows=20_000, n_cols=20_000, stored_nnz=1),
    )

    batch = import_selected_archives_to_csr(
        selection_rows,
        header_probe_rows=header_rows,
        max_matrices=4,
        max_rows=100,
        max_cols=100,
        max_stored_entries=100,
        max_archive_size_bytes=1_000_000,
    )

    assert batch.summary.status == "passed"
    assert batch.summary.eligible_rows == 1
    assert batch.summary.imported_matrices == 1
    assert batch.records[0].csr_nnz == 4
    assert batch.records[0].row_ptr == (0, 2, 4)
    assert batch.records[0].values == (2.0, -1.0, -1.0, 2.0)

    csr_path = tmp_path / "csr_matrices.jsonl"
    summary_path = tmp_path / "csr_import_summary.json"
    report_path = tmp_path / "csr_import_report.md"
    write_csr_import_records(batch, csr_path)
    write_csr_import_summary(batch.summary, summary_path)
    write_csr_import_report(batch, report_path)
    assert len(read_jsonl(csr_path)) == 1
    assert '"imported_matrices": 1' in summary_path.read_text(encoding="utf-8")
    assert "SuiteSparse CSR Import Boundary" in report_path.read_text(encoding="utf-8")


def test_public_csr_import_api_reads_selection_files(tmp_path):
    archive_path = _write_archive(
        tmp_path,
        matrix_name="tiny",
        content="\n".join(
            [
                "%%MatrixMarket matrix coordinate real general",
                "2 2 2",
                "1 1 1.0",
                "2 2 2.0",
                "",
            ]
        ),
    )
    selection_path = tmp_path / "selected_matrices.jsonl"
    header_path = tmp_path / "archive_header_probe.jsonl"
    write_jsonl(
        (_selected_row(archive_path, name="tiny", n_rows=2, n_cols=2, nnz=2, rank=1),),
        selection_path,
    )
    write_jsonl(
        (_header_row("suitesparse:Test/tiny", n_rows=2, n_cols=2, stored_nnz=2),),
        header_path,
    )

    batch = tss.import_suitesparse_csr(selection_path, header_probe_path=header_path)

    assert batch.summary.status == "passed"
    assert batch.summary.imported_matrices == 1
    assert batch.records[0].col_ind == (0, 1)


def _write_archive(tmp_path, *, matrix_name: str, content: str):
    archive_path = tmp_path / f"{matrix_name}.tar.gz"
    payload = content.encode("utf-8")
    info = tarfile.TarInfo(f"{matrix_name}/{matrix_name}.mtx")
    info.size = len(payload)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))
    return archive_path


def _selected_row(
    archive_path,
    *,
    name: str,
    n_rows: int,
    n_cols: int,
    nnz: int,
    rank: int,
):
    return {
        "selection_rank": rank,
        "matrix_id": f"suitesparse:Test/{name}",
        "source": "suitesparse",
        "group": "Test",
        "name": name,
        "local_path": str(archive_path),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": nnz,
        "archive_size_bytes": archive_path.stat().st_size,
        "kind": "unit-test problem",
        "size_bucket": "tiny",
    }


def _header_row(matrix_id: str, *, n_rows: int, n_cols: int, stored_nnz: int):
    return {
        "matrix_id": matrix_id,
        "status": "success",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "stored_nnz": stored_nnz,
        "shape_matches": True,
    }
