import io
import tarfile

import transsolvestack as tss
from transsolvestack.datasets.archive_probe import (
    inspect_archive_matrix_market_header,
    probe_selected_matrix_archives,
    write_archive_header_probe,
    write_archive_header_report,
    write_archive_header_summary,
)
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


def test_archive_probe_reads_matrix_market_header_without_extracting(tmp_path):
    archive_path = _write_archive(
        tmp_path,
        matrix_name="tiny",
        header="%%MatrixMarket matrix coordinate real symmetric\n3 3 5\n",
    )

    member_path, header = inspect_archive_matrix_market_header(
        archive_path,
        expected_name="tiny",
    )

    assert member_path == "tiny/tiny.mtx"
    assert header.object_type == "matrix"
    assert header.storage_format == "coordinate"
    assert header.field == "real"
    assert header.symmetry == "symmetric"
    assert header.n_rows == 3
    assert header.n_cols == 3
    assert header.nnz == 5


def test_selected_archive_probe_reports_success_and_mismatches(tmp_path):
    archive_ok = _write_archive(
        tmp_path,
        matrix_name="ok",
        header="%%MatrixMarket matrix coordinate pattern general\n4 4 8\n",
    )
    archive_bad = _write_archive(
        tmp_path,
        matrix_name="bad",
        header="%%MatrixMarket matrix coordinate real general\n9 9 7\n",
    )
    rows = (
        _selected_row(archive_ok, name="ok", n_rows=4, n_cols=4, nnz=8, is_binary=True),
        _selected_row(archive_bad, name="bad", n_rows=5, n_cols=5, nnz=7),
    )

    probe = probe_selected_matrix_archives(rows)

    assert probe.summary.status == "failed"
    assert probe.summary.probed_archives == 2
    assert probe.summary.success_count == 2
    assert probe.summary.failure_count == 0
    assert probe.summary.shape_mismatch_count == 1
    assert probe.summary.stored_nnz_mismatch_count == 0
    assert probe.records[0].field == "pattern"
    assert probe.records[0].field_matches_expected is True
    assert probe.records[1].shape_matches is False


def test_archive_probe_artifacts_round_trip(tmp_path):
    archive_path = _write_archive(
        tmp_path,
        matrix_name="tiny",
        header="%%MatrixMarket matrix coordinate real general\n3 3 5\n",
    )
    probe = probe_selected_matrix_archives(
        (_selected_row(archive_path, name="tiny", n_rows=3, n_cols=3, nnz=5),)
    )
    probe_path = tmp_path / "archive_header_probe.jsonl"
    summary_path = tmp_path / "archive_header_summary.json"
    report_path = tmp_path / "archive_header_report.md"

    write_archive_header_probe(probe, probe_path)
    write_archive_header_summary(probe.summary, summary_path)
    write_archive_header_report(probe, report_path)

    rows = read_jsonl(probe_path)
    assert len(rows) == 1
    assert rows[0]["matrix_id"] == "suitesparse:Test/tiny"
    assert '"status": "passed"' in summary_path.read_text(encoding="utf-8")
    assert "SuiteSparse Archive Header Probe" in report_path.read_text(
        encoding="utf-8"
    )


def test_public_archive_probe_api_reads_selection_file(tmp_path):
    archive_path = _write_archive(
        tmp_path,
        matrix_name="tiny",
        header="%%MatrixMarket matrix coordinate real general\n3 3 5\n",
    )
    selection_path = tmp_path / "selected_matrices.jsonl"
    write_jsonl(
        (
            _selected_row(
                archive_path,
                name="tiny",
                n_rows=3,
                n_cols=3,
                nnz=5,
            ),
        ),
        selection_path,
    )

    probe = tss.probe_suitesparse_headers(selection_path)

    assert probe.summary.status == "passed"
    assert probe.records[0].matrix_id == "suitesparse:Test/tiny"


def _write_archive(tmp_path, *, matrix_name: str, header: str):
    archive_path = tmp_path / f"{matrix_name}.tar.gz"
    payload = (header + "1 1 1.0\n").encode("utf-8")
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
    is_binary: bool = False,
):
    return {
        "selection_rank": 1,
        "matrix_id": f"suitesparse:Test/{name}",
        "source": "suitesparse",
        "group": "Test",
        "name": name,
        "local_path": str(archive_path),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": nnz,
        "is_binary": is_binary,
        "is_real": True,
        "archive_size_bytes": archive_path.stat().st_size,
        "kind": "unit-test problem",
        "size_bucket": "tiny",
    }
