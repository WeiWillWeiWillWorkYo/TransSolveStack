from pathlib import Path

from transsolvestack.datasets.suitesparse_index import (
    build_suitesparse_index,
    parse_suitesparse_stats,
    read_suitesparse_index,
    write_suitesparse_index,
    write_suitesparse_report,
    write_suitesparse_summary,
)


def test_parse_suitesparse_stats_reads_headerless_csv(tmp_path):
    stats_path = _write_stats(tmp_path)

    stats = parse_suitesparse_stats(stats_path)

    assert stats.expected_matrices == 2
    assert stats.source_timestamp == "31-Oct-2023 18:12:37"
    assert len(stats.rows) == 2
    assert stats.rows[0].matrix_id == "suitesparse:HB/1138_bus"
    assert stats.rows[0].n_rows == 1138
    assert stats.rows[0].n_cols == 1138
    assert stats.rows[0].nnz == 4054
    assert stats.rows[0].is_real is True
    assert stats.rows[0].is_binary is False
    assert stats.rows[0].is_pos_def is True
    assert stats.rows[0].pattern_symmetry == 1.0
    assert stats.rows[0].kind == "power network problem"


def test_build_suitesparse_index_aligns_stats_urls_and_archives(tmp_path):
    stats_path = _write_stats(tmp_path)
    urls_path = _write_urls(tmp_path)
    mm_root = tmp_path / "MM"
    hb_root = mm_root / "HB"
    hb_root.mkdir(parents=True)
    (hb_root / "1138_bus.tar.gz").write_bytes(b"archive-a")
    (hb_root / "abb313.tar.gz").write_bytes(b"archive-bb")

    index = build_suitesparse_index(stats_path, urls_path, mm_root)

    assert index.summary.status == "complete"
    assert index.summary.expected_matrices == 2
    assert index.summary.present_archive_files == 2
    assert index.summary.actual_archive_files == 2
    assert index.summary.missing_archive_files == 0
    assert index.summary.missing_url_entries == 0
    assert index.summary.total_archive_size_bytes == len(b"archive-a") + len(
        b"archive-bb"
    )
    assert index.entries[0].url == "https://sparse.tamu.edu/MM/HB/1138_bus.tar.gz"
    assert index.entries[0].download_present is True
    assert index.entries[0].archive_size_bytes == len(b"archive-a")
    assert index.entries[0].tags == (
        "square",
        "real",
        "pos_def",
        "pattern_symmetric",
        "numerically_symmetric",
    )


def test_build_suitesparse_index_reports_missing_archive(tmp_path):
    stats_path = _write_stats(tmp_path)
    urls_path = _write_urls(tmp_path)
    mm_root = tmp_path / "MM"
    (mm_root / "HB").mkdir(parents=True)
    (mm_root / "HB" / "1138_bus.tar.gz").write_bytes(b"archive-a")

    index = build_suitesparse_index(stats_path, urls_path, mm_root)

    assert index.summary.status == "incomplete"
    assert index.summary.present_archive_files == 1
    assert index.summary.missing_archive_files == 1
    assert index.summary.missing_matrix_ids == ("suitesparse:HB/abb313",)
    missing = [entry for entry in index.entries if not entry.download_present]
    assert len(missing) == 1
    assert missing[0].archive_size_bytes is None


def test_suitesparse_index_artifacts_round_trip(tmp_path):
    stats_path = _write_stats(tmp_path)
    urls_path = _write_urls(tmp_path)
    mm_root = tmp_path / "MM"
    hb_root = mm_root / "HB"
    hb_root.mkdir(parents=True)
    (hb_root / "1138_bus.tar.gz").write_bytes(b"archive-a")
    (hb_root / "abb313.tar.gz").write_bytes(b"archive-bb")
    index = build_suitesparse_index(stats_path, urls_path, mm_root)

    index_path = tmp_path / "index" / "matrix_manifest.jsonl"
    summary_path = tmp_path / "index" / "index_summary.json"
    report_path = tmp_path / "index" / "index_report.md"
    write_suitesparse_index(index, index_path)
    write_suitesparse_summary(index.summary, summary_path)
    write_suitesparse_report(index.summary, report_path)

    reloaded = read_suitesparse_index(index_path)
    assert len(reloaded) == 2
    assert reloaded[1].matrix_id == "suitesparse:HB/abb313"
    assert summary_path.read_text(encoding="utf-8").count('"status": "complete"') == 1
    report = report_path.read_text(encoding="utf-8")
    assert "SuiteSparse Full Index" in report
    assert "present_archive_files: `2`" in report


def _write_stats(tmp_path: Path) -> Path:
    path = tmp_path / "ssstats.csv"
    path.write_text(
        "\n".join(
            [
                "2",
                "31-Oct-2023 18:12:37",
                "HB,1138_bus,1138,1138,4054,1,0,0,1,1,1,power network problem,4054",
                "HB,abb313,313,176,1557,1,1,0,0,0,0,least squares problem,1557",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_urls(tmp_path: Path) -> Path:
    path = tmp_path / "mm_urls.txt"
    path.write_text(
        "\n".join(
            [
                "https://sparse.tamu.edu/MM/HB/1138_bus.tar.gz",
                "https://sparse.tamu.edu/MM/HB/abb313.tar.gz",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path
