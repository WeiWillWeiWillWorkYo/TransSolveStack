from transsolvestack.datasets.downloader import planned_local_path
from transsolvestack.datasets.manifest import MatrixManifestEntry
from transsolvestack.datasets.sources import KNOWN_SOURCES


def test_known_matrix_sources_are_registered():
    assert "suitesparse" in KNOWN_SOURCES
    assert "nist_matrix_market" in KNOWN_SOURCES


def test_planned_local_path_preserves_archive_suffixes():
    entry = MatrixManifestEntry(
        matrix_id="m",
        source="suitesparse",
        group="HB",
        name="bcsstk01",
        url="https://example.test/bcsstk01.tar.gz",
        file_format="matrix_market",
    )
    assert (
        str(planned_local_path(entry, data_root="/tmp/tss_data"))
        == "/tmp/tss_data/suitesparse/HB/bcsstk01.tar.gz"
    )
