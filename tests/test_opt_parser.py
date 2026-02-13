from pathlib import Path

from pdfiles.opt_parser import (
    _parse_numeric_id,
    discover_opt_files,
    infer_opt_volume_root,
    iter_pdfs_excluding_roots,
    load_page_records,
    parse_opt,
    parse_opt_pdfs,
    walk_pdfs,
)

TEST_PDF = Path(__file__).parent / "fixtures" / "test.pdf"


def test_parse_numeric_id_plain():
    assert _parse_numeric_id("00039025") == 39025


def test_parse_numeric_id_alpha_prefix():
    assert _parse_numeric_id("ABC00039025") == 39025


def test_parse_numeric_id_no_digits():
    assert _parse_numeric_id("nodigits") is None


def test_parse_numeric_id_empty():
    assert _parse_numeric_id("") is None


def test_walk_pdfs_fixture(tmp_path):
    """Walk a directory containing the test fixture PDF."""
    import shutil

    # Copy fixture with a numeric filename
    dest = tmp_path / "00000001.pdf"
    shutil.copy(TEST_PDF, dest)

    records = walk_pdfs(tmp_path)

    assert len(records) == 27  # 27-page PDF
    assert records[0].page_id == "00000001"  # filename stem
    assert records[0].pdf_id == "00000001"
    assert records[0].page_index == 0
    assert records[0].total_pages == 27
    assert records[0].point_id == 1  # sequential from 1

    # All pages share the same page_id (stem), point_id is unique
    for i, r in enumerate(records):
        assert r.page_index == i
        assert r.page_id == "00000001"
        assert r.point_id == 1 + i


def test_discover_opt_files_nested(tmp_path):
    opt_a = tmp_path / "A" / "DATA" / "VOL00011.OPT"
    opt_b = tmp_path / "B" / "sub" / "DATA" / "OTHER.opt"
    opt_a.parent.mkdir(parents=True)
    opt_b.parent.mkdir(parents=True)
    opt_a.write_text("dummy\n")
    opt_b.write_text("dummy\n")

    found = discover_opt_files(tmp_path)

    assert opt_a in found
    assert opt_b in found
    assert len(found) == 2


def test_infer_opt_volume_root_data_dir(tmp_path):
    opt_path = tmp_path / "VOL1" / "DATA" / "VOL00011.OPT"
    opt_path.parent.mkdir(parents=True)
    opt_path.write_text("dummy\n")

    volume_root = infer_opt_volume_root(opt_path)
    assert volume_root == tmp_path / "VOL1"


def test_parse_opt_pdfs_reads_first_page_entries(tmp_path):
    volume_root = tmp_path / "VOL1"
    data_dir = volume_root / "DATA"
    images_root = volume_root / "IMAGES"
    data_dir.mkdir(parents=True)
    images_root.mkdir(parents=True)

    opt_path = data_dir / "VOL00011.OPT"
    opt_path.write_text(
        "00000001,VOL00011,IMAGES/0330/00000001.pdf,Y,,,3\n"
        "00000002,VOL00011,IMAGES/0330/00000001.pdf,N,,,\n"
        "00000003,VOL00011,IMAGES/0330/00000001.pdf,N,,,\n"
    )

    pdfs = parse_opt_pdfs(opt_path, images_root)

    assert pdfs == [(images_root / "0330" / "00000001.pdf", 3)]


def test_iter_pdfs_excluding_roots(tmp_path):
    import shutil

    covered_root = tmp_path / "vol_a"
    uncovered_root = tmp_path / "misc"
    (covered_root / "IMAGES").mkdir(parents=True)
    uncovered_root.mkdir(parents=True)

    covered_pdf = covered_root / "IMAGES" / "00000001.pdf"
    uncovered_pdf = uncovered_root / "00000002.PDF"
    shutil.copy(TEST_PDF, covered_pdf)
    shutil.copy(TEST_PDF, uncovered_pdf)

    found = list(iter_pdfs_excluding_roots(tmp_path, [covered_root]))

    assert uncovered_pdf.resolve() in found
    assert covered_pdf.resolve() not in found


def test_walk_pdfs_non_numeric_filename(tmp_path):
    """PDFs with non-numeric filenames should get valid sequential IDs."""
    import shutil

    dest = tmp_path / "my-report.pdf"
    shutil.copy(TEST_PDF, dest)

    records = walk_pdfs(tmp_path)

    assert len(records) == 27
    assert records[0].page_id == "my-report"
    assert records[0].point_id == 1
    for i, r in enumerate(records):
        assert r.point_id == 1 + i


def test_parse_opt_returns_none_point_ids(tmp_path):
    """parse_opt() should return records with point_id=None."""
    images_root = tmp_path / "IMAGES"
    images_root.mkdir()

    opt_path = tmp_path / "test.opt"
    opt_path.write_text(
        "EFTA00001,VOL00011,IMAGES/0001/doc.pdf,Y,,,2\n"
        "EFTA00002,VOL00011,IMAGES/0001/doc.pdf,N,,,\n"
    )

    records = parse_opt(opt_path, images_root)

    assert len(records) == 2
    assert all(r.point_id is None for r in records)


def test_load_page_records_prefers_manifest(tmp_path):
    """load_page_records() should use ManifestDB when it exists."""
    import shutil
    from dataclasses import dataclass

    from pdfiles.manifest import ManifestDB

    # Create a PDF in the data root
    pdf_dest = tmp_path / "00000001.pdf"
    shutil.copy(TEST_PDF, pdf_dest)

    # Create a ManifestDB with specific IDs
    db_path = tmp_path / "manifest.db"
    manifest = ManifestDB(db_path)
    manifest.insert_pdf(str(pdf_dest), 27, 1000)  # first_page_id=1000
    manifest.close()

    @dataclass
    class FakeCfg:
        data_root: Path = tmp_path
        manifest_db: Path = db_path

    records = load_page_records(FakeCfg())

    assert len(records) == 27
    # Should use ManifestDB IDs (1000-based), not sequential (1-based)
    assert records[0].point_id == 1000
    assert records[-1].point_id == 1026


def test_load_page_records_falls_back_to_walk(tmp_path):
    """load_page_records() should fall back to walk when no ManifestDB exists."""
    import shutil
    from dataclasses import dataclass

    pdf_dest = tmp_path / "report.pdf"
    shutil.copy(TEST_PDF, pdf_dest)

    @dataclass
    class FakeCfg:
        data_root: Path = tmp_path
        manifest_db: Path = tmp_path / "nonexistent.db"

    records = load_page_records(FakeCfg())

    assert len(records) == 27
    # Sequential IDs assigned by final renumbering pass
    assert records[0].point_id == 1
    assert records[-1].point_id == 27
