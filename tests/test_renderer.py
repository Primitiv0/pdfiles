from pathlib import Path

from pdfiles.renderer import render_page, render_all_pages

TEST_PDF = Path(__file__).parent / "fixtures" / "test.pdf"


def test_render_page_dimensions():
    """Render first page of test PDF and check dimensions at 200 DPI."""
    img = render_page(TEST_PDF, 0, dpi=200)

    # At 200 DPI for a standard letter page (8.5x11"):
    # Width ~= 8.5 * 200/72 * 72 = 1700, Height ~= 11 * 200/72 * 72 = 2200
    assert img.width > 1000, f"Width too small: {img.width}"
    assert img.height > 1000, f"Height too small: {img.height}"
    assert img.mode == "RGB"


def test_render_all_pages():
    """Test PDF has 27 pages."""
    images = render_all_pages(TEST_PDF, dpi=200)

    assert len(images) == 27
    for img in images:
        assert img.mode == "RGB"
        assert img.width > 1000
        assert img.height > 1000
