from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


def render_page(pdf_path: Path, page_index: int, dpi: int = 200) -> Image.Image:
    """Render a single page of a PDF to a PIL Image."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img
    finally:
        doc.close()


def render_all_pages(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    """Render all pages of a PDF to PIL Images."""
    doc = fitz.open(pdf_path)
    images = []
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
        return images
    finally:
        doc.close()
