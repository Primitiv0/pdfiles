import logging
import os
from dataclasses import dataclass
from typing import Iterator
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


@dataclass
class PageRecord:
    page_id: str  # Display name (filename stem)
    volume: str
    pdf_path: Path  # Normalized path relative to images_root
    page_index: int
    pdf_id: str  # ID of the first page in this PDF
    total_pages: int  # Total pages in this PDF document
    point_id: int | None = None  # Qdrant integer point ID


def parse_opt(opt_path: Path, images_root: Path) -> list[PageRecord]:
    """Parse an Opticon .OPT file into PageRecord entries.

    Each line: ID,VOLUME,PATH,Y_FLAG,,,PAGE_COUNT
    Y_FLAG='Y' marks the first page of a PDF document.
    """
    records: list[PageRecord] = []
    current_pdf_id = ""
    current_total_pages = 0
    page_index = 0

    with open(opt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            raw_id = parts[0]
            volume = parts[1]
            raw_path = parts[2]
            is_first = parts[3] == "Y" if len(parts) > 3 else False

            # Keep the raw ID as page_id (the filename-like identifier)
            page_id = raw_id

            # Normalize path: backslash -> forward slash, resolve relative to images_root
            rel_path = raw_path.replace("\\", "/")
            # Strip leading volume/IMAGES prefix if present (path starts like IMAGES/0001/...)
            if rel_path.startswith("IMAGES/"):
                rel_path = rel_path[len("IMAGES/"):]
            pdf_path = images_root / rel_path

            if is_first:
                total_pages = int(parts[6]) if len(parts) > 6 and parts[6] else 1
                current_pdf_id = page_id
                current_total_pages = total_pages
                page_index = 0
            else:
                page_index += 1

            records.append(PageRecord(
                page_id=page_id,
                volume=volume,
                pdf_path=pdf_path,
                page_index=page_index,
                pdf_id=current_pdf_id,
                total_pages=current_total_pages,
                point_id=None,
            ))

    return records


def _parse_numeric_id(stem: str) -> int | None:
    """Extract numeric ID from a filename stem like 'ABC00039025' or '00039025'.

    Strips any alphabetic prefix and returns the integer value.
    Returns None if no valid numeric ID is found.
    """
    stripped = stem.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    if stripped and stripped.isdigit():
        return int(stripped)
    return None


def discover_opt_files(data_root: Path) -> list[Path]:
    """Find all OPT manifests under data_root (recursive)."""
    if not data_root.exists():
        return []

    manifests: list[Path] = []
    for root, _, files in os.walk(data_root):
        root_path = Path(root)
        for filename in files:
            if filename.lower().endswith(".opt"):
                manifests.append(root_path / filename)
    manifests.sort()
    return manifests


def infer_opt_volume_root(opt_path: Path) -> Path:
    """Infer the volume root for an OPT manifest path."""
    parent = opt_path.parent
    if parent.name.upper() == "DATA" and parent.parent != parent:
        return parent.parent
    return parent


def parse_opt_pdfs(opt_path: Path, images_root: Path) -> list[tuple[Path, int]]:
    """Parse an OPT file into per-PDF entries: (pdf_path, total_pages).

    Only first-page lines (Y flag) are used, which avoids materializing
    every page record when discovery only needs document-level info.
    """
    pdfs: list[tuple[Path, int]] = []
    with open(opt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 4 or parts[3] != "Y":
                continue

            raw_path = parts[2]
            total_pages = int(parts[6]) if len(parts) > 6 and parts[6] else 1

            rel_path = raw_path.replace("\\", "/")
            if rel_path.startswith("IMAGES/"):
                rel_path = rel_path[len("IMAGES/"):]

            pdf_path = images_root / rel_path
            if pdf_path.suffix.lower() != ".pdf":
                continue
            pdfs.append((pdf_path, total_pages))
    return pdfs


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def iter_pdfs_excluding_roots(data_root: Path, excluded_roots: list[Path]) -> Iterator[Path]:
    """Yield PDFs under data_root, excluding excluded_roots recursively."""
    if not data_root.exists():
        return

    excluded = [p.resolve() for p in excluded_roots]

    for root, dirs, files in os.walk(data_root):
        root_path = Path(root).resolve()

        if any(_is_within(root_path, ex) for ex in excluded):
            dirs[:] = []
            continue

        dirs[:] = [
            d for d in dirs
            if not any(_is_within((root_path / d).resolve(), ex) for ex in excluded)
        ]

        for filename in files:
            if filename.lower().endswith(".pdf"):
                yield root_path / filename


def walk_pdfs(
    data_root: Path,
    progress_callback: callable = None,
    record_callback: callable = None,
    excluded_roots: list[Path] | None = None,
) -> list[PageRecord]:
    """Walk a directory for PDFs and generate PageRecord entries.

    Used when no OPT manifest file is available (e.g. user drops PDFs in a folder).
    Numeric IDs are extracted from filenames (strips alpha prefix → numeric).
    Multi-page PDFs use base_id + page_index for subsequent pages.

    Args:
        progress_callback: Called with (pdfs_done, total_pdfs) after each PDF.
        record_callback: Called with list[PageRecord] per PDF for pipelining.
    """
    pdf_paths = sorted(iter_pdfs_excluding_roots(data_root, excluded_roots or []))
    total_pdfs = len(pdf_paths)
    logger.info("Found %d PDFs in %s", total_pdfs, data_root)

    records: list[PageRecord] = []
    next_id = 1

    for i, pdf_path in enumerate(pdf_paths):
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
        except Exception:
            logger.warning("Skipping unreadable PDF: %s", pdf_path)
            if progress_callback:
                progress_callback(i + 1, total_pdfs)
            continue

        pdf_id = pdf_path.stem

        new_records = []
        for page_idx in range(page_count):
            rec = PageRecord(
                page_id=pdf_path.stem,
                volume="LOCAL",
                pdf_path=pdf_path,
                page_index=page_idx,
                pdf_id=pdf_id,
                total_pages=page_count,
                point_id=next_id,
            )
            next_id += 1
            records.append(rec)
            new_records.append(rec)

        if record_callback and new_records:
            record_callback(new_records)

        if progress_callback:
            progress_callback(i + 1, total_pdfs)

        if (i + 1) % 500 == 0:
            logger.info("Scanning: %d/%d PDFs (%d pages so far)", i + 1, total_pdfs, len(records))

    logger.info("Generated %d page records from %d PDFs", len(records), total_pdfs)
    return records


def load_page_records(cfg) -> list[PageRecord]:
    """Load page records from ManifestDB, OPT manifest, or directory walk.

    Priority order:
    1. ManifestDB (authoritative IDs from a prior pipeline run)
    2. OPT manifests + directory walk (with sequential ID assignment)

    This is the single entry point for getting page records -- use this
    instead of calling parse_opt() or walk_pdfs() directly.
    """
    from pdfiles.manifest import ManifestDB

    # Priority 1: ManifestDB (authoritative from pipeline)
    if ManifestDB.exists(cfg.manifest_db):
        manifest = ManifestDB(cfg.manifest_db)
        records = manifest.iter_page_records()
        manifest.close()
        if records:
            logger.info(
                "Loaded %d page records from ManifestDB (%d unique PDFs)",
                len(records), len({r.pdf_path for r in records}),
            )
            return records

    # Priority 2: File discovery (OPT + walk) with sequential IDs
    records: list[PageRecord] = []
    covered_roots: list[Path] = []

    opt_files = discover_opt_files(cfg.data_root)
    if opt_files:
        logger.info("Found %d OPT manifest(s) under %s", len(opt_files), cfg.data_root)
        for opt_path in opt_files:
            volume_root = infer_opt_volume_root(opt_path)
            covered_roots.append(volume_root)
            images_root = volume_root / "IMAGES"
            logger.info("Parsing OPT manifest: %s", opt_path)
            opt_records = parse_opt(opt_path, images_root)
            records.extend(opt_records)

        logger.info(
            "Parsed %d OPT page records from %d unique PDFs",
            len(records), len({r.pdf_path for r in records}),
        )

        # Also walk uncovered roots for PDFs not represented by any OPT volume.
        walked = walk_pdfs(cfg.data_root, excluded_roots=covered_roots)
        existing = {(str(r.pdf_path), r.page_index) for r in records}
        extra = [r for r in walked if (str(r.pdf_path), r.page_index) not in existing]
        if extra:
            logger.info("Found %d additional pages in uncovered directories", len(extra))
            records.extend(extra)
    else:
        logger.info("No OPT manifest found, walking data directory for PDFs...")
        records = walk_pdfs(cfg.data_root)
        logger.info("Found %d page records via directory walk", len(records))

    # Final pass: assign sequential point_ids to all records
    for i, record in enumerate(records, start=1):
        record.point_id = i

    return records
