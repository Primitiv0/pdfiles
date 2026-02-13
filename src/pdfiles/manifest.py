import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from pdfiles.opt_parser import PageRecord

logger = logging.getLogger(__name__)


class ManifestDB:
    """SQLite persistence for PDF scan results. Enables resume without re-scanning."""

    @staticmethod
    def exists(db_path: Path) -> bool:
        """Check if a ManifestDB file exists and is non-empty."""
        return db_path.exists() and db_path.stat().st_size > 0

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pdfs (
                pdf_path TEXT PRIMARY KEY,
                page_count INTEGER NOT NULL,
                first_page_id INTEGER NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS scan_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.commit()
        # Initialize next_id if not set
        row = self._conn.execute(
            "SELECT value FROM scan_meta WHERE key = 'next_id'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO scan_meta (key, value) VALUES ('next_id', '1')"
            )
            self._conn.execute(
                "INSERT INTO scan_meta (key, value) VALUES ('scan_complete', '0')"
            )
            self._conn.commit()

    def get_next_id(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM scan_meta WHERE key = 'next_id'"
        ).fetchone()
        return int(row[0])

    def _set_next_id(self, next_id: int) -> None:
        self._conn.execute(
            "UPDATE scan_meta SET value = ? WHERE key = 'next_id'",
            (str(next_id),),
        )

    def is_scan_complete(self) -> bool:
        row = self._conn.execute(
            "SELECT value FROM scan_meta WHERE key = 'scan_complete'"
        ).fetchone()
        return row is not None and row[0] == "1"

    def mark_scan_complete(self) -> None:
        self._conn.execute(
            "UPDATE scan_meta SET value = '1' WHERE key = 'scan_complete'"
        )
        self._conn.commit()

    def get_known_paths(self) -> set[str]:
        rows = self._conn.execute("SELECT pdf_path FROM pdfs").fetchall()
        return {r[0] for r in rows}

    def insert_pdf(self, pdf_path: str, page_count: int, first_page_id: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO pdfs (pdf_path, page_count, first_page_id) VALUES (?, ?, ?)",
            (pdf_path, page_count, first_page_id),
        )
        # Update next_id to be past this PDF's pages
        new_next = first_page_id + page_count
        current_next = self.get_next_id()
        if new_next > current_next:
            self._set_next_id(new_next)
        self._conn.commit()

    def iter_page_records(self) -> list[PageRecord]:
        """Expand all stored PDFs into PageRecords."""
        rows = self._conn.execute(
            "SELECT pdf_path, page_count, first_page_id FROM pdfs ORDER BY first_page_id"
        ).fetchall()
        records = []
        for pdf_path_str, page_count, first_page_id in rows:
            pdf_path = Path(pdf_path_str)
            pdf_id = pdf_path.stem
            for page_idx in range(page_count):
                records.append(PageRecord(
                    page_id=pdf_path.stem,
                    volume="LOCAL",
                    pdf_path=pdf_path,
                    page_index=page_idx,
                    pdf_id=pdf_id,
                    total_pages=page_count,
                    point_id=first_page_id + page_idx,
                ))
        return records

    def iter_page_record_batches(self, batch_size: int = 100) -> Iterator[list[PageRecord]]:
        """Yield stored PageRecords in small batches to avoid large allocations."""
        rows = self._conn.execute(
            "SELECT pdf_path, page_count, first_page_id FROM pdfs ORDER BY first_page_id"
        )
        batch: list[PageRecord] = []
        for pdf_path_str, page_count, first_page_id in rows:
            pdf_path = Path(pdf_path_str)
            pdf_id = pdf_path.stem
            for page_idx in range(page_count):
                batch.append(PageRecord(
                    page_id=pdf_path.stem,
                    volume="LOCAL",
                    pdf_path=pdf_path,
                    page_index=page_idx,
                    pdf_id=pdf_id,
                    total_pages=page_count,
                    point_id=first_page_id + page_idx,
                ))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def count_pdfs(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM pdfs").fetchone()
        return row[0]

    def count_pages(self) -> int:
        row = self._conn.execute("SELECT COALESCE(SUM(page_count), 0) FROM pdfs").fetchone()
        return row[0]

    def close(self) -> None:
        self._conn.close()
