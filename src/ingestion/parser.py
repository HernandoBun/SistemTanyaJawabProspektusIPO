"""
src/parser.py — PDF Parser for IPO Prospectus Documents

Strategi (urutan prioritas):
1. LlamaParse (Primary): Mengirim PDF ke LlamaParse API untuk ekstraksi
   teks dan tabel dalam format Markdown yang presisi.
   → Membutuhkan LLAMA_CLOUD_API_KEY di .env
2. pdfplumber (Fallback): Jika LlamaParse tidak tersedia atau gagal,
   gunakan pdfplumber lokal.

Output: List of ParsedPage objects dengan metadata halaman.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import hashlib
import json
import warnings
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Suppress harmless Pydantic v2 metadata warnings from third-party libraries (LlamaParse/LlamaIndex)
warnings.filterwarnings("ignore", message=".*validate_default.*")

import config
from src.ingestion.preprocessor import (
    get_preprocessing_config,
    preprocess_pages,
)

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """Represents a single parsed page from the PDF."""
    page_number: int          # 1-indexed
    text_content: str         # Narrative text (tables removed / clean)
    tables: List[str]         # Each table as Markdown string
    raw_text: str             # Full raw text (including any table content)
    removed_header_footer: List[str] = field(default_factory=list)
    preprocessing_stats: dict = field(default_factory=dict)


class ParserError(RuntimeError):
    """Raised when the configured research parser cannot complete."""


VALID_PARSER_MODES = {"strict_llamaparse", "llamaparse_fallback", "pdfplumber"}


def _source_digest(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_path(
    pdf_path: Path,
    parser_mode: str,
    source_digest: str,
    cache_version: str | None = None,
) -> Path:
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", pdf_path.stem).strip("_")
    filename = (
        f"{safe_stem}_{source_digest[:12]}_"
        f"{parser_mode}_{cache_version or config.PARSER_CACHE_VERSION}.json"
    )
    return config.PARSED_CACHE_DIR / filename


def _load_parse_cache(
    pdf_path: Path,
    parser_mode: str,
    source_digest: str,
    *,
    include_report: bool = False,
) -> tuple[List[ParsedPage], str] | tuple[List[ParsedPage], str, dict] | None:
    path = _cache_path(pdf_path, parser_mode, source_digest)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("source_sha256") != source_digest
            or payload.get("parser_mode") != parser_mode
            or payload.get("cache_version") != config.PARSER_CACHE_VERSION
            or payload.get("preprocessing_config") != get_preprocessing_config()
        ):
            return None
        pages = [ParsedPage(**item) for item in payload.get("pages", [])]
        if not pages:
            return None
        parser_used = payload.get("parser_used", parser_mode)
        if include_report:
            return pages, parser_used, payload.get("preprocessing_report", {})
        return pages, parser_used
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid parse cache %s: %s", path, exc)
        return None


def _load_legacy_v2_cache(
    pdf_path: Path,
    parser_mode: str,
    source_digest: str,
) -> tuple[List[ParsedPage], str] | None:
    """Load a valid pre-preprocessing cache for one-time local migration."""
    path = _cache_path(pdf_path, parser_mode, source_digest, cache_version="v2")
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("source_sha256") != source_digest
            or payload.get("parser_mode") != parser_mode
            or payload.get("cache_version") != "v2"
        ):
            return None
        pages = [ParsedPage(**item) for item in payload.get("pages", [])]
        if not pages:
            return None
        return pages, payload.get("parser_used", parser_mode)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid legacy parse cache %s: %s", path, exc)
        return None


def _save_parse_cache(
    pdf_path: Path,
    parser_mode: str,
    parser_used: str,
    source_digest: str,
    pages: List[ParsedPage],
    preprocessing_report: Optional[dict] = None,
) -> Path:
    config.PARSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(pdf_path, parser_mode, source_digest)
    payload = {
        "cache_version": config.PARSER_CACHE_VERSION,
        "source_file": pdf_path.name,
        "source_sha256": source_digest,
        "parser_mode": parser_mode,
        "parser_used": parser_used,
        "preprocessing_config": get_preprocessing_config(),
        "preprocessing_report": preprocessing_report or {},
        "page_count": len(pages),
        "pages": [asdict(page) for page in pages],
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)
    return path


# ------------------------------------------------------------------ #
#  Utility: table extraction from Markdown text
# ------------------------------------------------------------------ #

def _extract_md_tables(markdown_text: str) -> tuple[str, list[str]]:
    """
    Given a Markdown string, extract all Markdown table blocks.

    Returns:
        (text_without_tables, list_of_table_strings)
    """
    table_pattern = re.compile(
        r'(\|.+\|[ \t]*\n(?:\|[-: |]+\|[ \t]*\n)(?:\|.+\|[ \t]*\n?)+)',
        re.MULTILINE,
    )

    tables_found = []
    for match in table_pattern.finditer(markdown_text):
        tables_found.append(match.group(0).strip())

    # Remove tables from text
    clean_text = table_pattern.sub('\n', markdown_text).strip()

    return clean_text, tables_found


# ------------------------------------------------------------------ #
#  Parser 1: LlamaParse
# ------------------------------------------------------------------ #

def _parse_with_llamaparse(
    pdf_path: Path,
    *,
    strict: bool = False,
) -> Optional[List[ParsedPage]]:
    """
    Parse a PDF using LlamaParse API.
    Returns None if LlamaParse is unavailable.
    """
    api_key = config.LLAMA_CLOUD_API_KEY
    if not api_key or api_key == "your_llama_cloud_api_key_here":
        message = "LLAMA_CLOUD_API_KEY tidak tersedia."
        if strict:
            raise ParserError(message)
        logger.warning("%s Menggunakan parser fallback.", message)
        return None

    try:
        from llama_parse import LlamaParse
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "dependensi LlamaParse"
        message = (
            f"LlamaParse tidak dapat dimuat ({missing}). "
            "Pasang requirements.txt pada Python 3.12 yang digunakan proyek, "
            "lalu jalankan .\\proses_dokumen.ps1 -Check."
        )
        if strict:
            raise ParserError(message)
        logger.warning("%s Menggunakan parser fallback.", message)
        return None

    try:
        logger.info(f"Parsing with LlamaParse: {pdf_path.name}")
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",      # Get rich Markdown with table support
            verbose=False,
            language="id",               # Indonesian language hint
            system_prompt=(
                "Transkripsikan SELURUH isi setiap halaman Prospektus IPO ini "
                "secara setia ke Markdown. Pertahankan semua narasi, judul, "
                "subjudul, daftar, catatan kaki, nama orang, jabatan, tanggal, "
                "angka, satuan, serta tabel non-keuangan dan keuangan. Ubah tabel "
                "menjadi tabel Markdown dengan seluruh baris dan kolom tetap utuh. "
                "Jangan meringkas, menjawab, menjelaskan, memberi komentar, atau "
                "menulis kalimat seperti 'tidak ada tabel'. Jangan menambahkan "
                "informasi yang tidak tercetak pada halaman sumber."
            ),
        )

        # LlamaParse returns one Document per page
        documents = parser.load_data(str(pdf_path))

        if not documents:
            message = "LlamaParse tidak menghasilkan halaman apa pun."
            if strict:
                raise ParserError(message)
            logger.warning(message)
            return None

        parsed_pages: List[ParsedPage] = []
        for i, doc in enumerate(documents):
            page_num = i + 1
            md_content = doc.text or ""

            # Extract tables from Markdown
            clean_text, tables = _extract_md_tables(md_content)

            parsed_pages.append(ParsedPage(
                page_number=page_num,
                text_content=clean_text.strip(),
                tables=tables,
                raw_text=md_content.strip(),
            ))

        logger.info(f"LlamaParse: {len(parsed_pages)} pages parsed successfully.")
        return parsed_pages

    except ParserError:
        raise
    except Exception as e:
        if strict:
            raise ParserError(
                f"LlamaParse gagal untuk {pdf_path.name}: {e}"
            ) from e
        logger.error("LlamaParse gagal: %s. Menggunakan pdfplumber.", e)
        return None


# ------------------------------------------------------------------ #
#  Parser 2: pdfplumber (Fallback)
# ------------------------------------------------------------------ #

def _table_to_markdown(table: list) -> str:
    """Convert a pdfplumber table (list of lists) to a Markdown table string."""
    if not table or not table[0]:
        return ""

    cleaned = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        cleaned.append(cleaned_row)

    if not cleaned:
        return ""

    num_cols = max(len(row) for row in cleaned)
    padded = [row + [""] * (num_cols - len(row)) for row in cleaned]

    lines = []
    lines.append("| " + " | ".join(padded[0]) + " |")
    lines.append("| " + " | ".join(["---"] * num_cols) + " |")
    for row in padded[1:]:
        if any(cell.strip() for cell in row):
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _parse_with_pdfplumber(
    pdf_path: Path,
    progress_callback=None,
) -> List[ParsedPage]:
    """Fallback parser using pdfplumber."""
    import pdfplumber

    parsed_pages: List[ParsedPage] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info(f"pdfplumber: parsing {total_pages} pages from {pdf_path.name}")

        for i, page in enumerate(pdf.pages):
            page_num = i + 1

            # Extract tables
            tables_raw = page.extract_tables()
            tables_md: List[str] = []
            for tbl in tables_raw:
                md = _table_to_markdown(tbl)
                if md:
                    tables_md.append(md)

            # Extract narrative text (outside table bboxes)
            raw_text = page.extract_text() or ""
            try:
                table_bboxes = [t.bbox for t in page.find_tables()]
                if table_bboxes:
                    remaining = page
                    for bbox in table_bboxes:
                        try:
                            remaining = remaining.outside_bbox(bbox)
                        except Exception:
                            pass
                    narrative_text = remaining.extract_text() or ""
                else:
                    narrative_text = raw_text
            except Exception:
                narrative_text = raw_text

            parsed_pages.append(ParsedPage(
                page_number=page_num,
                text_content=narrative_text.strip(),
                tables=tables_md,
                raw_text=raw_text.strip(),
            ))

            if progress_callback:
                progress_callback(page_num, total_pages)

    logger.info(f"pdfplumber: {len(parsed_pages)} pages parsed.")
    return parsed_pages


# ------------------------------------------------------------------ #
#  Main Parser Class
# ------------------------------------------------------------------ #

class PDFParser:
    """
    Parses an IPO Prospectus PDF.

    Priority order:
    1. LlamaParse (if API key available — best for complex tables)
    2. pdfplumber (local fallback)
    """

    def __init__(self, pdf_path: str | Path, parser_mode: str | None = None):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        self.parser_mode = (parser_mode or config.PARSER_MODE).strip().lower()
        if self.parser_mode not in VALID_PARSER_MODES:
            raise ValueError(
                f"PARSER_MODE tidak valid: {self.parser_mode!r}. "
                f"Pilihan: {', '.join(sorted(VALID_PARSER_MODES))}"
            )
        self.parser_used: str = ""
        self.cache_hit: bool = False
        self.cache_migrated: bool = False
        self.cache_path: Optional[Path] = None
        self.preprocessing_report: dict = {}

    def parse(
        self,
        progress_callback=None,
        *,
        refresh_cache: bool = False,
    ) -> List[ParsedPage]:
        """
        Parse all pages in the PDF.

        Args:
            progress_callback: Optional callable(current, total) for progress.

        Returns:
            List of ParsedPage objects.
        """
        self.cache_migrated = False
        source_digest = _source_digest(self.pdf_path)
        self.cache_path = _cache_path(self.pdf_path, self.parser_mode, source_digest)
        if not refresh_cache:
            cached = _load_parse_cache(
                self.pdf_path,
                self.parser_mode,
                source_digest,
                include_report=True,
            )
            if cached is not None:
                pages, parser_used, preprocessing_report = cached
                self.parser_used = parser_used
                self.preprocessing_report = preprocessing_report
                self.cache_hit = True
                logger.info("Parse cache hit: %s", self.cache_path)
                return pages

            legacy = _load_legacy_v2_cache(
                self.pdf_path, self.parser_mode, source_digest
            )
            if legacy is not None:
                pages, parser_used = legacy
                self.parser_used = parser_used
                pages, self.preprocessing_report = preprocess_pages(pages)
                self.cache_path = _save_parse_cache(
                    self.pdf_path,
                    self.parser_mode,
                    self.parser_used,
                    source_digest,
                    pages,
                    preprocessing_report=self.preprocessing_report,
                )
                self.cache_hit = True
                self.cache_migrated = True
                logger.info("Legacy v2 parse cache migrated to: %s", self.cache_path)
                return pages

        self.cache_hit = False
        self.cache_migrated = False
        if self.parser_mode == "pdfplumber":
            pages = _parse_with_pdfplumber(
                self.pdf_path, progress_callback=progress_callback
            )
            self.parser_used = "pdfplumber"
        else:
            strict = self.parser_mode == "strict_llamaparse"
            pages = _parse_with_llamaparse(self.pdf_path, strict=strict)
            if pages is not None:
                self.parser_used = "llamaparse"
            else:
                pages = _parse_with_pdfplumber(
                    self.pdf_path, progress_callback=progress_callback
                )
                self.parser_used = "pdfplumber"

        pages, self.preprocessing_report = preprocess_pages(pages)
        self.cache_path = _save_parse_cache(
            self.pdf_path,
            self.parser_mode,
            self.parser_used,
            source_digest,
            pages,
            preprocessing_report=self.preprocessing_report,
        )
        logger.info("Parse cache saved: %s", self.cache_path)
        return pages


def get_document_metadata(pdf_path: str | Path) -> dict:
    """Extract basic document metadata."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            meta = pdf.metadata or {}
            return {
                "total_pages": len(pdf.pages),
                "title": meta.get("Title", ""),
                "author": meta.get("Author", ""),
                "creator": meta.get("Creator", ""),
                "file_name": Path(pdf_path).name,
            }
    except Exception:
        return {
            "total_pages": 0,
            "title": "",
            "author": "",
            "creator": "",
            "file_name": Path(pdf_path).name,
        }
