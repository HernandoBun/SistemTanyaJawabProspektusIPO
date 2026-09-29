"""
src/chunker.py — Advanced Chunking Strategy

Strategi:
1. Structural Chunking: Teks narasi dipecah berdasarkan heading Markdown (#, ##, ###)
   dengan sliding window dan overlap untuk menjaga konteks.
2. Table Isolation: Setiap tabel dipertahankan sebagai satu chunk atomik.
   Parent heading ditambahkan sebagai prefiks konteks (Parent-Child Architecture).

Output: List of Chunk dataclass objects dengan metadata lengkap.
"""

from __future__ import annotations

import re
import logging
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.ingestion.parser import ParsedPage
import config

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a single text/table chunk ready for embedding."""
    chunk_id: str
    content: str               # Teks yang akan di-embed
    chunk_type: str            # "narrative" | "table"
    page_number: int
    section_heading: str       # Heading/section terdekat di atas chunk ini
    doc_source: str            # Nama file PDF asli
    # --- Metadata enrichment (Poin 4) ---
    chapter: str = ""          # "BAB I", "BAB V", dll
    subchapter: str = ""       # "5.1 Pendapatan Usaha"
    table_title: str = ""      # Judul tabel jika chunk_type == "table"
    financial_unit: str = ""   # "(dalam jutaan Rupiah)" — kritis untuk angka!
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.content)



def _detect_heading(text: str) -> Optional[str]:
    """
    Detects if a line looks like a document section heading.
    Works for both Markdown headings and ALL-CAPS Indonesian headings.
    """
    lines = text.strip().split("\n")
    if not lines:
        return None

    first_line = lines[0].strip()

    # Markdown heading
    if re.match(r'^#{1,4}\s+.+', first_line):
        return re.sub(r'^#{1,4}\s+', '', first_line).strip()

    # ALL-CAPS line (common in Indonesian legal docs)
    if len(first_line) > 5 and first_line.isupper():
        return first_line

    # Roman numerals (BAB I, BAB II, etc.)
    if re.match(r'^(BAB|Bab)\s+[IVXLCDM\d]+', first_line, re.IGNORECASE):
        return first_line

    return None


def _extract_chapter(heading: str):
    """Extract (chapter, subchapter) from a heading string."""
    heading = re.sub(r'^#{1,4}\s+', '', heading).strip()
    bab_match = re.match(
        r'^(BAB\s+[IVXLCDM\d]+)\s*[—\-–:.]?\s*(.*)', heading, re.IGNORECASE
    )
    if bab_match:
        return bab_match.group(1).strip(), bab_match.group(2).strip()
    num_match = re.match(r'^(\d+(?:\.\d+)+)\s+(.*)', heading)
    if num_match:
        return "", heading
    return "", heading


def _extract_financial_unit(text: str) -> str:
    """Extract financial unit annotation like '(dalam jutaan Rupiah)' from text."""
    patterns = [
        r'\(dalam\s+(jutaan|ribuan|miliar|juta|ribu)\s+(?:Rupiah|USD|Rp\.?)\)',
        r'\(dalam\s+(?:Rupiah|USD|Rp\.?)\s+(jutaan|ribuan|miliar|juta|ribu)\)',
        r'dalam\s+(jutaan|ribuan|miliar)\s+(?:Rupiah|Rp)',
        r'\((?:Rp\.?\s*)?(jutaan|ribuan|miliar)\)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(0).strip('() ')
    return ""


def _extract_table_title(preceding_text: str) -> str:
    """Infer table title from the last few non-empty lines before the table."""
    if not preceding_text:
        return ""
    lines = [l.strip() for l in preceding_text.strip().split('\n') if l.strip()]
    candidates = lines[-3:] if len(lines) >= 3 else lines
    for line in reversed(candidates):
        if 10 < len(line) < 150 and not line.endswith(','):
            return line
    return ""


def _is_orphan_table(table_md: str) -> bool:
    """Detect if a table is a continuation from the previous page (no header row)."""
    lines = [l.strip() for l in table_md.strip().split('\n') if l.strip()]
    if not lines:
        return False
    return bool(re.match(r'^\|[-:\s|]+\|$', lines[0]))


def _merge_tables(header_table: str, continuation: str) -> str:
    """Merge a continuation table body into the header table."""
    cont_lines = [l for l in continuation.strip().split('\n')
                  if not re.match(r'^\|[-:\s|]+\|$', l.strip())]
    return header_table.rstrip() + '\n' + '\n'.join(cont_lines)


def _split_by_headings(
    text: str,
    page_number: int,
    initial_heading: str = "Pendahuluan",
) -> List[dict]:
    """
    Split a block of text into sections by heading markers.
    Returns list of {'heading': str, 'content': str, 'page': int}
    """
    # Pattern: line that looks like a section heading
    # Markdown: # heading, ## heading
    # Caps-lock: ANALISIS DAN PEMBAHASAN MANAJEMEN
    heading_pattern = re.compile(
        r'(?m)^(?:#{1,4}\s+.+|[A-Z][A-Z\s]{10,}|(?:BAB|Bab)\s+[IVXLCDM\d]+[^\n]*)$'
    )

    sections = []
    pos = 0
    current_heading = initial_heading or "Pendahuluan"
    matches = list(heading_pattern.finditer(text))

    for i, match in enumerate(matches):
        # Content before this heading belongs to current heading
        content_before = text[pos:match.start()].strip()
        if content_before:
            sections.append({
                "heading": current_heading,
                "content": content_before,
                "page": page_number,
            })

        current_heading = match.group(0).strip()
        pos = match.end()

    # Remaining text after last heading
    remaining = text[pos:].strip()
    if remaining:
        sections.append({
            "heading": current_heading,
            "content": remaining,
            "page": page_number,
        })

    if not sections:
        sections.append({
            "heading": "Konten",
            "content": text,
            "page": page_number,
        })

    return sections


def _sliding_window_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks by character count.
    Tries to split at sentence boundaries ('. ' or '\n') for cleaner chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to find a good split point (sentence boundary)
        split_pos = -1
        for sep in ['\n\n', '\n', '. ', '? ', '! ']:
            idx = text.rfind(sep, start + chunk_size // 2, end)
            if idx != -1:
                split_pos = idx + len(sep)
                break

        if split_pos == -1:
            split_pos = end

        chunk_text = text[start:split_pos].strip()
        if chunk_text:
            chunks.append(chunk_text)

        start = split_pos - overlap

    return chunks


class ProspectusChunker:
    """
    Chunks parsed prospectus pages into narrative and table chunks.
    Improvements:
    - Metadata enrichment (chapter, subchapter, table_title, financial_unit)
    - Cross-page table merging (Poin 3)
    """

    def __init__(
        self,
        doc_source: str,
        chunk_size: int = config.NARRATIVE_CHUNK_SIZE,
        chunk_overlap: int = config.NARRATIVE_CHUNK_OVERLAP,
        min_length: int = config.MIN_CHUNK_LENGTH,
    ):
        self.doc_source = doc_source
        normalized_stem = re.sub(
            r"[^a-z0-9]+", "_", Path(doc_source).stem.casefold()
        ).strip("_") or "document"
        source_hash = hashlib.sha1(
            doc_source.casefold().encode("utf-8")
        ).hexdigest()[:8]
        self._doc_id_prefix = f"{normalized_stem}_{source_hash}"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_length = min_length
        self._chunk_counter = 0
        # Structural state is carried across pages until a new heading appears.
        self._current_heading: str = "Pendahuluan"
        self._current_chapter: str = ""
        self._current_subchapter: str = ""
        # State for cross-page table merging
        self._pending_table: Optional[str] = None
        self._pending_heading: str = ""
        self._pending_page: int = 0
        self._pending_unit: str = ""
        self._pending_title: str = ""
        self._pending_chapter: str = ""
        self._pending_subchapter: str = ""

    def _new_id(self, prefix: str) -> str:
        self._chunk_counter += 1
        return f"{self._doc_id_prefix}__{prefix}_{self._chunk_counter:05d}"

    def chunk_pages(self, pages: List[ParsedPage]) -> List[Chunk]:
        """Process all parsed pages and return a flat list of Chunk objects."""
        all_chunks: List[Chunk] = []

        for page in pages:
            all_chunks.extend(self._chunk_narrative(page))
            all_chunks.extend(self._chunk_tables(page))

        # Flush any remaining pending cross-page table
        if self._pending_table:
            all_chunks.append(self._flush_pending_table())

        logger.info(
            f"Chunking complete: {len(all_chunks)} chunks total "
            f"({sum(1 for c in all_chunks if c.chunk_type == 'narrative')} narrative, "
            f"{sum(1 for c in all_chunks if c.chunk_type == 'table')} table)"
        )
        return all_chunks

    def _chunk_narrative(self, page: ParsedPage) -> List[Chunk]:
        """Split narrative text into structural chunks with heading + metadata context."""
        text = page.text_content
        if not text:
            return []

        page_unit = _extract_financial_unit(text)
        chunks: List[Chunk] = []
        sections = _split_by_headings(
            text, page.page_number, initial_heading=self._current_heading
        )

        for section in sections:
            heading = section["heading"]
            content = section["content"]

            if heading != self._current_heading:
                detected_chapter, detected_subchapter = _extract_chapter(heading)
                self._current_heading = heading
                if detected_chapter:
                    self._current_chapter = detected_chapter
                    self._current_subchapter = detected_subchapter
                else:
                    # A numbered or descriptive subheading remains under the
                    # most recently detected BAB.
                    self._current_subchapter = detected_subchapter

            chapter = self._current_chapter
            subchapter = self._current_subchapter
            if len(content) < self.min_length:
                continue

            section_unit = _extract_financial_unit(content) or page_unit
            sub_texts = _sliding_window_split(content, self.chunk_size, self.chunk_overlap)

            for sub_text in sub_texts:
                if len(sub_text) < self.min_length:
                    continue
                enriched_content = f"[{heading}]\n\n{sub_text}"
                chunks.append(Chunk(
                    chunk_id=self._new_id("narr"),
                    content=enriched_content,
                    chunk_type="narrative",
                    page_number=page.page_number,
                    section_heading=heading,
                    doc_source=self.doc_source,
                    chapter=chapter,
                    subchapter=subchapter,
                    financial_unit=section_unit,
                ))
        return chunks

    def _chunk_tables(self, page: ParsedPage) -> List[Chunk]:
        """
        Isolate each table as one chunk with cross-page merging support.
        """
        chunks: List[Chunk] = []
        page_text = page.text_content or ""
        page_heading = self._current_heading or "Tabel"
        chapter = self._current_chapter
        subchapter = self._current_subchapter
        page_unit = _extract_financial_unit(page_text)

        for table_md in page.tables:
            if len(table_md.strip()) < self.min_length:
                continue

            table_unit = _extract_financial_unit(table_md) or page_unit

            # Cross-page merge: orphan table (no header) merges with pending
            if _is_orphan_table(table_md) and self._pending_table:
                self._pending_table = _merge_tables(self._pending_table, table_md)
                self._pending_unit = self._pending_unit or table_unit
                logger.debug(f"Merged cross-page table at page {page.page_number}")
                continue

            # Flush completed pending table
            if self._pending_table:
                chunks.append(self._flush_pending_table())

            # Store new table as pending
            self._pending_table = table_md
            self._pending_heading = page_heading
            self._pending_page = page.page_number
            self._pending_unit = table_unit
            self._pending_title = _extract_table_title(page_text)
            self._pending_chapter = chapter
            self._pending_subchapter = subchapter

        return chunks

    def _flush_pending_table(self) -> Chunk:
        """Finalize pending table into a Chunk with full metadata."""
        heading = self._pending_heading
        unit_hint = f"\n(Satuan: {self._pending_unit})" if self._pending_unit else ""
        title_hint = f"\nJudul: {self._pending_title}" if self._pending_title else ""
        content = (
            f"[{heading}]{title_hint}{unit_hint}\n\n"
            f"Tabel berikut dari halaman {self._pending_page}:\n\n"
            f"{self._pending_table}"
        )
        chunk = Chunk(
            chunk_id=self._new_id("tbl"),
            content=content,
            chunk_type="table",
            page_number=self._pending_page,
            section_heading=heading,
            doc_source=self.doc_source,
            chapter=self._pending_chapter,
            subchapter=self._pending_subchapter,
            table_title=self._pending_title,
            financial_unit=self._pending_unit,
        )
        # Reset state
        self._pending_table = None
        self._pending_heading = ""
        self._pending_page = 0
        self._pending_unit = ""
        self._pending_title = ""
        self._pending_chapter = ""
        self._pending_subchapter = ""
        return chunk
