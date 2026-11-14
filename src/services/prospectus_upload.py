"""Validate an uploaded prospectus and add it to the existing indexes."""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

import config
from src.services.diagnostics import check_embedding_connection
from src.services.index_lock import index_write_lock
from src.services.pipeline import RAGPipeline


MAX_UPLOAD_MB = 50
MAX_UPLOAD_PAGES = 1000
_FILENAME = re.compile(r"[A-Z]{4}_(?:19|20)[0-9]{2}\.pdf")


class ProspectusUploadError(ValueError):
    """An upload error with a message suitable for the interface."""


def _validate_pdf(content: bytes) -> None:
    try:
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ProspectusUploadError("Gunakan PDF yang bisa dibuka tanpa kata sandi.")
            if not pdf.is_pdf or pdf.is_repaired or pdf.page_count == 0:
                raise ProspectusUploadError("PDF rusak atau tidak memiliki halaman.")
            if pdf.page_count > MAX_UPLOAD_PAGES:
                raise ProspectusUploadError(f"PDF boleh berisi maksimal {MAX_UPLOAD_PAGES} halaman.")
    except pymupdf.FileDataError as exc:
        raise ProspectusUploadError("File tidak dapat dibaca sebagai PDF.") from exc


def upload_prospectus(filename: str, content: bytes, progress_callback=None) -> None:
    """Publish the PDF and its registry entry once both indexes are ready."""
    if not _FILENAME.fullmatch(filename):
        raise ProspectusUploadError("Gunakan format KODE_TAHUN.pdf, misalnya ANDA_2026.pdf.")
    if not content:
        raise ProspectusUploadError("File PDF kosong. Pilih file prospektus yang berisi data.")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ProspectusUploadError(f"Ukuran PDF melebihi batas {MAX_UPLOAD_MB} MB.")
    if not content.startswith(b"%PDF-"):
        raise ProspectusUploadError("File tidak dapat dibaca sebagai PDF.")

    with index_write_lock():
        _validate_pdf(content)
        filename_key = filename.casefold()
        if any(path.name.casefold() == filename_key for path in config.PROSPEKTUS_DIR.iterdir()):
            raise ProspectusUploadError(f"{filename} sudah ada di penyimpanan prospektus.")

        pipeline = RAGPipeline(strict_registry=True)
        registered_names = pipeline.get_registered_document_names()
        if any(name.casefold() == filename_key for name in registered_names):
            raise ProspectusUploadError(f"{filename} sudah terdaftar dalam indeks.")
        if registered_names and not pipeline.is_ready():
            raise ProspectusUploadError("Indeks prospektus yang ada gagal dimuat. Hubungi pengelola aplikasi.")

        try:
            check_embedding_connection()
        except (ValueError, ConnectionError) as exc:
            raise ProspectusUploadError(str(exc)) from exc

        with TemporaryDirectory(prefix=".upload-", dir=config.PROSPEKTUS_DIR) as folder:
            staged_pdf = Path(folder) / filename
            staged_pdf.write_bytes(content)
            pipeline.ingest(
                staged_pdf,
                progress_callback=progress_callback,
                upload_destination=config.PROSPEKTUS_DIR / filename,
            )
