from pathlib import Path

import pytest

import config
from scripts.rebuild_indexes import select_pdfs


def test_select_pdfs_preserves_requested_order(tmp_path, monkeypatch):
    prospectus_dir = tmp_path / "prospectuses"
    prospectus_dir.mkdir()
    for name in ("BACH_2026.pdf", "JECX_2026.pdf"):
        (prospectus_dir / name).write_bytes(b"pdf")
    monkeypatch.setattr(config, "PROSPEKTUS_DIR", prospectus_dir)

    selected = select_pdfs(["JECX_2026.pdf", "BACH_2026.pdf"])
    assert [path.name for path in selected] == ["JECX_2026.pdf", "BACH_2026.pdf"]


def test_select_pdfs_rejects_missing_document(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROSPEKTUS_DIR", tmp_path)
    with pytest.raises(ValueError, match="PDF tidak ditemukan"):
        select_pdfs(["MISSING.pdf"])
