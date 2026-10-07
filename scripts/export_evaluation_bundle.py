"""Export project evaluation code, datasets, and an index snapshot for Colab."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def collect_files(root: Path) -> list[Path]:
    root = root.resolve()
    mandatory = ["config.py", "requirements-evaluation.txt", "requirements-colab.txt",
                 "data/indexes/bm25_index.pkl", "data/indexes/chroma_db/chroma.sqlite3",
                 "data/registry/index_registry.json",
                 "data/evaluation/expert_rans/questions.jsonl",
                 "data/evaluation/expert_rans/ground_truth.draft.jsonl"]
    missing = [name for name in mandatory if not (root / name).is_file()]
    if missing:
        raise ValueError("Berkas untuk Colab belum lengkap: " + ", ".join(missing))
    files = [root / name for name in mandatory]
    for folder, suffixes in (("src", {".py"}), ("scripts", {".py"}),
                             ("evaluation", {".py", ".md"}),
                             ("data/evaluation", {".jsonl", ".json", ".md", ".txt"}),
                             ("data/indexes", None)):
        for path in (root / folder).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            if relative.parts[:2] == ("evaluation", "results"):
                continue
            if path.name.endswith(("-wal", "-shm")):
                continue
            if suffixes is None or path.suffix in suffixes:
                files.append(path)
    result = sorted(set(files))
    for path in result:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Berkas di luar proyek tidak diekspor: {path.name}")
    return result


def build_bundle(root: Path, output: Path) -> dict:
    root, output = root.resolve(), output.resolve()
    files = collect_files(root)
    if output.exists():
        raise FileExistsError(f"Paket sudah ada; gunakan nama keluaran baru: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in files}
    manifest = {"format_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contents": "evaluation_code_datasets_existing_indexes", "files": {},
                "excluded": [".env", "API keys", ".git", "installed_dependencies", "old_evaluation_results", "raw_PDFs"]}
    try:
        with tempfile.TemporaryDirectory(prefix="colab-snapshot-", dir=output.parent) as temporary:
            snapshot = Path(temporary) / "chroma.sqlite3"
            database = root / "data/indexes/chroma_db/chroma.sqlite3"
            with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as source:
                with closing(sqlite3.connect(snapshot)) as destination:
                    source.backup(destination)
            with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
                for path in files:
                    relative = path.relative_to(root).as_posix()
                    content_source = snapshot if path == database else path
                    digest = hashlib.sha256()
                    with content_source.open("rb") as handle, archive.open("Program/" + relative, "w", force_zip64=True) as member:
                        while block := handle.read(1024 * 1024):
                            digest.update(block)
                            member.write(block)
                    manifest["files"]["Program/" + relative] = {"sha256": digest.hexdigest(), "size": content_source.stat().st_size}
                changed = [path.name for path, state in before.items()
                           if (path.stat().st_size, path.stat().st_mtime_ns) != state]
                if changed:
                    raise RuntimeError("Berkas berubah saat ekspor; hentikan pemrosesan dokumen lalu ulangi: " + ", ".join(changed))
                archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception:
        # Only the new, partially written output owned by this export is removed.
        output.unlink(missing_ok=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--output", type=Path, default=default_root / "output/colab/evaluation_bundle.zip")
    args = parser.parse_args()
    manifest = build_bundle(default_root, args.output)
    print(f"Paket Colab: {args.output.resolve()}")
    print(f"Isi: {len(manifest['files'])} berkas; ukuran ZIP: {args.output.stat().st_size / 1024**2:.1f} MB")
    print("Dataset disalin sesuai kondisi saat ekspor. Anotasi yang belum lengkap tetap belum lengkap.")


if __name__ == "__main__":
    main()
