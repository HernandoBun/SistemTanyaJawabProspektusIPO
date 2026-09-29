"""Create a minimal deployment snapshot; run while ingestion is stopped."""
from pathlib import Path
import hashlib
import json
import shutil
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "deploy" / "community-cloud"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    if DEST.exists():
        raise SystemExit(f"Destination already exists; preserve/review it first: {DEST}")
    files = [ROOT / name for name in ("app.py", "config.py", "requirements.txt")]
    files += [p for p in (ROOT / "src").rglob("*") if p.suffix in {".py", ".css"}]
    files += [ROOT / ".streamlit" / name for name in ("config.toml", "secrets.toml.example")]
    files += [p for p in (ROOT / "data/indexes").rglob("*") if p.is_file()]
    files += [ROOT / "data/registry/index_registry.json"]
    # Hash before and after copying to detect concurrent changes to the snapshot.
    before = {p: digest(p) for p in files}
    for source in files:
        target = DEST / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if any(digest(p) != expected or digest(DEST / p.relative_to(ROOT)) != expected
           for p, expected in before.items()):
        raise SystemExit("Files changed during copy. Do not deploy this snapshot.")
    db = DEST / "data/indexes/chroma_db/chroma.sqlite3"
    with sqlite3.connect(f"{db.as_uri()}?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise SystemExit("Snapshot SQLite integrity check failed.")
    registry = json.loads((DEST / "data/registry/index_registry.json").read_text(encoding="utf-8"))
    shutil.copy2(ROOT / "scripts/community-cloud.gitignore", DEST / ".gitignore")
    shutil.copy2(ROOT / "docs/technical/deploy_streamlit_community.md", DEST / "README.md")
    print(f"Prepared {len(files)} files, {len(registry)} registry entries: {DEST}")
    print("Snapshot hashes and SQLite integrity: OK. No API requests made.")


if __name__ == "__main__":
    main()
