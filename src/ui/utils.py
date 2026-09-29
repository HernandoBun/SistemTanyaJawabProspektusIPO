"""Pure helper functions used by the Streamlit interface."""
from __future__ import annotations

import re
from pathlib import Path


def get_ticker_from_filename(filename: str | None) -> str:
    if not filename:
        return "Unknown"
    stem = Path(filename).stem
    stem = re.sub(r"\s*\(\d+\)", "", stem)
    stem = re.sub(r"[_\s]+", "-", stem)
    ignored = {
        "prospektus", "prospectus", "2026", "2025", "2024",
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    }
    candidates = [
        part.upper()
        for part in stem.lower().split("-")
        if part and part not in ignored and not part.isdigit()
    ]
    return candidates[0] if candidates else stem.upper()
