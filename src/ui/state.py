"""Session-state initialization for the Streamlit application."""
from __future__ import annotations

import streamlit as st


DEFAULT_SESSION_STATE = {
    "current_page": "Beranda",
    "messages": [],
    "ingested": False,
    "ingestion_stats": {},
    "pdf_name": None,
    "pipeline_error": None,
}


def init_session() -> None:
    for key, value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value
