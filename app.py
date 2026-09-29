"""
app.py — Sistem Tanya Jawab Prospektus IPO
Streamlit Web Application

Fitur:
- Pemilihan prospektus yang sudah diproses secara offline
- Chat interface dengan chat history
- Sitasi halaman sumber untuk setiap jawaban
- Tampilan chunk konteks (expandable)
- Statistik sistem (chunk count, model info)
"""

import logging
import html
from pathlib import Path
import sys

import streamlit as st

# ---- Setup logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---- Page config MUST be first Streamlit call ----
st.set_page_config(
    page_title="Sistem Tanya Jawab Prospektus IPO",
    page_icon=":material/description:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep styling separate from application logic.
st.markdown(
    "<style>" + (Path(__file__).parent / "src/ui/styles.css").read_text(encoding="utf-8") + "</style>",
    unsafe_allow_html=True,
)

# ---- Import Pipeline (after page config) ----
import config
from src.ui.state import init_session
from src.ui.utils import get_ticker_from_filename
from src.generation.generator import clean_generated_answer

def load_pipeline():
    """Keep active-document state private to this browser session."""
    from src.services.pipeline import RAGPipeline
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
    return st.session_state.pipeline


# ============================================================
#  Session State
# ============================================================
init_session()


# ============================================================
#  Sidebar Navigation
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.subheader("Sistem Tanya Jawab Prospektus IPO")
        st.caption("Dokumen dan tanya jawab")
        st.divider()
        pages = ["Beranda", "Manajemen Emiten", "Tanya Jawab", "Tentang"]
        # Normalize page names from sessions opened before the UI update.
        current = st.session_state.current_page.replace("Pilih Dokumen", "Manajemen Emiten")
        current = next((page for page in pages if current.endswith(page)), "Beranda")
        st.session_state.current_page = current
        for page in pages:
            if st.button(page, key=f"nav_{page}", use_container_width=True,
                         type="primary" if current == page else "secondary"):
                st.session_state.current_page = page
                st.rerun()
        st.divider()
        active_doc = load_pipeline().get_active_document()
        st.caption("Emiten aktif")
        if active_doc:
            st.write(get_ticker_from_filename(active_doc))
        else:
            st.caption("Belum dipilih")

def render_home():
    st.title("Sistem Tanya Jawab Prospektus IPO")

    pipeline = load_pipeline()
    active_doc = pipeline.get_active_document()
    active_ticker = get_ticker_from_filename(active_doc) if active_doc else None

    # Status emiten aktif
    if active_ticker:
        status_badge = f'<div class="home-badge active">Emiten Aktif: &nbsp;<strong>{html.escape(active_ticker)}</strong></div>'
    else:
        status_badge = '<div class="home-badge none">Emiten Aktif: &nbsp;Belum dipilih</div>'

    st.markdown(
        f"""
        <div class="home-hero">
            {status_badge}
            <p style="color: #94A3B8; font-size: 0.9rem; line-height: 1.6; margin-bottom: 0;">
                Sistem ini menggunakan arsitektur <em>Retrieval-Augmented Generation (RAG)</em> dengan 
                pencarian hibrida untuk membantu Anda menemukan fakta keuangan, rencana penggunaan dana, 
                risiko bisnis, hingga struktur permodalan langsung dari dokumen prospektus IPO.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Mulai dari Dokumen")
    st.write(
        "Pilih dokumen prospektus yang tersedia, lalu mulai bertanya tentang perusahaan tersebut. "
        "Seluruh jawaban dilengkapi rujukan halaman agar dapat diperiksa kembali pada dokumen asli."
    )

    col_btn1, col_btn2 = st.columns([1, 1] if active_doc else [1, 2])
    with col_btn1:
        if active_doc:
            if st.button(f"Lanjutkan Tanya Jawab ({active_ticker})", type="primary", use_container_width=True):
                st.session_state.current_page = "Tanya Jawab"
                st.rerun()
        else:
            if st.button("Pilih Emiten", type="primary", use_container_width=True):
                st.session_state.current_page = "Manajemen Emiten"
                st.rerun()
    with col_btn2:
        if active_doc:
            if st.button("Ganti Dokumen Emiten", type="secondary", use_container_width=True):
                st.session_state.current_page = "Manajemen Emiten"
                st.rerun()

    st.markdown(
        """
        <div class="home-feature-grid">
            <div class="home-feature-item">
                <div class="home-feature-title">📑 Berbasis Dokumen Resmi</div>
                <div class="home-feature-desc">Informasi diperoleh langsung dari teks dan tabel prospektus IPO tanpa rekayasa data.</div>
            </div>
            <div class="home-feature-item">
                <div class="home-feature-title">⚡ Pencarian Hibrida & RRF</div>
                <div class="home-feature-desc">Kombinasi pencarian makna (vektor) dan kata kunci (BM25) menghasilkan rujukan yang presisi.</div>
            </div>
            <div class="home-feature-item">
                <div class="home-feature-title">📄 Terverifikasi Halaman</div>
                <div class="home-feature-desc">Setiap jawaban menyertakan rujukan nomor halaman sumber agar mudah ditelusuri kembali.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_management():
    st.title("Manajemen Emiten")
    st.caption("Pilih emiten yang sudah siap untuk memulai tanya jawab.")
    pipeline = load_pipeline()
    indexed = pipeline.get_indexed_docs()
    active = pipeline.get_active_document()
    names = set(indexed) | {p.name for p in config.PROSPEKTUS_DIR.glob("*.pdf")}
    if not names:
        st.info("Belum ada dokumen. Prospektus akan tampil setelah disiapkan oleh pengelola.")
        return
    st.subheader("Daftar Dokumen di Server")
    ordered = sorted(names, key=lambda name: (0 if name == active else 1 if name in indexed else 2, name))
    for start in range(0, len(ordered), 3):
        columns = st.columns(3)
        for column, name in zip(columns, ordered[start:start + 3]):
            info = indexed.get(name, {})
            ready, is_active = name in indexed, name == active
            state = "active" if is_active else "ready" if ready else "pending"
            label = "AKTIF" if is_active else "SIAP" if ready else "BELUM DIINDEKS"
            statistics = (
                f"{info.get('chunk_count', 0):,} chunks<br>"
                f"{info.get('total_pages', 0):,} halaman &nbsp; · &nbsp; "
                f"{info.get('removed_header_footer_lines', 0):,} baris dibersihkan"
                if ready else "Menunggu pemrosesan offline"
            )
            with column:
                ticker = html.escape(get_ticker_from_filename(name))
                st.markdown(
                    f'<div class="emiten-card {state}"><div class="emiten-heading">'
                    f'<span class="emiten-ticker">{ticker}</span>'
                    f'<span class="emiten-status">{label}</span></div>'
                    f'<div class="emiten-file">Prospektus {ticker}</div>'
                    f'<div class="emiten-stats">{statistics}</div></div>', unsafe_allow_html=True,
                )
                if st.button("Buka Chat" if is_active else "Pilih & Chat" if ready else "Belum tersedia",
                             key=f"select_{name}", type="primary" if ready else "secondary",
                             disabled=not ready, use_container_width=True):
                    if not is_active:
                        pipeline.set_active_document(name)
                        st.session_state.messages = []
                    st.session_state.pdf_name = name
                    st.session_state.current_page = "Tanya Jawab"
                    st.rerun()

def render_chat():
    st.title("Sistem Tanya Jawab Prospektus IPO")
    pipeline = load_pipeline()
    doc_name = pipeline.get_active_document()
    if not doc_name:
        st.write("Pilih emiten terlebih dahulu untuk mulai bertanya tentang prospektusnya.")
        if st.button("Pilih emiten", type="primary"):
            st.session_state.current_page = "Manajemen Emiten"
            st.rerun()
        return
    heading, action = st.columns([3, 1])
    with heading:
        st.write(f"Emiten: **{get_ticker_from_filename(doc_name)}**")
    with action:
        if st.button("Hapus percakapan", use_container_width=True,
                     disabled=not st.session_state.messages):
            st.session_state.messages = []
            st.rerun()
    st.divider()
    if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        st.error("Kunci API belum tersedia. Atur OPENROUTER_API_KEY pada file .env untuk menggunakan tanya jawab.")
        return
    if not st.session_state.messages:
        st.write("Apa yang ingin Anda ketahui?")
        st.caption("Misalnya: Untuk apa dana hasil IPO akan digunakan?")
    for msg in st.session_state.messages:
        _render_message(msg)
    if question := st.chat_input("Tanyakan sesuatu tentang prospektus ini..."):
        st.session_state.messages.append({"role": "user", "content": question})
        st.rerun()
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        _generate_and_append_response(st.session_state.messages[-1]["content"], pipeline)

def render_about():
    st.markdown(f"""
    ## Tentang Sistem
    
    Aplikasi ini adalah implementasi dari skripsi:
    **"Implementasi Hybrid Retrieval dengan Reciprocal Rank Fusion pada Retrieval-Augmented Generation (RAG) untuk Sistem Tanya Jawab Prospektus IPO"**
    
    Oleh: **Hernando Bun**
    
    ### Arsitektur Sistem
    Dokumen disiapkan melalui proses offline. Pengguna memilih dokumen yang sudah tersedia,
    lalu mengajukan pertanyaan. Jawaban merupakan informasi prospektus, bukan rekomendasi investasi.
    Sistem menggunakan arsitektur RAG lanjutan (Advanced RAG) untuk mengatasi dokumen yang panjang dan kompleks (300-500 halaman).
    
    1. **Parsing, Preprocessing & Chunking**
       - **LlamaParse**: Digunakan untuk mengekstraksi teks dan secara khusus mengisolasi tabel agar struktur baris/kolom tidak hancur.
       - **Preprocessing konservatif**: Unicode, line ending, whitespace, karakter kontrol, serta header/footer berulang dinormalisasi. Simbol keuangan, karakter khusus bermakna, newline, dan struktur tabel tetap dipertahankan.
       - **Structural Chunking**: Narasi dipotong berbasis karakter (*sliding window*), sedangkan tabel dipertahankan utuh secara atomik.
       
    2. **Dual Indexing (Hybrid)**
       - **Dense Retrieval (ChromaDB)**: Menggunakan model embedding `{config.OPENROUTER_EMBEDDING_MODEL}` (Multilingual, 1024 dimensi) via OpenRouter API untuk pencarian berbasis **makna/semantik**.
       - **Sparse Retrieval (BM25Okapi)**: Indeks leksikal berbasis frekuensi kata, sangat unggul untuk mencari angka spesifik (laba, tahun, rasio) di dalam tabel.
       
    3. **Reciprocal Rank Fusion (RRF)**
       - Algoritma reranking tanpa bobot heuristik. Hasil dari Dense dan Sparse digabungkan menggunakan rumus: `Score = 1 / (k + rank)`.
       - RRF secara otomatis mengangkat chunk dokumen yang disetujui relevan oleh *kedua* metode pencarian.
       
    4. **Generation**
       - **LLM**: `{config.OPENROUTER_MODEL_NAME}` melalui OpenRouter API. Model digunakan untuk menyusun jawaban berdasarkan konteks hasil retrieval.
    """)


def _render_message(msg: dict):
    if msg["role"] == "user":
        content = html.escape(str(msg["content"])).replace("\n", "<br>")
        st.markdown(
            f'<div class="user-message"><div class="user-bubble">{content}</div>'
            '<span class="user-avatar" aria-label="Anda"><svg width="20" height="20" viewBox="0 0 24 24" '
            'fill="currentColor" aria-hidden="true"><circle cx="12" cy="8" r="4"/>'
            '<path d="M4 21v-2a8 8 0 0 1 16 0v2z"/></svg></span></div>',
            unsafe_allow_html=True,
        )
        return
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        raw_ans = msg.get("answer", msg.get("content", "")) or ""
        st.markdown(clean_generated_answer(raw_ans))

        pages = msg.get("source_pages", [])
        if pages:
            badges = " ".join(f'<span class="source-badge">Hal. {html.escape(str(p))}</span>' for p in sorted(set(pages)))
            st.markdown(
                f'<div class="answer-meta">📄 Sumber: &nbsp;{badges}</div>',
                unsafe_allow_html=True,
            )

    chunks = msg.get("source_chunks", [])
    if chunks:
        with st.expander(f"Lihat Detail Sumber ({len(chunks)} chunk)", expanded=False):
            if msg.get("sources_are_candidates"):
                st.caption(msg.get("source_note") or "Potongan ini adalah hasil pencarian yang belum memenuhi ambang relevansi 0,02, bukan sumber pendukung jawaban.")
            for i, chunk in enumerate(chunks, 1):
                with st.container(border=True):
                    kind = "Tabel" if chunk.get("chunk_type") == "table" else "Narasi"
                    page = html.escape(str(chunk.get("page_number", "?")))
                    score = chunk.get("rrf_score", 0)
                    dense = chunk.get("dense_rank", 0)
                    sparse = chunk.get("sparse_rank", 0)
                    st.markdown(
                        f'<div class="source-heading"><strong>#{i} — Hal. {page}</strong><span>{kind}</span>'
                        f'<span class="source-score">RRF: {score:.5f} (D:{dense} S:{sparse})</span></div>',
                        unsafe_allow_html=True,
                    )
                    obj = chunk.get("chunk")
                    for field, label in (("section_heading", "Bagian"), ("chapter", "Bab"),
                                         ("table_title", "Judul tabel"), ("financial_unit", "Satuan")):
                        value = getattr(obj, field, "") if obj else chunk.get(field, "")
                        if value:
                            st.caption(f"{label}: {value}")
                    # Native Markdown preserves full tables, links and paragraph formatting.
                    st.markdown(chunk.get("content", ""))


def _generate_and_append_response(question: str, pipeline):
    """Call the pipeline and append the response to session messages."""
    with st.spinner("Mencari informasi dalam prospektus..."):
        try:
            response = pipeline.query(question)

            serialized_chunks = []
            candidates = getattr(response, "retrieval_candidates", [])
            for rc in response.source_chunks or candidates:
                serialized_chunks.append({
                    "chunk":        rc.chunk,
                    "chunk_type":   rc.chunk.chunk_type,
                    "page_number":  rc.chunk.page_number,
                    "content":      rc.chunk.content,
                    "rrf_score":    rc.rrf_score,
                    "dense_rank":   rc.dense_rank,
                    "sparse_rank":  rc.sparse_rank,
                })

            clean_ans = clean_generated_answer(response.answer)
            st.session_state.messages.append({
                "role":         "assistant",
                "content":      clean_ans,
                "answer":       clean_ans,
                "source_pages": response.source_pages,
                "source_chunks":serialized_chunks,
                "sources_are_candidates": bool(candidates and not response.source_chunks),
                "source_note": getattr(response, "retrieval_note", ""),
                "tokens_used":  response.tokens_used,
                "model":        response.model_used,
            })

        except Exception as e:
            from src.services.diagnostics import describe_error
            error_msg = "Permintaan belum berhasil. " + describe_error(e)
            st.session_state.messages.append({
                "role":         "assistant",
                "content":      error_msg,
                "answer":       error_msg,
                "source_pages": [],
                "source_chunks":[],
            })
            logger.exception("Query failed")

    st.rerun()



# ============================================================
#  Main Entry Point (Router)
# ============================================================
def main():
    render_sidebar()
    
    page = st.session_state.current_page
    
    # Explicitly replace the whole page before any slow retrieval/generation.
    # Streamlit otherwise retains stale elements from the previous run until it ends.
    page_slot = st.empty()
    page_slot.empty()
    with page_slot.container():
        if page == "Beranda":
            render_home()
        elif page == "Manajemen Emiten":
            render_management()
        elif page == "Tanya Jawab":
            render_chat()
        elif page == "Tentang":
            render_about()

if __name__ == "__main__":
    main()
