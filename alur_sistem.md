# Alur Sistem Tanya Jawab Prospektus IPO

Dokumen ini menjelaskan proses sistem secara menyeluruh: konfigurasi, ingestion
PDF, parsing, preprocessing, chunking, indexing, retrieval, query expansion & synonym mapping,
generasi jawaban, sitasi, evaluasi, penyimpanan, dan penanganan kegagalan. Dokumen ini mengikuti
implementasi aktual pada folder `Program/` dengan arsitektur **Hybrid Retrieval (BAAI/bge-m3 + BM25Okapi)**,
**Reciprocal Rank Fusion (RRF)**, dan generator **Qwen (qwen/qwen3.7-flash) via OpenRouter API**.

---

## 1. Tujuan dan Batas Sistem

Sistem menjawab pertanyaan tentang satu prospektus IPO yang dipilih pengguna.
Jawaban tidak dibentuk hanya dari pengetahuan parametrik LLM. Sistem terlebih
dahulu mengambil potongan teks atau tabel yang relevan dari prospektus terkait, kemudian mengirimkan
potongan tersebut bersama pertanyaan ke LLM.

Batas utamanya adalah sebagai berikut:

- Sumber pengetahuan utama adalah teks narasi dan tabel hasil parsing PDF prospektus.
- Pencarian selalu dibatasi pada `doc_source` dokumen aktif agar data antar
  emiten tidak tercampur.
- Gambar, diagram murni, tanda tangan, dan hubungan visual non-tabel tidak dianalisis secara khusus.
- Sitasi halaman berasal dari metadata hasil parsing dan retrieval, bukan dari
  halaman yang ditebak oleh LLM.
- Prompt anti-halusinasi dan aturan konversi nominal finansial menjamin jawaban terikat ketat pada fakta dokumen.

---

## 2. Ringkasan Arsitektur

```text
                         PIPELINE OFFLINE / INGESTION

PDF prospektus (data/raw/prospectuses/)
    |
    v
Validasi file + SHA-256 + pemeriksaan registry/cache
    |
    v
LlamaParse (baku: strict_llamaparse) atau pdfplumber (sesuai PARSER_MODE)
    |
    +--> teks naratif per halaman
    +--> tabel Markdown per halaman
    +--> metadata nomor halaman
    |
    v
Preprocessing konservatif per halaman (src/ingestion/preprocessor.py)
    +--> Unicode NFC + line ending LF
    +--> karakter kontrol/invisible dibuang
    +--> whitespace dinormalisasi, newline dipertahankan
    +--> header/footer berulang dihapus dengan audit (RepeatRatio >= 0.65)
    |
    v
Structural chunking narasi + chunk tabel atomik + metadata (src/ingestion/chunker.py)
    |
    +--> BAAI/bge-m3 (OpenRouter / Local) --> embedding ternormalisasi (1024-dim) --> ChromaDB (Dense Index)
    |
    +--> tokenisasi finansial numerik --> BM25Okapi --> pickle (Sparse Index)
    |
    v
Registry dokumen dan statistik ingestion (data/registry/index_registry.json)


                           PIPELINE ONLINE / QUERY

Pertanyaan pengguna + dokumen aktif
    |
    v
Query Expansion (Definisi + Kamus Sinonim Istilah Finansial, Korporasi, & Bahasa Santai)
    |
    +--> Dense Retrieval BGE-M3/ChromaDB (top 10 kandidat)
    |
    +--> Sparse Retrieval BM25 (top 10 kandidat)
    |
    v
Reciprocal Rank Fusion (RRF, k=60) --> fusi peringkat kandidat
    |
    v
Ambang Relevansi MIN_RRF_SCORE = 0.02 (atau konfigurasi custom)
    |
    +--> di bawah ambang: "Informasi tersebut tidak tersedia di dokumen prospektus ini."
    |
    +--> memenuhi ambang: kirim top-6 chunk ke Generator
                               |
                               v
               Qwen (qwen/qwen3.7-flash via OpenRouter API)
               [effort: none reasoning bypass]
                               |
                               v
       Jawaban natural + sitasi halaman + preview sumber + token info
```

---

## 3. Konfigurasi Utama

Konfigurasi terpusat berada di `config.py`. Nilai rahasia dan parameter dinamis dibaca dari `.env`.

| Variabel | Nilai Baku / Aktual | Fungsi |
|---|---|---|
| `OPENROUTER_API_KEY` | *(Tersimpan di .env)* | Kredensial API OpenRouter untuk generasi jawaban, embedding, dan evaluator RAGAS |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Endpoint OpenRouter yang kompatibel dengan OpenAI API |
| `OPENROUTER_MODEL_NAME` | `qwen/qwen3.7-flash` | ID model Qwen di OpenRouter untuk generasi jawaban |
| `OPENROUTER_TEMPERATURE` | `0.1` | Suhu generasi rendah untuk menjamin faktualitas dan reproduktibilitas |
| `OPENROUTER_MAX_TOKENS` | `2048` | Batas maksimum token keluaran jawaban |
| `EMBEDDING_PROVIDER` | `openrouter` | Provider embedding: `openrouter` (BGE-M3 via API) atau `local` (SentenceTransformer) |
| `OPENROUTER_EMBEDDING_MODEL` | `baai/bge-m3` | ID model embedding BGE-M3 di OpenRouter |
| `EMBEDDING_BATCH_SIZE` | `32` | Ukuran batch embedding per permintaan API |
| `LLAMA_CLOUD_API_KEY` | *(Tersimpan di .env)* | Kredensial LlamaParse untuk parsing PDF berbasis multimodal layout-aware |
| `PARSER_MODE` | `strict_llamaparse` | Menentukan parser (`strict_llamaparse`, `llamaparse_fallback`, `pdfplumber`) |
| `PREPROCESSING_ENABLED` | `true` | Mengaktifkan preprocessing teks hasil parser |
| `PREPROCESS_REMOVE_REPEATED_EDGES` | `true` | Mengaktifkan deteksi header/footer berulang lintas halaman |
| `PREPROCESS_EDGE_LINES` | `3` | Banyak baris non-kosong di zona atas/bawah yang diaudit |
| `PREPROCESS_REPEAT_RATIO` | `0.65` | Proporsi minimum kemunculan baris berulang |
| `PREPROCESS_MIN_REPEAT_PAGES` | `3` | Minimum halaman sebelum penghapusan repetisi aktif |
| `PREPROCESS_MAX_BLANK_LINES` | `2` | Maksimum baris kosong berurutan |
| `MIN_RRF_SCORE` | `0.02` | Ambang relevansi minimum gabungan RRF sebelum LLM dipanggil |
| `RAGAS_EVALUATOR_MODEL` | `qwen/qwen3.7-flash` | Model LLM penilai evaluasi RAGAS |
| `RAGAS_EVALUATOR_MAX_TOKENS` | `2048` | Batas keluaran evaluator RAGAS |

Konfigurasi tetap pada kode:

| Komponen | Nilai | Keterangan |
|---|---:|---|
| Model embedding | `BAAI/bge-m3` | Dense multilingual & financial representations |
| Dimensi vektor embedding | 1024 | Panjang vektor representasi semantik |
| Ukuran chunk narasi | 800 karakter | Panjang jendela potongan teks narasi |
| Overlap narasi | 150 karakter | Irisan konteks antar-chunk narasi berurutan |
| Panjang minimum chunk | 50 karakter | Mengabaikan potongan terlalu pendek yang tidak bermakna |
| Kandidat dense | 10 | Top-K potongan dari ChromaDB |
| Kandidat sparse | 10 | Top-K potongan dari BM25Okapi |
| Konteks akhir | 6 chunk | Jumlah potongan akhir yang dikirim ke Generator LLM |
| Konstanta RRF ($k$) | 60 | Faktor kehalusan perankingan Reciprocal Rank Fusion |
| Chroma collection | `ipo_prospectus` | Koleksi database vektor ChromaDB |
| Versi cache parser | `v3` | Identifier versi serialisasi hasil parsing |
| Versi skema indeks | `v3-preprocessing-v1` | Identifier kompatibilitas indeks |

---

## 4. Startup Aplikasi

1. Pengguna menjalankan Streamlit melalui `streamlit run app.py`.
2. `config.py` memuat konfigurasi dari `.env` dan memastikan folder `data/raw/prospectuses`, `data/indexes/chroma_db`, `data/registry`, dll. tersedia.
3. `RAGPipeline` diinisialisasi sebagai singleton pada sesi Streamlit.
4. `Embedder` mengonfigurasi akses ke `BAAI/bge-m3` (OpenRouter API atau Local).
5. `DualIndexer` memuat registry `data/registry/index_registry.json`, membuka ChromaDB collection `ipo_prospectus`, dan memuat indeks sparse `bm25_index.pkl`.
6. Generator `RAGGenerator` diinisialisasi menggunakan `ChatOpenAI` dengan endpoint OpenRouter dan parameter `effort: "none"`.
7. Antarmuka Streamlit menampilkan halaman utama dengan emiten aktif terpilih di urutan teratas.

---

## 5. Pipeline Ingestion Dokumen

### 5.1 Pemilihan dan Validasi Awal
Dokumen disimpan pada `data/raw/prospectuses/`. Saat ingestion dimulai, pipeline menggunakan nama berkas sebagai `doc_source` (misal `JECX_2026.pdf`).

- Jika dokumen sudah terindeks di registry dan `force=False`, ingestion dilewati.
- Jika `force=True`, dokumen di-ingest ulang dan entri indeks lama digantikan secara atomik.
- Progres dilaporkan ke UI dengan tahapan: `parsing`, `preprocessing`, `chunking`, `embedding`, `indexing`, dan `done`.

### 5.2 Identitas Sumber dan Cache Parsing
Parser menghitung SHA-256 seluruh isi PDF. Nama cache dibentuk dari:
```text
<nama_aman>_<12-karakter-sha256>_<parser_mode>_<cache_version>.json
```
Cache menggunakan `PARSER_CACHE_VERSION = "v3"`. Cache hanya dipakai jika SHA-256 sumber, mode parser, dan konfigurasi preprocessing cocok.

### 5.3 Mode Parser
1. **`strict_llamaparse`**: Mode baku penelitian. PDF dikirim ke LlamaParse Cloud dengan instruksi tata letak tabel dan teks bahasa Indonesia untuk menghasilkan format Markdown presisi tinggi.
2. **`llamaparse_fallback`**: Mencoba LlamaParse; jika kuota habis atau jaringan gagal, otomatis beralih ke pdfplumber.
3. **`pdfplumber`**: Ekstraksi lokal murni tanpa koneksi internet.

### 5.4 Pemisahan Tabel dari Markdown
Blok tabel Markdown diekstrak dan disimpan pada objek `ParsedPage.tables`, sedangkan teks narasi disimpan pada `ParsedPage.text_content`.

---

## 6. Preprocessing Teks yang Diterapkan

Preprocessing berada di `src/ingestion/preprocessor.py` dan dipanggil oleh `PDFParser` sebelum chunking:

| Proses | Status Implementasi | Keterangan |
|---|---|---|
| Ekstraksi BAB/subbab | Diterapkan | Mendeteksi format Markdown `#`, `BAB I`, dan ALL-CAPS |
| Pemisahan tabel | Diterapkan | Tabel Markdown dipisahkan dari narasi dan diperlakukan atomik |
| Cleaning header/footer | Diterapkan | Menghapus baris berulang lintas halaman (RepeatRatio >= 0.65) |
| Preservasi simbol keuangan | Diterapkan | Simbol penting `%`, `Rp`, `(`, `)`, `,`, `.`, `|` dipertahankan |
| Penghapusan artefak | Diterapkan | Karakter kontrol, BOM, NUL, dan soft-hyphen dibuang |
| Retensi baris & newline | Diterapkan | Struktur baris tabel dan heading tidak dirusak |
| Normalisasi whitespace | Diterapkan | Normalisasi Unicode NFC, LF line endings, dan spasi ganda |
| Normalisasi angka BM25 | Diterapkan | `1.250.000` -> `1250000`, `18,7%` -> `187_persen`, `Rp 780.000` -> `rp780000 780000` |

---

## 7. Strategi Chunking

Chunker (`src/ingestion/chunker.py`) menghasilkan objek `Chunk` terstruktur:

1. **Sliding Window Narasi**:
   - Panjang chunk: 800 karakter.
   - Overlap: 150 karakter.
   - Setiap sub-chunk mewarisi heading konteks: `[BAB IV: RENCANA PENGGUNAAN DANA...]`.

2. **Atomic Table Chunking**:
   - Satu tabel Markdown dipertahankan sebagai satu kesatuan chunk utuh tanpa dipotong sliding window.

3. **Cross-Page Table Merging**:
   - Mendeteksi tabel lanjutan (*orphan table*) pada halaman berikutnya dan menggabungkannya dengan tabel induk di halaman sebelumnya.

4. **Metadata Enrichment**:
   - Setiap chunk diperkaya dengan `doc_source`, `page_number`, `chunk_type`, `chapter`, `subchapter`, `table_title`, dan `financial_unit`.

---

## 8. Indexing Ganda (Dual Indexing)

1. **Dense Index (ChromaDB)**:
   - Model `BAAI/bge-m3` menghasilkan vektor 1024-dimensi.
   - Vektor dinormalisasi L2 sebelum disimpan ke ChromaDB.
   - Koleksi: `ipo_prospectus` dengan filter metadata `doc_source`.

2. **Sparse Index (BM25Okapi)**:
   - Menggunakan tokenisasi khusus dengan normalisasi angka keuangan Indonesia.
   - Disimpan ke `data/indexes/bm25_index.pkl`.

---

## 9. Query Expansion & Kamus Sinonim Terintegrasi

Sebelum dikirim ke retriever, pertanyaan diproses melalui `_expand_query` di `src/services/pipeline.py`:
1. **Definisi Istilah**: Jika pertanyaan mencari pengertian ("apa itu", "definisi", "maksud"), kueri diperluas dengan istilah kamus prospektus ("definisi dan singkatan", "berarti", "artinya").
2. **Kamus Sinonim Keuangan & Bahasa Santai (Slang)**:
   - *Laba / Profit*: `keuntungan`, `untung`, `cuan`, `profit`, `marjin` $\rightarrow$ `laba laba bersih laba tahun berjalan`
   - *Pendapatan / Omzet*: `omset`, `omzet`, `pemasukan`, `penjualan` $\rightarrow$ `pendapatan penjualan neto pendapatan usaha`
   - *Utang / Kewajiban*: `utang`, `hutang`, `pinjaman`, `kewajiban` $\rightarrow$ `liabilitas pinjaman bank fasilitas kredit`
   - *Aset / Harta*: `harta`, `kekayaan`, `aktiva` $\rightarrow$ `aset aset lancar aset tidak lancar total aset`
   - *Kerugian / Beban*: `kerugian`, `rugi`, `boncos`, `minus` $\rightarrow$ `rugi tahun berjalan beban pokok pendapatan`
   - *Pimpinan / Manajemen*: `bos`, `pimpinan`, `owner`, `pemilik`, `direktur`, `komisaris` $\rightarrow$ `direksi direktur utama dewan komisaris pemegang saham pengendali`
   - *IPO & Saham*: `duit`, `dana`, `saham publik`, `dividen` $\rightarrow$ `rencana penggunaan dana modal kerja belanja modal capex kebijakan dividen`

---

## 10. Hybrid Retrieval dan Reciprocal Rank Fusion (RRF)

1. **Dense Search**: Mengambil Top-10 kandidat chunk berdasarkan Cosine Similarity dari ChromaDB dengan filter dokumen aktif.
2. **Sparse Search**: Mengambil Top-10 kandidat chunk berdasarkan BM25Okapi score dengan filter dokumen aktif.
3. **Reciprocal Rank Fusion**:
   $$RRF(d) = \sum_{m \in \{\text{Dense}, \text{Sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$$
4. **Relevance Threshold Gating**:
   - Jika skor RRF terbaik $< \text{MIN\_RRF\_SCORE}$, sistem langsung mengembalikan respon *"Informasi tersebut tidak tersedia di dokumen prospektus ini."* tanpa memanggil LLM.
   - Jika $\ge \text{MIN\_RRF\_SCORE}$, Top-6 chunk diteruskan sebagai konteks ke generator LLM.

---

## 11. Generasi Jawaban via OpenRouter (Qwen 3.7 Flash)

### 11.1 Integrasi Client
Menggunakan `langchain_openai.ChatOpenAI` yang diarahkan ke OpenRouter endpoint (`https://openrouter.ai/api/v1`) dengan model `qwen/qwen3.7-flash` dan `extra_body={"reasoning": {"effort": "none"}}` untuk memotong latensi reasoning.

### 11.2 Format Injeksi Konteks
Setiap chunk diberi tag metadata terstruktur:
```text
[KONTEKS 1 | Halaman 62 | TABEL KEUANGAN | Relevansi: 0.0328]
[BAB V: IKHTISAR DATA KEUANGAN PENTING]
(Satuan: jutaan Rupiah)
<isi tabel Markdown>
```

### 11.3 Aturan Generasi Jawaban
- Menjawab langsung dan natural tanpa format heading pengantar kaku ("Ringkasan:", "Detail:").
- Mengonversi nominal keuangan secara penuh (misal `712.031` pada tabel jutaan rupiah dikonversi menjadi `Rp712.031.000.000 (sekitar Rp712 miliar)`).
- Menolak menjawab jika informasi tidak ditemukan dalam konteks dokumen.

---

## 12. Tampilan Aplikasi Streamlit (`app.py`)

- **Sidebar Dinamis**:
  - Tombol sembunyikan/kecilkan sidebar di bagian atas dan tombol panah buka mengambang (*expand*) di pojok kiri atas layar.
  - Menu navigasi bersih: Beranda, Manajemen Emiten, Tanya Jawab, dan Tentang.
  - Badge emiten aktif dan status kesiapan indeks.
- **Manajemen Emiten**:
  - Emiten aktif otomatis diurutkan di posisi paling atas (#1).
  - Kartu emiten responsif dengan tombol aksi terstandarisasi (`💬 Pilih & Chat` / `💬 Tanya Jawab` dan `🗑️` hapus indeks).
- **Tanya Jawab (Chat Interface)**:
  - Header emiten terintegrasi dengan tombol `🗑️ Hapus Chat`.
  - Bubble chat interaktif dengan sitasi nomor halaman (`📄 Sumber: Hal. X, Y`).
  - Expander rincian 6 chunk referensi lengkap beserta tipe potongan dan skor RRF.

---

## 13. Evaluasi Sistem Lengkap (Sesuai Bab Skripsi)

Sistem evaluasi mencakup 4 pilar pengujian:

### 13.1 Evaluasi Retrieval Multi-Konfigurasi (`evaluation/evaluate_retrieval.py`)
Otomatis membandingkan 3 konfigurasi retrieval:
1. **Sparse BM25 Only** (`dense_k: 0, sparse_k: 10, final_k: 6`)
2. **Dense BGE-M3 Only** (`dense_k: 10, sparse_k: 0, final_k: 6`)
3. **Hybrid RRF** (`dense_k: 10, sparse_k: 10, final_k: 6`)

Metrik yang dihitung:
- **Precision@K**: $P@1, P@3, P@6$
- **Recall@K**: $R@1, R@3, R@6$
- **Mean Reciprocal Rank (MRR)**

### 13.2 Evaluasi RAGAS (`evaluation/evaluate_ragas.py` & `evaluation/ragas_evaluator.py`)
Mengevaluasi 4 metrik standar RAGAS menggunakan model judge `qwen/qwen3.7-flash` dan embedding adapter `baai/bge-m3`:
- **Faithfulness**: Kejujuran faktual jawaban terhadap konteks (anti-halusinasi).
- **Answer Relevancy**: Kesesuaian jawaban dengan maksud pertanyaan pengguna.
- **Context Precision**: Kualitas perankingan chunk relevan di posisi teratas.
- **Context Recall**: Kelengkapan informasi ground truth di dalam konteks retrieval.

### 13.3 Evaluasi Generasi Deterministik
- **Exact Match (EM)**
- **Token F1 Score**
- **Numerical Accuracy** (toleransi 5% untuk nominal finansial dan exact match untuk tahun).
- **Citation Page Accuracy**

### 13.4 Black Box Testing & Human Evaluation
- **Black Box Testing**: Pengujian fungsionalitas UI Streamlit (upload, switching emiten, out-of-scope query rejection, reset chat).
- **Human Evaluation**: Pengujian pengguna skala Likert 1–5 untuk mengukur akurasi faktual, ketepatan konversi angka, kejelasan bahasa, dan kenyamanan antarmuka.

---

## 14. Struktur Penyimpanan dan Berkas

| Folder / Berkas | Fungsi |
|---|---|
| `app.py` | Antarmuka pengguna Streamlit (Responsive & Modern) |
| `config.py` | Konfigurasi terpusat sistem |
| `src/ingestion/` | Parser PDF (LlamaParse/pdfplumber), Preprocessor, dan Chunker |
| `src/retrieval/` | Embedder (BGE-M3), Indexer, Tokenizer Finansial, dan Hybrid Retriever |
| `src/generation/` | Generator jawaban via OpenRouter (Qwen 3.7 Flash) |
| `src/services/` | Pipeline orkestrasi RAG & Kamus Ekspansi Kueri |
| `data/raw/prospectuses/` | Berkas PDF prospektus IPO |
| `data/parsed/` | Cache JSON hasil parsing LlamaParse/pdfplumber |
| `data/indexes/chroma_db/` | Database vektor ChromaDB |
| `data/indexes/bm25_index.pkl` | Indeks sparse BM25 |
| `data/registry/index_registry.json` | Status dan metadata indeks per dokumen |
| `data/evaluation/` | Dataset evaluasi (questions & ground truth) |
| `evaluation/` | Modul pengujian retrieval (3 konfigurasi), generasi, dan RAGAS |
| `docs/technical/` | Dokumentasi teknis & lampiran contoh perhitungan |
