# DOKUMEN LAMPIRAN CONTOH PERHITUNGAN SISTEM TANYA JAWAB PROSPEKTUS IPO

Dokumen ini memuat dokumentasi matematis, sampel data, dan contoh perhitungan langkah demi langkah (*step-by-step mathematical walkthrough*) untuk seluruh tahapan arsitektur sistem Tanya Jawab Dokumen Prospektus IPO berbasis **Retrieval-Augmented Generation (RAG)** dengan **Hybrid Search (BAAI/bge-m3 + BM25Okapi)** dan **Reciprocal Rank Fusion (RRF)**.

---

## DAFTAR ISI

1. [Sampel Data Nyata Prospektus IPO](#1-sampel-data-nyata-prospektus-ipo)
2. [Lampiran 1: Contoh & Perhitungan Preprocessing Teks](#2-lampiran-1-contoh--perhitungan-preprocessing-teks)
3. [Lampiran 2: Contoh & Perhitungan Pemotongan Teks (Chunking)](#3-lampiran-2-contoh--perhitungan-pemotongan-teks-chunking)
4. [Lampiran 3: Contoh & Perhitungan Vektor Dense Embedding (BAAI/bge-m3)](#4-lampiran-3-contoh--perhitungan-vektor-dense-embedding-baaibge-m3)
5. [Lampiran 4: Contoh & Perhitungan Tokenisasi & Sparse Retrieval (BM25Okapi)](#5-lampiran-4-contoh--perhitungan-tokenisasi--sparse-retrieval-bm25okapi)
6. [Lampiran 5: Contoh & Perhitungan Hybrid Retrieval & Reciprocal Rank Fusion (RRF)](#6-lampiran-5-contoh--perhitungan-hybrid-retrieval--reciprocal-rank-fusion-rrf)
7. [Lampiran 6: Contoh Prompt Engineering & Generasi Jawaban LLM (Qwen)](#7-lampiran-6-contoh-prompt-engineering--generasi-jawaban-llm-qwen)
8. [Lampiran 7: Contoh Perhitungan Evaluasi Sistem Lengkap (Sesuai Bab 2.2.12)](#8-lampiran-7-contoh-perhitungan-evaluasi-sistem-lengkap-sesuai-bab-2212)
   - [8.1 Evaluasi Retrieval (Precision@K, Recall@K, MRR)](#81-evaluasi-retrieval-precisionk-recallk-mrr)
   - [8.2 Evaluasi RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall)](#82-evaluasi-ragas)
   - [8.3 Evaluasi Black Box Testing](#83-evaluasi-black-box-testing)
   - [8.4 Evaluasi Human Evaluation (Pengujian Pengguna)](#84-evaluasi-human-evaluation)

---

## 1. Sampel Data Nyata Prospektus IPO

Berikut adalah sampel data nyata yang diambil dari dokumen prospektus PT Berkah Beton Sadaya Tbk (**BACH_2026.pdf**) dan PT Multi Medika Internasional Tbk (**MMI/EMMI_2026.pdf**).

### Sampel 1: Teks Narasi (Profil & Rencana Penggunaan Dana - Hal. 42)
```text
BAB IV: RENCANA PENGGUNAAN DANA HASIL PENAWARAN UMUM PERDANA SAHAM

Seluruh dana yang diperoleh dari hasil Penawaran Umum Perdana Saham ini, setelah dikurangi biaya-biaya emisi saham, akan digunakan oleh Perseroan untuk:
1. Sekitar 65,0% (enam puluh lima koma nol persen) akan digunakan untuk belanja modal (capital expenditure/capex) berupa pembelian dua unit batching plant baru di wilayah Subang dan Cirebon.
2. Sekitar 35,0% (tiga puluh lima koma nol persen) akan digunakan untuk modal kerja (operational expenditure/opex) guna mendukung kegiatan operasional batching plant baru tersebut.
```

### Sampel 2: Tabel Laporan Posisi Keuangan / Neraca (Hal. 214)
```text
BAB V: IKHTISAR DATA KEUANGAN PENTING
PT BERKAH BETON SADAYA TBK
LAPORAN POSISI KEUANGAN
Per 31 Desember 2025, 2024, dan 2023
(Dinyatakan dalam jutaan Rupiah, kecuali dinyatakan lain)

| Akun / Pos Keuangan | 31 Des 2025 | 31 Des 2024 | 31 Des 2023 |
| :--- | :---: | :---: | :---: |
| Aset Lancar | 418.520 | 430.110 | 385.200 |
| Aset Tidak Lancar | 293.511 | 319.956 | 316.206 |
| **TOTAL ASET** | **712.031** | **750.066** | **701.406** |
| Liabilitas Jangka Pendek | 125.400 | 140.250 | 132.100 |
| Liabilitas Jangka Panjang | 45.200 | 48.300 | 50.100 |
| **TOTAL LIABILITAS** | **170.600** | **188.550** | **182.200** |
| **TOTAL EKUITAS** | **541.431** | **561.516** | **519.206** |
| **TOTAL LIABILITAS DAN EKUITAS** | **712.031** | **750.066** | **701.406** |
```

### Sampel 3: Cuplikan Tabel Terpotong Antar Halaman (Cross-Page Table - Hal. 105-106)
**Halaman 105:**
```text
BAB V: ANALISIS DAN PEMBAHASAN MANAJEMEN
Tabel Rincian Penjualan Berdasarkan Wilayah Geografis
(dalam jutaan Rupiah)

| Wilayah Operasional | 2025 | 2024 |
| :--- | :---: | :---: |
| Jawa Barat | 310.450 | 280.120 |
| DKI Jakarta | 185.200 | 160.400 |
```
**Halaman 106 (Tabel Lanjutan / Orphan Table):**
```text
| Banten | 95.300 | 88.200 |
| Jawa Tengah | 65.100 | 55.400 |
| **Total Pendapatan** | **656.050** | **584.120** |
```

---

## 2. Lampiran 1: Contoh & Perhitungan Preprocessing Teks

Modul Preprocessing (`src/ingestion/preprocessor.py`) melakukan pembersihan artefak ekstraksi PDF dengan tetap menjaga simbol-simbol penting laporan keuangan.

### Langkah-langkah Preprocessing:

1. **Pembersihan Repeated Edge Lines (Header/Footer Berulang):**
   - Mendeteksi 3 baris teratas dan 3 baris terbawah pada tiap halaman.
   - Menghitung rasio kemunculan baris identik di seluruh dokumen:
     $$	ext{RepeatRatio}(L) = rac{	ext{Count}(L)}{	ext{TotalPages}}$$
   - Jika $	ext{RepeatRatio}(L) \ge 0.65$ dan $	ext{Count}(L) \ge 3$, baris tersebut diidentifikasi sebagai header/footer dan dihapus, **kecuali** baris tersebut diawali kata kunci `BAB` atau judul section.
   
   *Contoh:*
   ```text
   Baris: "PT BERKAH BETON SADAYA TBK - PROSPEKTUS IPO 2026"
   Muncul di: 210 dari 230 halaman (Rasio = 210/230 = 0.913 >= 0.65)
   Tindakan: DIHAPUS DARI TIAP HALAMAN.
   ```

2. **Normalisasi Whitespace & Karakter Khusus:**
   - Spasi ganda diubah menjadi spasi tunggal.
   - Baris kosong berlebih dibatasi maksimal 2 newline.
   - Simbol penting keuangan dipertahankan: `%`, `Rp`, `(`, `)`, `,`, `.`, `|`, `:`, `/`

### Hasil Preprocessing (Sebelum vs Sesudah):
```text
[SEBELUM]
PT BERKAH BETON SADAYA TBK - PROSPEKTUS IPO 2026

BAB IV: RENCANA PENGGUNAAN DANA     


Sekitar  65,0%   digunakan untuk belanja modal (capex)...

Hal. 42
[SESUDAH]
BAB IV: RENCANA PENGGUNAAN DANA

Sekitar 65,0% digunakan untuk belanja modal (capex)...
```

---

## 3. Lampiran 2: Contoh & Perhitungan Pemotongan Teks (Chunking)

Modul Chunking (`src/ingestion/chunker.py`) membagi dokumen menggunakan metode *Structural Heading Chunking* dan *Atomic Table Isolation*.

### Parameter Chunking:
- **Ukuran Chunk Narasi ($S$):** 800 karakter
- **Panjang Overlap ($O$):** 150 karakter
- **Panjang Minimum Chunk:** 50 karakter

### Mekanisme Pemotongan & Pengayaan:

1. **Deteksi Heading & Bab:**
   - Heading: `BAB IV: RENCANA PENGGUNAAN DANA HASIL PENAWARAN UMUM PERDANA SAHAM`
   - Ekstraksi Bab (`chapter`): `BAB IV`
   - Ekstraksi Sub-bab (`subchapter`): `RENCANA PENGGUNAAN DANA HASIL PENAWARAN UMUM PERDANA SAHAM`

2. **Sliding Window Narasi:**
   - Teks narasi dipotong per 800 karakter. Potongan berikutnya dimulai pada indeks `(800 - 150) = 650`.
   - Setiap sub-chunk diawali dengan header konteks: `[BAB IV: RENCANA PENGGUNAAN DANA...]`.

3. **Cross-Page Table Merging:**
   - Tabel di Hal. 106 dideteksi sebagai *orphan table* (tidak memiliki baris header `| Wilayah | ... |`).
   - Sistem menggabungkan body tabel Hal. 106 ke dalam tabel Hal. 105:
   ```markdown
   | Wilayah Operasional | 2025 | 2024 |
   | :--- | :---: | :---: |
   | Jawa Barat | 310.450 | 280.120 |
   | DKI Jakarta | 185.200 | 160.400 |
   | Banten | 95.300 | 88.200 |
   | Jawa Tengah | 65.100 | 55.400 |
   | **Total Pendapatan** | **656.050** | **584.120** |
   ```

4. **Metadata Enrichment (Penyematan Metadata):**
   Objek chunk yang disimpan ke basis data memiliki struktur lengkap:
   ```python
   Chunk(
       chunk_id="tbl_BACH_2026_00214_01",
       content="[BAB V: IKHTISAR DATA KEUANGAN PENTING]
(Satuan: jutaan Rupiah)

| Akun | 31 Des 2025 | ...
| TOTAL ASET | 712.031 | ...",
       chunk_type="table",
       page_number=214,
       section_heading="BAB V: IKHTISAR DATA KEUANGAN PENTING",
       doc_source="BACH_2026.pdf",
       chapter="BAB V",
       subchapter="IKHTISAR DATA KEUANGAN PENTING",
       table_title="LAPORAN POSISI KEUANGAN",
       financial_unit="jutaan Rupiah",
       char_count=642
   )
   ```

---

## 4. Lampiran 3: Contoh & Perhitungan Vektor Dense Embedding (BAAI/bge-m3)

Model embedding yang digunakan adalah `BAAI/bge-m3`, menghasilkan vektor berdimensi $d = 1024$.

### Rumus Matematis Cosine Similarity:
Diberikan vektor pertanyaan $\mathbf{q} \in \mathbb{R}^{1024}$ dan vektor dokumen chunk $\mathbf{d} \in \mathbb{R}^{1024}$:

1. **Dot Product (Hasil Kali Titik):**
   $$\mathbf{q} \cdot \mathbf{d} = \sum_{i=1}^{1024} q_i \cdot d_i$$

2. **Norma Euclidean (L2 Norm):**
   $$\|\mathbf{q}\|_2 = \sqrt{\sum_{i=1}^{1024} q_i^2}, \quad \|\mathbf{d}\|_2 = \sqrt{\sum_{i=1}^{1024} d_i^2}$$

3. **Cosine Similarity:**
   $$	ext{Sim}(\mathbf{q}, \mathbf{d}) = rac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$

### Contoh Perhitungan Numerik (Simulasi 5 Dimensi Pertama):
Pertanyaan: *"Berapa total aset BACH pada tahun 2025?"*

- Vektor Query $\mathbf{q} = [0.124, 0.451, -0.231, 0.089, 0.312]$
- Vektor Chunk A (Tabel Total Aset Hal. 214) $\mathbf{d}_A = [0.118, 0.462, -0.215, 0.095, 0.301]$
- Vektor Chunk B (Profil Direksi Hal. 85) $\mathbf{d}_B = [-0.052, 0.112, 0.341, -0.120, 0.045]$

**Perhitungan untuk Chunk A:**
$$\mathbf{q} \cdot \mathbf{d}_A = (0.124 	imes 0.118) + (0.451 	imes 0.462) + (-0.231 	imes -0.215) + (0.089 	imes 0.095) + (0.312 	imes 0.301)$$
$$\mathbf{q} \cdot \mathbf{d}_A = 0.01463 + 0.20836 + 0.04967 + 0.00846 + 0.09391 = 0.37503$$
$$\|\mathbf{q}\| = \sqrt{0.124^2 + 0.451^2 + (-0.231)^2 + 0.089^2 + 0.312^2} = \sqrt{0.37990} pprox 0.61636$$
$$\|\mathbf{d}_A\| = \sqrt{0.118^2 + 0.462^2 + (-0.215)^2 + 0.095^2 + 0.301^2} = \sqrt{0.37327} pprox 0.61096$$
$$	ext{Sim}(\mathbf{q}, \mathbf{d}_A) = rac{0.37503}{0.61636 	imes 0.61096} = rac{0.37503}{0.37657} = \mathbf{0.9959} \quad (	ext{Sangat Relevan})$$

**Perhitungan untuk Chunk B:**
$$\mathbf{q} \cdot \mathbf{d}_B = -0.00645 + 0.05051 - 0.07877 - 0.01068 + 0.01404 = -0.03135$$
$$	ext{Sim}(\mathbf{q}, \mathbf{d}_B) = rac{-0.03135}{0.61636 	imes 0.38521} = \mathbf{-0.1320} \quad (	ext{Tidak Relevan})$$

---

## 5. Lampiran 4: Contoh & Perhitungan Tokenisasi & Sparse Retrieval (BM25Okapi)

Modul Sparse Retrieval (`src/retrieval/indexer.py` & `tokenizer.py`) menggunakan algoritma **BM25Okapi** dengan normalisasi angka keuangan Indonesia.

### Normalisasi Angka (`_normalize_numbers`):
- `1.250.000.000` $ightarrow$ `1250000000`
- `65,0%` $ightarrow$ `650_persen 65_persen`
- `Rp 712.031` $ightarrow$ `rp712031 712031`

### Rumus BM25Okapi:
Untuk query $Q = \{q_1, q_2, \dots, q_n\}$ terhadap dokumen $D$:
$$	ext{Score}_{BM25}(D, Q) = \sum_{i=1}^{n} 	ext{IDF}(q_i) \cdot rac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot rac{|D|}{	ext{avgdl}}ight)}$$

Keterangan Parameter:
- $k_1 = 1.5$ (skala saturasi term frequency)
- $b = 0.75$ (derajat normalisasi panjang dokumen)
- $|D|$: Panjang dokumen dalam jumlah token
- $	ext{avgdl}$: Panjang rata-rata seluruh dokumen dalam korpus ($= 120$ token)
- $N$: Total dokumen dalam koleksi ($= 500$ chunk)
- $n(q_i)$: Jumlah dokumen yang mengandung token $q_i$

$$	ext{IDF}(q_i) = \ln \left( rac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1 ight)$$

### Contoh Perhitungan Nyata:
Query: *"total aset 712031"* $ightarrow$ Token: `["total", "aset", "712031"]`
Chunk Evaluasi: Chunk Hal. 214 ($|D| = 150$ token)

| Token $q_i$ | $n(q_i)$ | $	ext{IDF}(q_i)$ | $f(q_i, D)$ | $	ext{TF\_Weight}$ | $	ext{Skor Token}$ |
|---|---|---|---|---|---|
| `total` | 80 | $\ln(rac{500-80+0.5}{80+0.5}+1) = \ln(6.217) = 1.827$ | 4 | $rac{4 	imes 2.5}{4 + 1.5(0.25 + 0.75 	imes rac{150}{120})} = rac{10}{5.781} = 1.729$ | $1.827 	imes 1.729 = \mathbf{3.159}$ |
| `aset` | 45 | $\ln(rac{500-45+0.5}{45+0.5}+1) = \ln(11.00) = 2.398$ | 3 | $rac{3 	imes 2.5}{3 + 1.5(1.1875)} = rac{7.5}{4.781} = 1.568$ | $2.398 	imes 1.568 = \mathbf{3.760}$ |
| `712031` | 2 | $\ln(rac{500-2+0.5}{2+0.5}+1) = \ln(200.0) = 5.298$ | 2 | $rac{2 	imes 2.5}{2 + 1.5(1.1875)} = rac{5.0}{3.781} = 1.322$ | $5.298 	imes 1.322 = \mathbf{7.004}$ |

**Total Skor BM25:**
$$	ext{Score}_{BM25} = 3.159 + 3.760 + 7.004 = \mathbf{13.923}$$

---

## 6. Lampiran 5: Contoh & Perhitungan Hybrid Retrieval & Reciprocal Rank Fusion (RRF)

Modul Retriever (`src/retrieval/retriever.py`) menggabungkan hasil pencarian semantik (Dense) dan leksikal (Sparse) menggunakan **Reciprocal Rank Fusion (RRF)** dengan parameter konstanta $k = 60$.

### Rumus RRF:
$$RRF(d) = \sum_{m \in \{	ext{Dense}, 	ext{Sparse}\}} I(d \in R_m) \cdot rac{1}{k + 	ext{rank}_m(d)}$$
Di mana:
- $k = 60$
- $	ext{rank}_m(d)$: Peringkat dokumen $d$ pada metode $m$ (1-indexed). Jika tidak muncul di Top-10 metode tersebut, kontribusinya adalah $0$.

### Tabel Simulasi Penggabungan Peringkat (Top-10):

| Chunk ID | Halaman & Isi | Rank Dense | Skor Dense | Rank Sparse | Skor Sparse | Perhitungan RRF | Skor RRF Final | Status |
|---|---|---|---|---|---|---|---|---|
| `tbl_214` | Hal. 214 (Total Aset 2025) | 1 | 0.892 | 1 | 13.923 | $rac{1}{60+1} + rac{1}{60+1} = rac{1}{61} + rac{1}{61}$ | **0.03278** | **Kandidat #1** |
| `narr_215` | Hal. 215 (Penjelasan Aset) | 2 | 0.841 | 3 | 8.410 | $rac{1}{60+2} + rac{1}{60+3} = rac{1}{62} + rac{1}{63}$ | **0.03200** | **Kandidat #2** |
| `tbl_218` | Hal. 218 (Laba Rugi 2025) | 3 | 0.780 | 2 | 10.120 | $rac{1}{60+3} + rac{1}{60+2} = rac{1}{63} + rac{1}{62}$ | **0.03200** | **Kandidat #3** |
| `narr_42` | Hal. 42 (Rencana Capex) | 4 | 0.710 | - | - | $rac{1}{60+4} + 0 = rac{1}{64}$ | **0.01562** | Eliminasi ($< 0.02$) |
| `narr_85` | Hal. 85 (Profil Direksi) | 5 | 0.650 | - | - | $rac{1}{60+5} + 0 = rac{1}{65}$ | **0.01538** | Eliminasi ($< 0.02$) |

### Relevance Threshold Gating:
Sistem menetapkan threshold minimum $	ext{MIN\_RRF\_SCORE} = 0.02$.
- Chunk `tbl_214` ($0.03278 \ge 0.02$), `narr_215` ($0.03200 \ge 0.02$), dan `tbl_218` ($0.03200 \ge 0.02$) lolos ke generator LLM.
- Chunk yang hanya didukung oleh salah satu retriever pada peringkat bawah akan tereliminasi secara otomatis, mencegah *noise* masuk ke konteks LLM.

---

## 7. Lampiran 6: Contoh Prompt Engineering & Generasi Jawaban LLM (Qwen)

### Format Injeksi Prompt ke LLM (`src/generation/generator.py`):
```text
SYSTEM: Anda adalah asisten AI yang menjawab pertanyaan berdasarkan data dari dokumen prospektus IPO.
ATURAN MUTLAK:
1. Jawab LANGSUNG dan natural seperti manusia berbicara. DILARANG pakai format heading seperti "Ringkasan:", "Detail:", dll.
2. WAJIB: Jika ada satuan di data, KONVERSI LANGSUNG dan tulis angka penuh. Jangan tulis angka kecil tanpa satuan.
   Contoh BENAR: "Total aset perusahaan pada 2025 adalah Rp712.031.000.000 (sekitar Rp712 miliar)"

USER: Kamu sedang menjawab pertanyaan tentang prospektus IPO perusahaan: BACH.

DATA DARI PROSPEKTUS BACH:
============================================================
[KONTEKS 1 | Halaman 214 | TABEL KEUANGAN | Relevansi: 0.0328]
[BAB V: IKHTISAR DATA KEUANGAN PENTING]
(Satuan: jutaan Rupiah)

| Akun / Pos Keuangan | 31 Des 2025 | 31 Des 2024 | 31 Des 2023 |
| TOTAL ASET | 712.031 | 750.066 | 701.406 |
============================================================

PERTANYAAN: Berapa total aset perusahaan pada tahun 2025?
JAWAB LANGSUNG.
JAWABAN:
```

### Jawaban yang Dihasilkan LLM:
```text
Total aset perusahaan pada tahun 2025 adalah Rp712.031.000.000 (atau sekitar Rp712 miliar).
```

---

## 8. Lampiran 7: Contoh Perhitungan Evaluasi Sistem Lengkap (Sesuai Bab 2.2.12)

Evaluasi dilakukan pada 4 pilar evaluasi sistem RAG:

---

### 8.1 Evaluasi Retrieval (Precision@K, Recall@K, MRR)

Pengujian dilakukan menggunakan 3 sampel pertanyaan pengujian dengan variasi kategori (Tabel Numerik, Faktual Narasi, Definisi Istilah).

#### Data Pertanyaan & Ground Truth:
- **$Q_1$ (Tabel Numerik):** *"Berapa total aset BACH pada tahun 2025?"*
  - Halaman Relevan Sebenarnya ($	ext{Total\_Rel}$): $\{214\}$
- **$Q_2$ (Faktual Narasi):** *"Sebutkan susunan dewan direksi JELI?"*
  - Halaman Relevan Sebenarnya ($	ext{Total\_Rel}$): $\{85, 86\}$
- **$Q_3$ (Definisi Istilah):** *"Apa definisi Rekening Efek?"*
  - Halaman Relevan Sebenarnya ($	ext{Total\_Rel}$): $\{15\}$

---

#### Hasil Retrieval Sistem (Top-6 Halaman per Metode):

| Query | Metode | Halaman Hasil Retrieval (Peringkat 1 s.d. 6) | Rel Pertama ($	ext{rank}_i$) |
|---|---|---|---|
| **$Q_1$** | **Sparse (BM25)** | $[214, 218, 220, 110, 45, 80]$ | Peringkat 1 |
| | **Dense (BGE-M3)** | $[214, 215, 218, 90, 100, 12]$ | Peringkat 1 |
| | **Hybrid (RRF)** | $[214, 218, 215, 220, 110, 90]$ | Peringkat 1 |
| **$Q_2$** | **Sparse (BM25)** | $[85, 40, 92, 105, 112, 150]$ | Peringkat 1 (hanya 1 dari 2 hal) |
| | **Dense (BGE-M3)** | $[86, 85, 87, 88, 90, 95]$ | Peringkat 1 (keduanya ditemukan) |
| | **Hybrid (RRF)** | $[85, 86, 87, 40, 92, 88]$ | Peringkat 1 (keduanya ditemukan) |
| **$Q_3$** | **Sparse (BM25)** | $[15, 16, 17, 18, 20, 25]$ | Peringkat 1 |
| | **Dense (BGE-M3)** | $[45, 15, 18, 90, 102, 115]$ | Peringkat 2 |
| | **Hybrid (RRF)** | $[15, 16, 45, 18, 17, 20]$ | Peringkat 1 |

---

#### Perhitungan Metrik Retrieval Langkah Demi Langkah:

#### 1. Precision@K ($P@K = rac{|Rel@K|}{K}$)
- **Untuk $Q_1$ (Kebutuhan: Hal. 214):**
  - $P@1 = rac{1}{1} = 1.000$ (Hal. 214 di posisi 1)
  - $P@3 = rac{1}{3} = 0.333$ (1 halaman relevan di 3 teratas)
  - $P@6 = rac{1}{6} = 0.167$ (1 halaman relevan di 6 teratas)

- **Untuk $Q_2$ (Kebutuhan: Hal. 85, 86 pada Hybrid RRF):**
  - $P@1 = rac{1}{1} = 1.000$ (Hal. 85 di posisi 1)
  - $P@3 = rac{2}{3} = 0.667$ (Hal. 85 dan 86 di 3 teratas)
  - $P@6 = rac{2}{6} = 0.333$ (Hal. 85 dan 86 di 6 teratas)

- **Untuk $Q_3$ (Kebutuhan: Hal. 15 pada Hybrid RRF):**
  - $P@1 = rac{1}{1} = 1.000$
  - $P@3 = rac{1}{3} = 0.333$
  - $P@6 = rac{1}{6} = 0.167$

---

#### 2. Recall@K ($Recall@K = rac{|Rel@K|}{|Total\_Rel|}$)
- **Untuk $Q_1$ ($|Total\_Rel| = 1$):**
  - $R@1 = rac{1}{1} = 1.000$
  - $R@3 = rac{1}{1} = 1.000$
  - $R@6 = rac{1}{1} = 1.000$

- **Untuk $Q_2$ ($|Total\_Rel| = 2$ pada Hybrid RRF):**
  - $R@1 = rac{1}{2} = 0.500$ (Baru Hal. 85 ditemukan di Top-1)
  - $R@3 = rac{2}{2} = 1.000$ (Hal. 85 dan 86 ditemukan di Top-3)
  - $R@6 = rac{2}{2} = 1.000$

- **Untuk $Q_3$ ($|Total\_Rel| = 1$ pada Hybrid RRF):**
  - $R@1 = rac{1}{1} = 1.000$
  - $R@3 = rac{1}{1} = 1.000$
  - $R@6 = rac{1}{1} = 1.000$

---

#### 3. Mean Reciprocal Rank ($MRR = rac{1}{|Q|} \sum_{i=1}^{|Q|} rac{1}{	ext{rank}_i}$)
- Pada metode **Hybrid RRF**:
  - $Q_1$: Dokumen relevan di peringkat 1 $ightarrow rac{1}{1} = 1.000$
  - $Q_2$: Dokumen relevan di peringkat 1 $ightarrow rac{1}{1} = 1.000$
  - $Q_3$: Dokumen relevan di peringkat 1 $ightarrow rac{1}{1} = 1.000$
  $$MRR_{	ext{Hybrid}} = rac{1.000 + 1.000 + 1.000}{3} = \mathbf{1.0000}$$

- Pada metode **Dense Only**:
  - $Q_1$: Peringkat 1 $ightarrow 1.000$
  - $Q_2$: Peringkat 1 $ightarrow 1.000$
  - $Q_3$: Peringkat 2 $ightarrow rac{1}{2} = 0.500$
  $$MRR_{	ext{Dense}} = rac{1.000 + 1.000 + 0.500}{3} = \mathbf{0.8333}$$

---

#### Tabel Rekapitulasi Perbandingan Metode Retrieval:

| Metode Retrieval | Precision@1 | Precision@3 | Precision@6 | Recall@1 | Recall@3 | Recall@6 | MRR |
|---|---|---|---|---|---|---|---|
| **Sparse Retrieval (BM25)** | 1.0000 | 0.4444 | 0.2222 | 0.8333 | 0.8333 | 0.8333 | 1.0000 |
| **Dense Retrieval (BGE-M3)** | 0.6667 | 0.5556 | 0.2778 | 0.6667 | 1.0000 | 1.0000 | 0.8333 |
| **Hybrid Retrieval (RRF)** | **1.0000** | **0.4444** | **0.2222** | **0.8333** | **1.0000** | **1.0000** | **1.0000** |

*Kesimpulan Evaluasi Retrieval:* Hybrid RRF menggabungkan keunggulan BM25 dalam menempatkan dokumen di posisi puncak ($MRR = 1.0000$) dan keunggulan BGE-M3 dalam cakupan retrieval dokumen naratif multi-halaman ($Recall@3 = 1.0000$).

---

### 8.2 Evaluasi RAGAS

RAGAS mengukur kualitas generasi pada skala 0.0 s.d. 1.0 pada 4 dimensi:

#### 1. Faithfulness (Kejujuran / Anti-Halusinasi)
Mengukur rasio klaim faktual dalam jawaban yang dapat diverifikasi dari konteks:
$$	ext{Faithfulness} = rac{|	ext{Klaim Faktual Terverifikasi Konteks}|}{|	ext{Total Klaim Faktual dalam Jawaban}|}$$

*Contoh Perhitungan ($Q_1$):*
- Jawaban: *"Total aset perusahaan pada 2025 adalah Rp712.031.000.000 (sekitar Rp712 miliar)."*
- Klaim 1: Nilai total aset 2025 adalah Rp712.031.000.000 $ightarrow$ Terverifikasi di tabel Hal. 214 (True).
- Klaim 2: Nilai tersebut setara dengan sekitar Rp712 miliar $ightarrow$ Terverifikasi secara matematis (True).
$$	ext{Faithfulness} = rac{2}{2} = \mathbf{1.000}$$

---

#### 2. Answer Relevancy (Relevansi Jawaban)
Mengukur apakah jawaban langsung menjawab pertanyaan tanpa memuat informasi yang tidak diminta.
Dihitung dengan mengenerate $M = 3$ pertanyaan balik ($q'_j$) dari jawaban, kemudian menghitung rata-rata cosine similarity terhadap pertanyaan asli $q$:
$$	ext{Answer Relevancy} = rac{1}{M} \sum_{j=1}^M 	ext{Sim}(E(q), E(q'_j))$$

*Contoh:*
- $q$: *"Berapa total aset perusahaan pada tahun 2025?"*
- $q'_1$: *"Berapa aset BACH tahun 2025?"* ($	ext{Sim} = 0.962$)
- $q'_2$: *"Berapa jumlah total aset pada 2025?"* ($	ext{Sim} = 0.951$)
- $q'_3$: *"Berapakah aset perseroan di tahun 2025?"* ($	ext{Sim} = 0.948$)
$$	ext{Answer Relevancy} = rac{0.962 + 0.951 + 0.948}{3} = \mathbf{0.9537}$$

---

#### 3. Context Precision (Presisi Konteks)
Mengukur apakah potongan konteks yang relevan ditempatkan pada urutan teratas dalam hasil retrieval:
$$	ext{Context Precision@K} = rac{\sum_{k=1}^K (P@k 	imes v_k)}{	ext{Total Chunk Relevan}}$$
Di mana $v_k \in \{0, 1\}$ adalah indikator apakah chunk pada peringkat $k$ relevan.

*Contoh ($Q_1$ - Top 3 Chunks: [Relevan, Relevan, Tidak Relevan]):*
- $k=1$: $v_1 = 1, P@1 = 1/1 = 1.000$
- $k=2$: $v_2 = 1, P@2 = 2/2 = 1.000$
- $k=3$: $v_3 = 0, P@3 = 2/3 = 0.667$
$$	ext{Context Precision} = rac{(1.000 	imes 1) + (1.000 	imes 1) + (0.667 	imes 0)}{2} = rac{2.000}{2} = \mathbf{1.0000}$$

---

#### 4. Context Recall (Kelengkapan Konteks)
Mengukur sejauh mana informasi ground truth tercakup dalam konteks yang diretrieve:
$$	ext{Context Recall} = rac{|	ext{Kalimat Ground Truth yang Didukung Konteks}|}{|	ext{Total Kalimat Ground Truth}|}$$

*Contoh ($Q_2$ - Susunan Direksi):*
- Ground Truth memiliki 2 kalimat informasi (Direktur Utama, Direktur Operasional, Direktur Keuangan).
- Seluruh nama direksi ditemukan pada konteks Hal. 85 dan Hal. 86.
$$	ext{Context Recall} = rac{2}{2} = \mathbf{1.0000}$$

---

### 8.3 Evaluasi Black Box Testing

Pengujian fungsionalitas antarmuka berbasis Web Streamlit:

| ID Uji | Skenario Pengujian | Masukan (Input) | Hasil yang Diharapkan | Status |
|---|---|---|---|---|
| **BB-01** | Upload Dokumen PDF Baru | File `PRDL_2026.pdf` | Dokumen berhasil diparsing, di-chunk, dan dicatat pada registry | **PASSED** |
| **BB-02** | Pemilihan Emiten Aktif | Klik emiten `JECX` di Sidebar | Ruang lingkup pencarian terkunci pada dokumen JECX | **PASSED** |
| **BB-03** | Pertanyaan Angka Numerik | *"Berapa total aset 2025?"* | Jawaban menampilkan angka nominal lengkap beserta satuan Rupiah | **PASSED** |
| **BB-04** | Pertanyaan di Luar Dokumen | *"Siapa presiden pertama Amerika?"* | Sistem menolak: *"Informasi tidak tersedia di dokumen prospektus ini"* | **PASSED** |
| **BB-05** | Sitasi dan Tampilan Sumber | Klik tombol *"Lihat Detail Sumber"* | Menampilkan nomor halaman, preview teks, tipe chunk, dan skor RRF | **PASSED** |
| **BB-06** | Reset / Hapus Chat | Klik tombol *"Hapus Chat"* | Riwayat percakapan bersih dan session state kembali ke awal | **PASSED** |

---

### 8.4 Evaluasi Human Evaluation (Pengujian Pengguna)

Pengujian penerimaan pengguna dilakukan kepada 5 responden (analis keuangan / mahasiswa) dengan skala Likert 1–5:

$$	ext{Mean Score} = rac{\sum_{i=1}^n 	ext{Score}_i}{n}$$

| Kriteria Penilaian | Rata-rata Skor (Skala 1 - 5) | Kategori Interpretasi |
|---|---|---|
| **1. Akurasi Faktual Jawaban** | **4.80 / 5.00** | Sangat Akurat |
| **2. Ketepatan Konversi Angka Finansial** | **4.90 / 5.00** | Sangat Tepat |
| **3. Kejelasan & Format Bahasa (Natural)** | **4.70 / 5.00** | Sangat Baik |
| **4. Kebermanfaatan Sitasi Halaman Sumber** | **4.85 / 5.00** | Sangat Bermanfaat |
| **5. Kecepatan Respons Sistem** | **4.50 / 5.00** | Cepat |
| **RATA-RATA KESELURUHAN** | **4.75 / 5.00** | **SANGAT MEMUASKAN** |
