# Sistem Tanya Jawab Prospektus IPO

## Deploy ke Streamlit Community Cloud

Pilih repository ini, branch `main`, entry point `app.py`, dan Python **3.12**.
Isi Advanced settings > Secrets menggunakan format `.streamlit/secrets.toml.example`,
lalu masukkan API key OpenRouter langsung di dashboard (jangan commit key).
Snapshot ChromaDB, BM25, dan registry untuk 8 prospektus, termasuk EMMI dan WBSA,
disertakan di `data/`.
Parsing ulang PDF tidak diperlukan untuk melayani pertanyaan.
Setelah deploy, periksa daftar emiten, jawaban beserta sumber, dan reboot aplikasi
untuk memastikan snapshot tetap terbaca. Pemakaian OpenRouter mengikuti biaya API.


Implementasi mengacu pada proposal final **535230133.pdf**: LlamaParse → preprocessing → structural chunking → BGE-M3 + BM25 → RRF → Qwen 3.7 Flash melalui OpenRouter.

## Menjalankan pada komputer ini

Buka terminal di folder `Program` dan gunakan lingkungan Python 3.12 yang telah dipasangi dependensi proyek.

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

## Menambahkan prospektus

1. Simpan PDF di `data/raw/prospectuses/` dengan nama `KODE_TAHUN.pdf`, misalnya
   `KODE_2025.pdf`. Gunakan tahun prospektus/IPO yang sebenarnya.
2. Jalankan pemrosesan dari terminal:

```powershell
.venv\Scripts\python.exe -m scripts.ingest_documents --documents "RANS_2026.pdf"
```

3. Tunggu `[OK] Indexed ...` dan `Ingestion complete.`. Jika `[GAGAL]` muncul, dokumen belum dinyatakan selesai. Pesan menunjukkan tahap dan tindakan yang diperlukan.
4. Muat ulang halaman **Manajemen Emiten**, pilih **Tahun prospektus**, lalu
   pilih emiten yang berstatus **SIAP** untuk mulai bertanya. Ulangi perintah
   pemrosesan untuk setiap PDF yang ingin ditambahkan satu per satu.

Pemrosesan offline berarti terpisah dari tanya jawab. LlamaParse dan OpenRouter tetap membutuhkan internet. Cache parsing dipakai ulang selama sumber dan konfigurasi sesuai. Kegagalan embedding tidak mengganti provider secara diam-diam.

## Instalasi baru

Gunakan Python 3.12. Lingkungan baru tidak memakai `.python-deps` yang mungkin dibuat dengan interpreter lain.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Buat `.env` dari `.env.example` hanya jika `.env` belum ada. Jangan menimpa API key yang sudah disimpan. Isi `OPENROUTER_API_KEY` dan `LLAMA_CLOUD_API_KEY`. Konfigurasi utama:

```text
PARSER_MODE=strict_llamaparse
EMBEDDING_PROVIDER=openrouter
OPENROUTER_EMBEDDING_MODEL=baai/bge-m3
OPENROUTER_MODEL_NAME=qwen/qwen3.7-flash
MIN_RRF_SCORE=0.02
```

`requirements-local.txt` hanya diperlukan untuk eksperimen embedding lokal yang dipilih secara eksplisit. Model dan provider baku tetap mengikuti proposal.

## Parameter penelitian

| Komponen | Konfigurasi |
| --- | --- |
| Narasi | 800 karakter, overlap 150, minimum 50 |
| Tabel | Chunk atomik, metadata judul dan satuan, penggabungan tabel lanjutan |
| Embedding | BGE-M3, 1024 dimensi, normalisasi L2 |
| Dense | ChromaDB cosine, 10 kandidat dari dokumen aktif |
| Sparse | BM25 per dokumen, 10 kandidat |
| Fusion | RRF k=60, maksimal 6 konteks |
| Relevansi | Skor terbaik minimal 0,02 untuk memanggil LLM |
| Generasi | Qwen 3.7 Flash, temperature 0,1, maksimum 2048 token |

Pertanyaan dinormalisasi untuk pencarian; pertanyaan asli diberikan kepada generator. Jawaban mempertahankan periode dan satuan serta menampilkan sumber. Pilihan emiten terpisah untuk setiap sesi pengguna.

## Pengujian dan evaluasi

Gunakan satu notebook [Evaluasi_Retrieval_RAGAS.ipynb](Evaluasi_Retrieval_RAGAS.ipynb)
di VS Code. Pilih kernel Python lokal dan jalankan **Run All**.
Notebook menghitung ulang retrieval dan menjalankan RAGAS dari enam konteks teratas.
Pengujian pertama membutuhkan koneksi API. Checkpoint berikutnya dapat dilanjutkan.
Ubah `ULANG_RAGAS = True` untuk menghasilkan dan menilai jawaban baru melalui API.

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
.venv\Scripts\python.exe -m pytest tests -q
```

Set uji memakai 63 pertanyaan dari tujuh prospektus pada
`data/evaluation/thesis_benchmark/`: 56 pertanyaan dengan jawaban dan tujuh tanpa jawaban.
Retrieval menampilkan distribusi dokumen, Precision@K, Recall@K, MRR@K pada K=1,3,6,
serta hasil empat kategori pada K=6. Rerata retrieval memakai 56 pertanyaan dengan jawaban.
RAGAS hanya memproses 56 pertanyaan yang mempunyai jawaban, dengan metrik
Faithfulness, Answer Relevancy, Context Precision, dan Context Recall.
Mode RAGAS `retrieved_top_k` menguji retrieval dan generasi tanpa ambang penolakan
aplikasi. Hasil baru tersimpan di `evaluation/results/ragas_56_konteks/`.
Hasil lama di `ragas_56/` tetap menjadi arsip. Pengujian dinyatakan lengkap setelah
seluruh 56 pertanyaan memperoleh empat skor. Rerata disertai jumlah pertanyaan dinilai.
Tabel dan rincian jawaban disimpan di `evaluation/results/evaluasi_retrieval_ragas/`.
Setiap pertanyaan mempunyai rincian retrieval untuk tiga metode pada K=1,3,6,
peringkat dan teks chunk, serta hasil RAGAS dengan jawaban dan konteks aktual.
Tabel ditampilkan langsung sebagai Markdown di notebook, dengan ekspor `Hasil_Evaluasi.md` dan CSV.
API key dibaca dari `.env`. Anotasi yang masih berstatus pending tetap memerlukan validasi pakar.

Black Box Testing serta Human Evaluation dilaksanakan menggunakan
[panduan evaluasi](docs/technical/panduan_evaluasi.md).

Rincian pemetaan, keputusan implementasi, dan batas yang perlu dilaporkan tersedia di [kesesuaian proposal final](docs/technical/kesesuaian_proposal_final.md).

## Struktur

Semua lokasi berikut dihitung dari folder `Program`.

| Lokasi | Fungsi |
| --- | --- |
| `app.py` | Aplikasi Streamlit, navigasi halaman, pilihan emiten, percakapan, dan tampilan sumber jawaban. |
| `config.py` | Lokasi penyimpanan, parameter preprocessing, chunking, retrieval, model, dan pembacaan konfigurasi lingkungan. |
| `Evaluasi_Retrieval_RAGAS.ipynb` | Satu notebook evaluasi lokal. Menampilkan ringkasan dan hasil setiap pertanyaan dalam tabel Markdown. |
| `src/ingestion/parser.py` | Ekstraksi halaman dan tabel PDF melalui LlamaParse atau parser lokal, beserta cache parsing. |
| `src/ingestion/preprocessor.py` | Pembersihan teks dokumen, header dan footer berulang, serta normalisasi pertanyaan. |
| `src/ingestion/chunker.py` | Pembentukan chunk narasi dan tabel dengan metadata halaman, bagian, serta satuan keuangan. |
| `src/retrieval/embedder.py` | Embedding BGE-M3 melalui penyedia yang dipilih. |
| `src/retrieval/tokenizer.py` | Tokenisasi teks dan pertanyaan untuk pencarian BM25. |
| `src/retrieval/indexer.py` | Penyimpanan dan pencarian melalui ChromaDB serta BM25 per emiten. |
| `src/retrieval/retriever.py` | Pengambilan kandidat sparse dan dense, perluasan kueri, serta penggabungan peringkat menggunakan RRF. |
| `src/generation/generator.py` | Penyusunan konteks dan prompt, pemanggilan model, pembersihan jawaban, serta pemilihan sumber sitasi. |
| `src/services/pipeline.py` | Penghubung pengindeksan dokumen, pencarian, pemeriksaan ambang, dan pembentukan jawaban. |
| `src/services/diagnostics.py` | Pesan diagnostik saat layanan atau pemrosesan gagal. |
| `src/ui/` | State sesi, utilitas identitas emiten, dan stylesheet aplikasi. |
| `scripts/ingest_documents.py` | Perintah terminal untuk memproses prospektus yang dipilih. |
| `scripts/rebuild_indexes.py` | Pemilihan dan pengindeksan ulang dokumen. |
| `scripts/check_registry.py` | Pemeriksaan kesesuaian registry dan indeks. |
| `scripts/clean_indexes.py` | Utilitas untuk mereset indeks atau menghapus emiten sesuai argumen perintah. |
| `scripts/cache_evaluation_embeddings.py` | Pembentukan dan pembacaan cache embedding pertanyaan evaluasi. |
| `scripts/prepare_community_cloud.py` | Pembuatan salinan deployment dan pemeriksaan integritas berkas. |
| `scripts/run_evaluation.py`, `scripts/export_paper_tables.py`, `scripts/prepare_annotations.py` | Evaluasi terminal, ekspor tabel penelitian, dan persiapan anotasi. |
| `evaluation/notebook_evaluation.py` | Perhitungan metrik retrieval, rincian chunk, hasil RAGAS, serta ekspor Markdown dan CSV. |
| `evaluation/run_ragas_complete.py` | Pelaksana RAGAS untuk pertanyaan yang memiliki jawaban, dengan checkpoint dan pencatatan kegagalan. |
| `evaluation/ragas_evaluator.py` | Adapter empat metrik RAGAS dan penyimpanan skor parsial. |
| `evaluation/evaluate_retrieval.py`, `evaluation/evaluate_generation.py`, `evaluation/compare_methods.py` | Pengujian retrieval, jawaban, dan perbandingan konfigurasi pencarian. |
| `evaluation/preflight.py`, `evaluation/validate_dataset.py`, `evaluation/connection_check.py` | Validasi dataset, kesiapan indeks, dependensi, dan koneksi layanan. |
| `evaluation/reporting.py`, `evaluation/paper_tables.py`, `evaluation/human_evaluation.py` | Metadata eksperimen, laporan, tabel penelitian, dan pengolahan penilaian manusia. |
| `evaluation/thesis_*.py`, `evaluation/verify_*.py` | Pembentukan dataset penelitian, eksperimen tambahan, dan verifikasi fungsionalitas. |
| `evaluation/cache/` | Embedding pertanyaan yang telah dihitung untuk evaluasi. |
| `evaluation/results/` | Checkpoint, hasil pengujian, dan laporan eksperimen. Hasil notebook berada di `evaluasi_retrieval_ragas/`. |
| `data/raw/prospectuses/` | PDF prospektus sumber. |
| `data/parsed/` | Hasil ekstraksi dan preprocessing PDF. |
| `data/indexes/` | Indeks ChromaDB dan BM25. |
| `data/registry/` | Metadata dokumen yang telah diindeks. |
| `data/evaluation/thesis_benchmark/` | Pertanyaan, jawaban acuan, label chunk relevan, dan berkas anotasi. |
| `tests/unit/`, `tests/integration/` | Pengujian fungsi per modul dan hubungan antarmodul. |
| `docs/`, `alur_sistem.md` | Panduan penggunaan, pengindeksan, deployment, dan alur sistem. |
| `deploy/` | Salinan berkas aplikasi dan indeks untuk pengiriman atau arsip deployment. |
| `output/` | Hasil jawaban sistem dan anotasi dari pengujian sebelumnya. |
| `tmp/` | Berkas perantara pembuatan dokumen, render, dan pengujian lokal. |
| `requirements.txt` | Dependensi utama aplikasi. |
| `requirements-evaluation.txt` | Dependensi evaluasi dan kernel notebook. |
| `requirements-dev.txt` | Dependensi pengujian otomatis. |
| `requirements-local.txt` | Dependensi tambahan untuk embedding lokal. |
| `pyproject.toml` | Pengaturan pytest dan cakupan pengujian. |

Jangan menghapus atau membangun ulang indeks tanpa kebutuhan. Untuk mengganti dokumen yang telah diindeks, skrip Python menyediakan `--force`; gunakan Python 3.12 dengan dependensi proyek. Pemrosesan dokumen dan penggunaan antarmuka sebaiknya dijalankan secara bergantian.
