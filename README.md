# Sistem Tanya Jawab Prospektus IPO

## Deploy ke Streamlit Community Cloud

Pilih repository ini, branch `main`, entry point `app.py`, dan Python **3.12**.
Isi Advanced settings > Secrets menggunakan format `.streamlit/secrets.toml.example`,
lalu masukkan API key OpenRouter langsung di dashboard (jangan commit key).
Snapshot ChromaDB, BM25, dan registry untuk 6 prospektus disertakan di `data/`.
Parsing ulang PDF tidak diperlukan untuk melayani pertanyaan.
Setelah deploy, periksa daftar emiten, jawaban beserta sumber, dan reboot aplikasi
untuk memastikan snapshot tetap terbaca. Pemakaian OpenRouter mengikuti biaya API.


Implementasi mengacu pada proposal final **535230133.pdf**: LlamaParse → preprocessing → structural chunking → BGE-M3 + BM25 → RRF → Qwen 3.7 Flash melalui OpenRouter.

## Menjalankan pada komputer ini

Buka terminal di folder `Program`. Launcher memilih Python 3.12 dan dependensi yang sesuai; jangan menjalankan Python 3.14 dengan `.python-deps`.

```powershell
.\jalankan.ps1 -Mode check
.\jalankan.ps1
```

Mode `check` memeriksa dependensi dan inisialisasi pipeline. Ini bukan pengujian koneksi API atau kebenaran jawaban.

## Menambahkan prospektus

1. Simpan PDF di `data/raw/prospectuses/`.
2. Jalankan pemrosesan dari terminal:

```powershell
.\proses_dokumen.ps1 -Documents "RANS_2026.pdf"
```

3. Tunggu `[OK] Indexed ...` dan `Ingestion complete.`. Jika `[GAGAL]` muncul, dokumen belum dinyatakan selesai. Pesan menunjukkan tahap dan tindakan yang diperlukan.
4. Mulai ulang Streamlit. Pengguna membuka **Manajemen Emiten**, memilih emiten, lalu mengajukan pertanyaan.

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

```powershell
.\jalankan.ps1 -Mode test
.\jalankan.ps1 -Mode evaluation
```

Evaluasi chunk membutuhkan `relevant_chunk_ids` pada ground truth. Dataset lama hanya mempunyai `source_pages`; anotasi chunk perlu dilengkapi peneliti. Pemeriksaan berdasarkan halaman tersedia secara eksplisit:

```powershell
.\jalankan.ps1 -Mode evaluation -RelevanceUnit page
```

Laporan halaman tidak boleh diklaim sebagai evaluasi chunk. Ketiga konfigurasi menggunakan pertanyaan dan korpus yang sama. Precision@1/3/6, Recall@1/3/6, dan MRR dihitung terpisah dari generasi. Ambang penolakan 0,02 hanya berlaku untuk hybrid RRF.

Untuk RAGAS pada lingkungan Python 3.12 yang telah terpasang:

```powershell
.venv\Scripts\python.exe -m scripts.run_evaluation --with-ragas
```

RAGAS mengukur faithfulness, answer relevancy, context precision, dan context recall. Black Box Testing serta Human Evaluation dilaksanakan menggunakan [panduan evaluasi](docs/technical/panduan_evaluasi.md). Hasil penilaian pengguna tidak dibuat oleh program.

Rincian pemetaan, keputusan implementasi, dan batas yang perlu dilaporkan tersedia di [kesesuaian proposal final](docs/technical/kesesuaian_proposal_final.md).

## Struktur

- `src/ingestion`: parsing, preprocessing, dan chunking.
- `src/retrieval`: embedding, indeks, tokenisasi, dan RRF.
- `src/generation`: prompt dan jawaban OpenRouter.
- `src/services`: orkestrasi dan pesan diagnostik.
- `src/ui` dan `app.py`: antarmuka pengguna.
- `scripts`: pemrosesan dokumen serta pengelolaan indeks.
- `evaluation`: retrieval, RAGAS, jawaban, dan ringkasan penilaian manusia.
- `tests`: pengujian otomatis.

Jangan menghapus atau membangun ulang indeks tanpa kebutuhan. Untuk mengganti dokumen yang telah diindeks, skrip Python menyediakan `--force`; gunakan Python 3.12 dengan dependensi proyek. Pemrosesan dokumen dan penggunaan antarmuka sebaiknya dijalankan secara bergantian.
