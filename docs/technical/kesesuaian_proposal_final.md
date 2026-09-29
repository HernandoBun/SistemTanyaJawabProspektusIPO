# Implementasi berdasarkan proposal final 535230133

Acuan: `535230133.pdf` pada folder PROPOSAL/PROPOSAL FINAL, 101 halaman PDF.
Nomor halaman di bawah adalah nomor tercetak dalam proposal. Dokumen proposal lain
dalam folder output bukan acuan implementasi ini.

| Bagian proposal | Implementasi |
| --- | --- |
| Hal. 45–49: alur offline dan online | `scripts/ingest_documents.py` menyiapkan indeks; Streamlit hanya memilih dokumen dan bertanya. |
| Hal. 50–52: parsing, preprocessing, chunking | LlamaParse ketat, cache berbasis hash, normalisasi karakter/spasi, pembersihan header/footer, narasi 800 karakter dengan overlap 150 dan minimum 50, tabel atomik serta pewarisan bagian. |
| Lampiran: BGE-M3 melalui OpenRouter | `Embedder` memakai baai/bge-m3, 1024 dimensi, normalisasi L2; tidak mengganti provider saat API key kosong. |
| Hal. 52–53: retrieval | BM25 dan dense masing-masing 10 kandidat, filter dokumen aktif, RRF k=60, maksimal 6 chunk. Kunci penggabungan adalah identitas chunk, bukan kemiripan awalan teks. |
| Hal. 53: relevansi | Hanya skor terbaik dibandingkan dengan 0,02. Tanpa konteks memadai, LLM tidak dipanggil. |
| Hal. 52: pertanyaan asli | Retrieval memakai normalisasi; prompt tetap menerima pertanyaan asli. Penambahan sinonim otomatis dihapus. |
| Hal. 43 dan 53: jawaban | Qwen 3.7 Flash melalui OpenRouter; instruksi mempertahankan angka, periode, dan satuan. Tidak memaksa konversi semua nominal. |
| Hal. 55–58: UI | Empat halaman, emiten aktif, informasi dokumen, chat, halaman sumber dan detail jenis/heading/tabel/satuan. Nama menu Manajemen Emiten dan layout kartu mengikuti proposal serta referensi pengguna. Warna gelap awal dipertahankan, Deploy disembunyikan. |
| Hal. 33–34: retrieval evaluation | Precision@K, Recall@K, MRR tersedia untuk identitas chunk; mode halaman tetap tersedia dan diberi label terpisah. |
| Hal. 34–35: RAGAS | Faithfulness, answer relevancy, context precision, context recall melalui evaluator OpenRouter dan embedding proyek. |
| Hal. 35–36: Black Box dan Human Evaluation | Skenario dan format penilaian di `panduan_evaluasi.md`; pengolah rating di `evaluation/human_evaluation.py`. |

## Batas dan keputusan implementasi

- “Offline” berarti persiapan sebelum tanya jawab, bukan bebas internet. LlamaParse
  dan OpenRouter memerlukan layanan eksternal. DNS/API yang gagal tidak diartikan
  sebagai dokumen berhasil diindeks.
- Metadata heading ditambahkan pada isi narasi; batas 800 karakter berlaku pada
  isi narasi sebelum prefiks metadata, sesuai perhitungan sliding window lampiran.
- Satu tabel tetap utuh dalam chunk tersimpan. Masukan embedding sangat panjang
  masih dibatasi oleh `EMBEDDING_MAX_INPUT_CHARS` untuk batas konteks layanan;
  isi sumber penuh tetap disimpan. Pemotongan ini dicatat melalui warning dan harus
  dilaporkan pada eksperimen. Parameter ini bukan klaim angka yang ditentukan proposal.
- Ground truth lama berisi halaman, bukan semua ID chunk relevan. Skor halaman
  tidak boleh dilaporkan sebagai skor chunk proposal. Anotasi chunk harus diperiksa
  peneliti; program tidak menebak label dari kesamaan nomor halaman.
- Ambang 0,02 hanya untuk gabungan dua peringkat. Laporan baseline sparse/dense
  tidak mengklaim tingkat penolakan dengan ambang hybrid yang belum dikalibrasi.
- Pencantuman halaman konteks menunjukkan sumber yang diberikan kepada model;
  kebenaran dukungan terhadap setiap klaim masih dinilai melalui sitasi/RAGAS dan audit.
- Penggantian indeks lama dilakukan setelah embedding berhasil. Ini melindungi dari
  kegagalan API embedding, tetapi bukan transaksi penuh lintas ChromaDB dan berkas BM25.
- Pilihan emiten dan pipeline disimpan per sesi pengguna agar pengguna tidak saling
  mengganti dokumen aktif. Mulai ulang aplikasi setelah pemrosesan offline selesai.
- Model evaluator memakai model yang sama secara default. Bias penilai perlu
  dilaporkan; tidak ada angka hasil penelitian yang dibuat otomatis.
