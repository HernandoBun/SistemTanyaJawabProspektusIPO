# Deploy demo ke Streamlit Community Cloud

## Paket deployment

Folder `deploy/community-cloud` berisi salinan kode runtime, konfigurasi tema,
requirements, serta snapshot indeks dan registry. Isi folder ini menjadi root
repository GitHub khusus deployment. Entry point-nya adalah `app.py`.
Jangan mengunggah folder proyek induk yang berisi `.env` dan dokumen penelitian.

Snapshot indeks memuat teks prospektus. Gunakan repository private untuk paket
deployment. File indeks terbesar melebihi batas upload browser GitHub;
unggah melalui Git atau GitHub Desktop. Jangan commit `.env` atau secrets.toml.

## Pengaturan Cloud

1. Hubungkan repository deployment di https://share.streamlit.io/.
2. Pilih branch repository dan entry point `app.py`.
3. Pada Advanced settings pilih Python 3.12.
4. Salin `.streamlit/secrets.toml.example` ke kolom Secrets dan isi API key
   OpenRouter langsung di dashboard. Semua variabel tetap di tingkat root TOML.
5. Deploy dan periksa log, daftar emiten, pertanyaan, serta sumber jawaban.
6. Reboot dari dashboard dan pastikan indeks tetap terbaca dari snapshot repository.

LlamaParse tidak diperlukan untuk melayani tanya jawab dari indeks yang sudah jadi.
Pemrosesan dokumen baru tetap dilakukan di komputer lokal menggunakan konfigurasi
asli, lalu snapshot indeks dan registry diperbarui bersamaan di repository.
Jangan menjalankan ingestion ketika menyalin snapshot. Data runtime Cloud bukan
backup; repository menyimpan snapshot awal untuk deployment ulang.

## Batas verifikasi

Pemeriksaan lokal tidak membuktikan kompatibilitas Linux di Community Cloud.
Periksa log instalasi (terutama ChromaDB), penggunaan memori, dan koneksi model
setelah deploy. Jika memori tidak cukup, kurangi korpus demo atau pindah hosting.
Jangan membangun ulang indeks otomatis pada setiap startup.

Dokumentasi resmi:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
