# Cara mengindeks prospektus

Buka terminal di folder **Program**. Masukkan PDF ke `data/raw/prospectuses/`.

## Menambah prospektus 2025 satu per satu

1. Beri nama PDF dengan format `KODE_2025.pdf`, memakai kode emiten dan tahun
   prospektus yang sebenarnya. Tahun ini adalah tahun prospektus/IPO, bukan tahun
   laporan keuangan atau tanggal dokumen diindeks.
2. Simpan PDF tersebut di `data/raw/prospectuses/`.
3. Proses satu PDF dari terminal pada folder **Program**:

   ```powershell
   .\proses_dokumen.ps1 -Documents "KODE_2025.pdf"
   ```

   Ganti `KODE` dengan kode emiten sesuai nama PDF yang Anda simpan.
4. Tunggu `[OK] Indexed KODE_2025.pdf` dan `Ingestion complete.` sebelum
   memproses emiten berikutnya. Perintah ini menambahkan dokumen ke korpus yang ada.
5. Buka **Manajemen Emiten**, pilih **Tahun prospektus → 2025**, lalu klik
   **Pilih & Chat** pada emiten berstatus **SIAP**.

Ulangi langkah yang sama untuk setiap prospektus 2025. File untuk emiten dan tahun
berbeda memakai nama berbeda, misalnya `KODE_2025.pdf` dan `KODE_2026.pdf`.
Filter tahun hanya mengubah daftar kartu; dokumen aktif tetap dipakai sampai
Anda menekan **Pilih & Chat** pada dokumen lain. Nama emiten dan tahunnya tampil
di sidebar dan halaman tanya jawab.

Tahun baru otomatis tersedia ketika PDF atau indeks dengan tahun tersebut ada.
Pilihan **2025** tetap tersedia sebelum PDF pertama ditambahkan, dengan pesan
bahwa prospektus tahun tersebut belum tersedia. File tanpa tahun yang jelas
ditampilkan dalam **Tahun belum diketahui**.

Untuk Streamlit Cloud, unggah perubahan `app.py` dan `src/ui/utils.py` ke
repository aplikasi. Setelah menambah PDF secara lokal, perbarui seluruh
`data/indexes/` dan `data/registry/index_registry.json` bersama-sama, lalu reboot
aplikasi Cloud. Filter tetap berfungsi dari registry walaupun PDF mentah tidak
diunggah.

## Contoh perintah per dokumen

Jalankan satu perintah sesuai dokumen yang ingin diproses:

```powershell
.\proses_dokumen.ps1 -Documents "ABDI_2025.pdf"
.\proses_dokumen.ps1 -Documents "BACH_2026.pdf"
.\proses_dokumen.ps1 -Documents "EMMI_2026.pdf"
.\proses_dokumen.ps1 -Documents "JECX_2026.pdf"
.\proses_dokumen.ps1 -Documents "JELI_2026.pdf"
.\proses_dokumen.ps1 -Documents "PRDL_2026.pdf"
.\proses_dokumen.ps1 -Documents "RANS_2026.pdf"
.\proses_dokumen.ps1 -Documents "SWAP_2026.pdf"
.\proses_dokumen.ps1 -Documents "WBSA_2026.pdf"
```

Tidak perlu menjalankan semuanya jika hanya membutuhkan satu emiten. Untuk
beberapa dokumen sekaligus, gunakan daftar nama dipisahkan koma:

```powershell
.\proses_dokumen.ps1 -Documents "JECX_2026.pdf","RANS_2026.pdf","SWAP_2026.pdf"
```

Tunggu sampai `[OK] Indexed nama_file.pdf` dan `Ingestion complete.` muncul.
Jangan menjalankan beberapa proses pengindeksan bersamaan pada indeks yang sama.
Dokumen dilewati jika skema indeks dan sidik SHA-256 isi PDF cocok. Mengganti
isi PDF dengan nama yang sama akan memicu pengindeksan ulang. Indeks lama yang
belum menyimpan SHA-256 tetap dapat digunakan untuk chat, tetapi akan diproses
ulang satu kali ketika perintah pengindeksannya dijalankan.

Jika penyimpanan indeks pengganti atau registry gagal, sistem mencoba memulihkan
indeks sebelumnya. Pemulihan ini menangani kegagalan yang tertangkap program;
belum menjamin pemulihan saat listrik padam atau proses dihentikan paksa.

Hasil disimpan dalam ChromaDB dan BM25 per dokumen. Registry mencatat dokumen
yang siap dipilih. File PDF saja belum cukup untuk menjadikannya siap digunakan.

Setelah pemrosesan, buka atau muat ulang halaman **Manajemen Emiten**. Aplikasi
memuat ulang indeks ketika registry berubah dan mempertahankan emiten aktif
yang masih tersedia. Jika aplikasi belum berjalan, jalankan:

```powershell
.\jalankan.ps1
```

Buka **Manajemen Emiten** dan klik **Pilih & Chat** pada emiten yang berstatus
**SIAP**. Perhatikan kode emiten di bagian atas chat. Seluruh pertanyaan berikutnya
dibatasi pada prospektus tersebut.

Proses offline tetap memerlukan koneksi untuk LlamaParse dan embedding OpenRouter.
Jika muncul `[GAGAL]`, periksa pesan kesalahannya; jangan menganggap indeks selesai.
Pemeriksaan `.\proses_dokumen.ps1 -Check` juga memeriksa apakah LlamaParse dapat
dimuat. Paket harus terpasang pada Python 3.12 yang dipakai launcher, bukan pada
Python 3.14 bawaan terminal. Pesan parsing 0% berarti layanan masih mengekstrak
PDF; tunggu hasil tahap berikutnya atau pesan kegagalan.
Perbaikan ID retrieval tidak memerlukan pengindeksan ulang RANS yang sudah berhasil.
