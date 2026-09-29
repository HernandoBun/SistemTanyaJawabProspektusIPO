# Pengujian sesuai proposal final

## Retrieval dan RAGAS

Gunakan pertanyaan dan korpus yang sama untuk BM25-only, dense-only, dan hybrid.
Ground truth untuk metrik chunk wajib mempunyai `relevant_chunk_ids`, yaitu
seluruh identitas chunk relevan yang ditetapkan peneliti. `source_pages` tetap
dipakai untuk pemeriksaan sitasi. K=1,3,6 adalah pilihan evaluasi operasional
mengikuti jumlah maksimum enam konteks; proposal mendefinisikan metrik secara umum.

Jalankan `jalankan.ps1 -Mode evaluation` setelah anotasi selesai. Dataset lama
dapat diperiksa dengan `-RelevanceUnit page`; laporan akan mencantumkan unit page.
Hasil itu tidak menggantikan penilaian chunk dalam proposal. RAGAS dijalankan
dengan `python -m scripts.run_evaluation --with-ragas` dari Python 3.12 yang telah
dipasangi dependensi. Jangan mengubah parameter menggunakan jawaban set uji akhir.

## Black Box Testing

Isi hasil aktual dan status setelah menjalankan skenario. Tanda belum diuji tidak
boleh diganti menjadi lulus hanya karena unit test lulus.

| ID | Masukan / langkah | Hasil yang diharapkan | Hasil aktual / status |
| --- | --- | --- | --- |
| BB01 | Buka aplikasi | Beranda dan empat menu dapat dibuka | Belum diuji manual |
| BB02 | Buka Tanya Jawab tanpa emiten | Pengguna diarahkan memilih dokumen | Belum diuji manual |
| BB03 | Pilih dokumen terindeks | Dokumen aktif benar dan chat terbuka | Belum diuji manual |
| BB04 | Ajukan pertanyaan faktual | Jawaban dan sumber sesuai dokumen aktif | Belum diuji manual |
| BB05 | Ajukan pertanyaan tanpa bukti | Informasi tidak tersedia; tidak mengarang jawaban | Belum diuji manual |
| BB06 | Buka detail sumber | Halaman, jenis chunk, heading, tabel/satuan dan isi terlihat | Belum diuji manual |
| BB07 | Ganti emiten | Riwayat sebelumnya dibersihkan dan filter berganti | Belum diuji manual |
| BB08 | Buka dua sesi dengan emiten berbeda | Dokumen aktif antarsesi tidak saling berubah | Belum diuji manual |
| BB09 | Hapus percakapan | Riwayat bersih, dokumen tetap aktif | Belum diuji manual |
| BB10 | Gangguan API / DNS | Pesan gagal yang jelas; tidak menampilkan keberhasilan palsu | Belum diuji manual |
| BB11 | Daftar dokumen kosong | Pesan belum tersedia, tidak ada unggah/proses/hapus indeks | Belum diuji manual |
| BB12 | Cek header dan sidebar pada layar kecil | Deploy tersembunyi; sidebar masih dapat dibuka | Belum diuji manual |

## Human Evaluation

Libatkan pengguna sasaran sesuai proposal. Gunakan identitas peserta anonim.
Simpan satu penilaian per pasangan peserta–pertanyaan dalam JSONL. Kolom:
`participant_id`, `question_id`, `relevance`, `usefulness`, `satisfaction`, dan
`comment` opsional. Tiga nilai memakai skala 1 (sangat rendah) hingga 5 (sangat
tinggi). Skala ini merupakan operasionalisasi untuk instrumen pengujian, bukan
kutipan skala baku dalam proposal, dan perlu ditetapkan sebelum pengumpulan data.

Jalankan `python -m evaluation.human_evaluation <berkas-rating.jsonl>` pada lingkungan
proyek untuk menghitung rata-rata dan distribusi. Program menolak data kosong,
duplikat peserta–pertanyaan, dan skor di luar 1–5. Pengujian otomatis pada data
simulasi hanya menguji penghitung, bukan menggantikan evaluasi pengguna.
# Persiapan anotasi chunk

`output/anotasi_kandidat.json` memuat potongan dari halaman acuan untuk diperiksa
terhadap PDF asli. Kandidat tersebut belum merupakan ground truth. Pilih ID
potongan yang benar-benar mendukung jawaban, kemudian isi `relevant_chunk_ids`
pada baris pertanyaan terkait dalam `data/evaluation/ground_truth.jsonl`.
Jika halaman acuan atau jawabannya salah, koreksi berdasarkan PDF terlebih dahulu.

Daftar dapat dibuat ulang dengan Python 3.12 lingkungan proyek:
`python -m scripts.prepare_annotations`. Pemeriksaan kelengkapan:
`python -m evaluation.validate_dataset --relevance-unit chunk`.

Dataset saat ini hanya memiliki empat pertanyaan answerable. Tambahkan pertanyaan
lintas emiten dan kategori serta pertanyaan yang tidak dapat dijawab dari dokumen,
kemudian tinjau labelnya secara manual. Jangan melaporkan skor dari empat contoh
ini sebagai hasil evaluasi menyeluruh. Penandaan sumber oleh model juga belum
membuktikan kebenaran semantik klaim; penilaian manusia tetap diperlukan.
