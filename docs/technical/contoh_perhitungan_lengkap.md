# CONTOH PERHITUNGAN LENGKAP SELURUH ALUR SISTEM

## Sistem Tanya Jawab Prospektus IPO Berbasis Hybrid Retrieval RAG

Dokumen ini menjelaskan satu alur utuh dari berkas PDF sampai jawaban dan evaluasi. Setiap angka diberi status agar tidak tercampur:

- **Aktual**: dibaca dari prospektus, cache parsing, konfigurasi, atau keluaran kode pada repositori saat dokumen ini disusun.
- **Hasil hitung ulang**: dihitung ulang secara lokal memakai rumus dan aturan kode saat ini, tanpa memanggil layanan eksternal.
- **Simulasi**: angka contoh untuk menjelaskan tahap yang tidak dapat direproduksi tanpa embedding/API. Angka simulasi bukan hasil eksperimen penelitian.

> **Skenario utama.** Dokumen yang digunakan adalah `BACH_2026.pdf`. Pertanyaan contoh: **“Berapa total aset perusahaan tahun 2025?”** Fakta sumber yang benar berada pada halaman 21: **Total Aset 2025 = Rp1.232.630.823.230**. Angka Rp712.031.000.000 dan halaman 214 pada dataset evaluasi lama tidak sesuai dengan prospektus BACH 2026 yang tersimpan saat ini.

## 1. Peta Alur dan Titik Perhitungan

| Tahap | Masukan | Operasi utama | Keluaran |
|---|---|---|---|
| 1. Identifikasi | PDF BACH | SHA-256 dan pemeriksaan cache | Identitas sumber yang stabil |
| 2. Parsing | PDF | LlamaParse dalam mode ketat | 402 objek halaman |
| 3. Preprocessing | Teks dan tabel per halaman | Normalisasi dan penghapusan tepi berulang | Teks bersih dan laporan audit |
| 4. Chunking | Halaman bersih | Pemisahan struktural, sliding window, tabel atomik | 2.033 chunk |
| 5. Dense indexing | Isi chunk | BGE-M3, normalisasi L2, ChromaDB | Vektor 1.024 dimensi |
| 6. Sparse indexing | Isi chunk | Tokenisasi finansial dan BM25Okapi | Indeks leksikal |
| 7. Persiapan query | Pertanyaan | Normalisasi dan ekspansi istilah | Query retrieval |
| 8. Retrieval | Query dan dokumen aktif | Dense Top-10 dan sparse Top-10 | Dua daftar kandidat |
| 9. Fusi | Dua daftar peringkat | Reciprocal Rank Fusion, k = 60 | Top-6 konteks |
| 10. Gating | Top-6 hasil RRF | Bandingkan skor terbaik dengan 0,02 | Tolak atau panggil LLM |
| 11. Generasi | Pertanyaan dan konteks | Qwen melalui OpenRouter | Jawaban natural |
| 12. Sitasi | Metadata chunk | Himpunan halaman dan preview | Halaman sumber |
| 13. Evaluasi | Prediksi dan ground truth | Retrieval, generasi, RAGAS, black-box, pengguna | Skor penelitian |

## 2. Data Sumber Aktual

### 2.1 Identitas Berkas

| Atribut | Nilai aktual |
|---|---|
| Nama berkas | `BACH_2026.pdf` |
| Emiten | PT Bach Multi Global Tbk. |
| Jumlah halaman hasil parsing | 402 |
| SHA-256 lengkap | `0e2e96118d0671e3a9475a5db427fd837762ae7f2092e8839bc9ebcf6f9d124f` |
| Dua belas karakter hash | `0e2e96118d06` |
| Mode parser | `strict_llamaparse` |
| Parser yang dipakai | `llamaparse` |
| Versi cache | `v3` |
| Nama cache | `BACH_2026_0e2e96118d06_strict_llamaparse_v3.json` |

SHA-256 dihitung dengan membaca PDF per blok 1 MiB. Secara matematis:

`H = SHA256(B1 || B2 || ... || Bm)`

`Bi` adalah blok byte ke-i dan `||` berarti konkatenasi. Perubahan satu byte saja menghasilkan hash berbeda sehingga cache lama tidak dipakai.

### 2.2 Fakta yang Akan Dijawab

Cuplikan aktual hasil parsing halaman 21:

```text
| Keterangan       | 2025              | 2024            | 2023            |
| Aset lancar      | 767.238.988.666   | 683.343.711.787 | 579.843.781.178 |
| Aset tidak lancar| 465.391.834.564   | 132.264.402.454 | 100.434.130.194 |
| Total Aset       | 1.232.630.823.230 | 815.608.114.271 | 680.277.911.372 |
```

Pemeriksaan aritmetika:

`Aset lancar + aset tidak lancar = Total aset`

`Rp767.238.988.666 + Rp465.391.834.564 = Rp1.232.630.823.230`

Selisihnya adalah nol, sehingga angka total konsisten dengan dua komponen aset pada tabel.

## 3. Parsing dan Cache

### 3.1 Keputusan Memakai Cache

Cache hanya valid bila seluruh kondisi berikut benar:

1. `source_sha256` cache sama dengan SHA-256 PDF.
2. `parser_mode` sama dengan mode yang diminta.
3. `cache_version` sama dengan `v3`.
4. konfigurasi preprocessing pada cache sama dengan konfigurasi aktif.
5. daftar halaman tidak kosong.

Untuk BACH, kelima kondisi terpenuhi sehingga parsing dapat dibaca dari cache tanpa mengirim PDF kembali ke LlamaParse.

### 3.2 Bentuk Data Halaman

Setiap halaman menjadi objek dengan elemen berikut:

| Elemen | Fungsi | Contoh halaman 21 |
|---|---|---|
| `page_number` | Nomor halaman 1-indexed | `21` |
| `text_content` | Narasi tanpa blok tabel Markdown | Judul ikhtisar keuangan |
| `tables` | Daftar tabel Markdown | 2 tabel |
| `raw_text` | Hasil parser sebelum pemisahan tabel | Narasi dan tabel |
| `removed_header_footer` | Baris tepi yang dihapus | Audit preprocessing |
| `preprocessing_stats` | Statistik per halaman | Jumlah kontrol/header/footer |

Nomor halaman ini kemudian diwariskan ke chunk. Sistem tidak meminta LLM menebak halaman.

## 4. Preprocessing secara Rinci

### 4.1 Konfigurasi Aktual

| Parameter | Nilai |
|---|---:|
| Preprocessing aktif | `true` |
| Baris tepi yang diperiksa | 3 baris nonkosong teratas dan terbawah |
| Rasio pengulangan minimum | 0,65 |
| Minimum halaman berulang | 3 |
| Maksimum baris kosong berurutan | 2 |

### 4.2 Ambang Baris Berulang

Rumus kode:

`threshold = max(min_repeat_pages, ceil(total_pages x repeat_ratio))`

Substitusi untuk BACH:

`threshold = max(3, ceil(402 x 0,65))`

`threshold = max(3, ceil(261,30)) = max(3, 262) = 262 halaman`

Jadi sebuah baris tepi harus muncul sedikitnya pada 262 dari 402 halaman agar menjadi kandidat header/footer berulang.

### 4.3 Pengaman agar Data Keuangan Tidak Terhapus

Walaupun berulang, baris tidak dihapus jika memuat unsur struktural atau finansial, antara lain heading Markdown, `BAB`, simbol tabel `|`, nilai `Rp`, `%`, satuan ribuan/jutaan/miliar, atau penanda daftar. Nomor halaman murni aman dihapus.

### 4.4 Hasil Audit Aktual BACH

| Ukuran | Nilai |
|---|---:|
| Halaman dinormalisasi | 402 |
| Karakter kontrol yang dihapus | 0 |
| Header berulang yang dihapus | 396 baris |
| Footer berulang yang dihapus | 396 baris |
| Total baris tepi dihapus | `396 + 396 = 792` baris |
| Karakter sebelum preprocessing | 1.310.625 |
| Karakter sesudah preprocessing | 1.065.570 |
| Karakter berkurang | `1.310.625 - 1.065.570 = 245.055` |
| Persentase pengurangan | `245.055 / 1.310.625 x 100% = 18,6976%` |
| Karakter yang dipertahankan | `1.065.570 / 1.310.625 x 100% = 81,3024%` |

Pengurangan 18,6976% bukan berarti isi substantif hilang. Sebagian besar merupakan pagar Markdown dan header/footer berulang. Tabel halaman 21 tetap utuh.

## 5. Chunking secara Rinci

### 5.1 Parameter

| Parameter | Nilai aktual |
|---|---:|
| Ukuran maksimum chunk narasi | 800 karakter |
| Overlap | 150 karakter |
| Panjang minimum | 50 karakter |
| Strategi tabel | Satu tabel menjadi satu chunk atomik |

### 5.2 Sliding Window Narasi

Jika tidak ada batas kalimat yang lebih baik, pergeseran teoritis adalah:

`stride = chunk_size - overlap = 800 - 150 = 650 karakter`

Untuk narasi sepanjang `L = 2.100` karakter, perkiraan minimum jumlah jendela:

`n = ceil((L - overlap) / (chunk_size - overlap))`

`n = ceil((2.100 - 150) / 650) = ceil(3) = 3 chunk`

Implementasi mencoba memotong pada paragraf, newline, titik, tanda tanya, atau tanda seru pada separuh akhir jendela. Karena itu, panjang aktual tiap chunk dapat lebih pendek dari 800 karakter dan titik awal berikutnya adalah `split_pos - 150`.

### 5.3 Hasil Aktual BACH

| Jenis chunk | Jumlah | Proporsi |
|---|---:|---:|
| Narasi | 1.773 | `1.773 / 2.033 = 87,21%` |
| Tabel | 260 | `260 / 2.033 = 12,79%` |
| Total | 2.033 | 100,00% |

### 5.4 Pembentukan ID Chunk

Nama sumber dinormalisasi menjadi `bach_2026`. SHA-1 dari nama sumber dalam huruf kecil menghasilkan awalan delapan karakter `282ba60d`. Pola ID:

`<stem>_<sha1-8>__<tipe>_<nomor-5-digit>`

Chunk tabel aset aktual memiliki ID:

`bach_2026_282ba60d__tbl_00133`

### 5.5 Isi dan Metadata Chunk Tabel Aktual

| Metadata | Nilai aktual |
|---|---|
| `chunk_type` | `table` |
| `page_number` | 21 |
| `char_count` | 798 |
| Jumlah token unik hasil tokenizer BM25 | 135 |
| `section_heading` | `# 5. IKHTISAR DATA KEUANGAN PENTING` |
| `financial_unit` | kosong |
| Nilai total aset | `1.232.630.823.230` |

`financial_unit` kosong karena tabel menyajikan Rupiah penuh, bukan “dalam jutaan Rupiah”. Ini benar untuk jawaban akhir: angka tidak boleh dikalikan lagi.

### 5.6 Catatan Metadata Judul Tabel

Pada chunk aktual, `table_title` terbaca sebagai judul terakhir pada teks halaman, yaitu “LAPORAN LABA RUGI DAN PENGHASILAN KOMPREHENSIF LAIN KONSOLIDASIAN”, padahal isi chunk pertama adalah laporan posisi keuangan. Hal ini terjadi karena fungsi judul mengambil tiga baris terakhir dari seluruh `page_text`, bukan baris tepat sebelum tiap tabel. Nomor halaman dan isi tabel tetap benar, tetapi judul ini tidak boleh dijadikan satu-satunya dasar evaluasi.

## 6. Dense Embedding dan Cosine Similarity

### 6.1 Proses Aktual

Setiap isi chunk diubah oleh `BAAI/bge-m3` menjadi vektor 1.024 dimensi. Query memperoleh prefix:

`Represent this sentence for searching relevant passages: <query>`

Vektor kemudian dinormalisasi L2:

`v_normal = v / ||v||2`

Karena query dan dokumen telah dinormalisasi, cosine similarity sama dengan dot product.

### 6.2 Simulasi Lima Dimensi

Bagian ini **simulasi**, bukan embedding BGE-M3 aktual. Lima dimensi dipakai agar seluruh operasi dapat dilihat.

`q = [0,124; 0,451; -0,231; 0,089; 0,312]`

`dA = [0,118; 0,462; -0,215; 0,095; 0,301]`

`dB = [-0,052; 0,112; 0,341; -0,120; 0,045]`

Untuk chunk A:

`q.dA = 0,014632 + 0,208362 + 0,049665 + 0,008455 + 0,093912 = 0,375026`

`||q|| = sqrt(0,377403) = 0,614331`

`||dA|| = sqrt(0,373219) = 0,610917`

`cos(q,dA) = 0,375026 / (0,614331 x 0,610917) = 0,999256`

Untuk chunk B:

`q.dB = -0,031347`

`||dB|| = sqrt(0,147954) = 0,384648`

`cos(q,dB) = -0,031347 / (0,614331 x 0,384648) = -0,132657`

Chunk A jauh lebih dekat secara semantik daripada chunk B.

### 6.3 Konversi Distance ChromaDB

ChromaDB dikonfigurasi dengan ruang cosine dan mengembalikan `distance`. Kode menghitung:

`similarity = 1 - distance`

Contoh: jika `distance = 0,107`, maka `similarity = 1 - 0,107 = 0,893`.

## 7. Tokenisasi Finansial dan BM25Okapi

### 7.1 Normalisasi Angka

| Teks asli | Bentuk normalisasi | Token akhir penting |
|---|---|---|
| `Rp1.232.630.823.230` | `rp1232630823230 1232630823230` | `rp1232630823230`, `1232630823230` |
| `15,06%` | `15_06_persen` | `15_06_persen` |
| `Rp 91.020.000.000` | `rp91020000000 91020000000` | `rp91020000000`, `91020000000` |
| `1.232.630.823.230` | `1232630823230` | `1232630823230` |

Tokenizer juga mempertahankan token dari bentuk asli. Karena hasil digabung dengan mekanisme `seen`, setiap token hanya muncul sekali dalam satu chunk. Konsekuensinya, frekuensi term BM25 `f(q,D)` pada korpus hasil tokenizer saat ini paling besar 1, walaupun kata yang sama muncul berulang dalam teks asli.

### 7.2 Token Query Aktual

Pertanyaan:

`Berapa total aset BACH pada tahun 2025?`

Token unik:

`[berapa, total, aset, bach, pada, tahun, 2025]`

### 7.3 Statistik Korpus BACH Hasil Hitung Ulang

| Statistik | Nilai |
|---|---:|
| Banyak dokumen/chunk, N | 2.033 |
| Panjang rata-rata, avgdl | 60,977865 token unik |
| Ukuran kosakata | 9.943 token |
| Average IDF | 6,150154 |
| Epsilon IDF | `0,25 x 6,150154 = 1,537539` |

### 7.4 Rumus yang Sesuai dengan `rank_bm25`

Parameter bawaan: `k1 = 1,5`, `b = 0,75`, `epsilon = 0,25`.

`IDF(q) = ln(N - n(q) + 0,5) - ln(n(q) + 0,5)`

IDF negatif diganti dengan `epsilon x average_idf`.

`TFweight(q,D) = f(q,D)(k1+1) / [f(q,D) + k1(1-b+b|D|/avgdl)]`

`BM25(D,Q) = sum IDF(q) x TFweight(q,D)`

### 7.5 Perhitungan Chunk Tabel Aset Halaman 21

Panjang token unik chunk: `|D| = 135`. Hanya token `total`, `aset`, dan `2025` dari query yang terdapat di chunk. Karena tokenizer melakukan deduplikasi, masing-masing `f = 1`.

Normalisasi panjang:

`K = 1,5 x (1 - 0,75 + 0,75 x 135 / 60,977865)`

`K = 1,5 x (0,25 + 1,660439) = 2,865658`

`TFweight = 1 x 2,5 / (1 + 2,865658) = 0,646720`

| Token | n(q) | IDF | f | TF weight | Kontribusi |
|---|---:|---:|---:|---:|---:|
| `total` | 96 | 2,999611 | 1 | 0,646720 | `2,999611 x 0,646720 = 1,939910` |
| `aset` | 256 | 1,935834 | 1 | 0,646720 | `1,935834 x 0,646720 = 1,251944` |
| `2025` | 498 | 1,125008 | 1 | 0,646720 | `1,125008 x 0,646720 = 0,727566` |

`Skor BM25 = 1,939910 + 1,251944 + 0,727566 = 3,919420`

Hasil hitung ulang seluruh 2.033 chunk menempatkan tabel halaman 21 pada **peringkat sparse 97**, bukan Top-10. Top-1 sparse adalah narasi halaman 166 dengan skor 8,383404. Ini menunjukkan alasan hybrid retrieval diperlukan: kecocokan kata tidak selalu mengangkat tabel angka yang paling tepat.

## 8. Persiapan Query dan Pembatasan Dokumen

### 8.1 Normalisasi Query

`preprocess_query` menerapkan Unicode NFC, menyatukan line ending, menghapus karakter kontrol, merapikan spasi, dan menggabungkan baris menjadi satu pertanyaan. Pertanyaan contoh tidak berubah.

### 8.2 Ekspansi Query

Pertanyaan contoh tidak memicu ekspansi karena bukan pertanyaan definisi dan tidak memuat kata sinonim yang terdaftar. Contoh yang memicu ekspansi:

`Berapa harta perusahaan tahun 2025?`

menjadi:

`Berapa harta perusahaan tahun 2025? aset aset lancar aset tidak lancar total aset`

Pertanyaan “Apa pengertian Rekening Efek?” diperluas dengan kata `definisi`, `berarti`, `artinya`, `pengertian`, serta `definisi dan singkatan`.

### 8.3 Filter Dokumen Aktif

Dense search memakai filter ChromaDB `where = {doc_source: BACH_2026.pdf}`. Sparse search mengambil kandidat global kemudian hanya mempertahankan chunk dengan `doc_source` yang sama. Ini mencegah data emiten lain masuk ke jawaban.

> **Catatan snapshot.** Cache BACH tersedia, tetapi registry indeks aktif pada saat pemeriksaan hanya memuat RANS, JECX, dan SWAP. Karena itu, angka BM25 BACH dalam dokumen ini adalah hasil hitung ulang dari cache dan chunker. BACH harus diindeks ulang sebelum skenario dijalankan melalui aplikasi secara live.

## 9. Reciprocal Rank Fusion dan Ambang Relevansi

### 9.1 Rumus

`RRF(d) = sum 1 / (60 + rank_m(d))`

Skor cosine dan BM25 tidak dijumlahkan karena skalanya berbeda. Yang dipakai hanya posisi peringkat.

### 9.2 Simulasi Gabungan

Tabel berikut adalah **simulasi dense ranking** yang digabung dengan pola sparse untuk menjelaskan hitungan. Ini bukan keluaran eksperimen resmi.

| Chunk | Halaman | Rank dense | Rank sparse | Perhitungan | RRF |
|---|---:|---:|---:|---|---:|
| Narasi perubahan aset | 166 | 2 | 1 | `1/62 + 1/61` | 0,032522 |
| Tabel entitas anak | 293 | 4 | 2 | `1/64 + 1/62` | 0,031754 |
| Tabel rasio | 22 | 3 | 4 | `1/63 + 1/64` | 0,031498 |
| Tabel pertumbuhan | 51 | 5 | 3 | `1/65 + 1/63` | 0,031258 |
| Tabel total aset utama | 21 | 1 | tidak Top-10 | `1/61 + 0` | 0,016393 |
| Narasi lain | 13 | 6 | 5 | `1/66 + 1/65` | 0,030536 |

Setelah diurutkan, enam chunk di atas dapat menjadi konteks. Tabel utama masih bisa masuk Top-6 walaupun hanya ditemukan dense.

### 9.3 Cara Ambang 0,02 Sebenarnya Diterapkan

Kode menghitung skor terbaik:

`best_score = max(rrf_score semua hasil Top-6)`

Lalu:

- jika `best_score < 0,02`, seluruh permintaan ditolak sebelum LLM;
- jika `best_score >= 0,02`, seluruh Top-6 diteruskan, termasuk chunk individual yang skornya kurang dari 0,02.

Pada simulasi:

`best_score = 0,032522 >= 0,02`, maka generator dipanggil. Chunk halaman 21 dengan skor 0,016393 tetap ikut karena kode tidak memfilter setiap chunk secara individual.

Nilai maksimum satu retriever saja adalah:

`1 / (60 + 1) = 1/61 = 0,016393 < 0,02`

Artinya, bila hasil terbaik hanya muncul pada satu retriever, permintaan berisiko ditolak. Evaluasi retrieval memakai fungsi `retrieve()` sehingga tidak menerapkan gate ini; gate berlaku pada `query()`.

## 10. Pembentukan Konteks dan Jawaban

### 10.1 Format Konteks

Setiap chunk disusun seperti berikut:

```text
[KONTEKS 5 | Halaman 21 | TABEL KEUANGAN | Relevansi: 0.0164]
[# 5. IKHTISAR DATA KEUANGAN PENTING]
Tabel berikut dari halaman 21:
| Total Aset | 1.232.630.823.230 | 815.608.114.271 | 680.277.911.372 |
```

### 10.2 Konversi Satuan

Untuk sumber BACH, nilai sudah Rupiah penuh:

`Rp1.232.630.823.230 x 1 = Rp1.232.630.823.230`

Jika suatu tabel lain menyatakan “dalam jutaan Rupiah”, contoh `712.031` harus dikonversi:

`712.031 x Rp1.000.000 = Rp712.031.000.000`

Perkiraan miliar:

`Rp712.031.000.000 / Rp1.000.000.000 = 712,031 miliar`

### 10.3 Jawaban yang Diharapkan

```text
Total aset perusahaan pada 2025 adalah Rp1.232.630.823.230 (sekitar Rp1,23 triliun).
```

Pembulatan triliun:

`Rp1.232.630.823.230 / Rp1.000.000.000.000 = 1,23263082323 triliun`

Dibulatkan dua desimal menjadi `Rp1,23 triliun`. Angka penuh tetap ditulis agar tidak kehilangan presisi.

### 10.4 Sitasi dan Metadata Respons

`source_pages` adalah himpunan nomor halaman semua chunk konteks yang kemudian diurutkan. Sitasi ini berbasis retrieval, bukan pemetaan klaim per kalimat. Jika Top-6 berasal dari halaman `[166, 293, 22, 51, 21, 13]`, hasilnya:

`source_pages = [13, 21, 22, 51, 166, 293]`

Setiap citation menyimpan `page_number`, `chunk_type`, `section_heading`, preview 300 karakter, dan `rrf_score`. Respons juga menyimpan nama model, total token jika tersedia, serta banyak konteks.

## 11. Alur Ujung ke Ujung dalam Satu Tabel

| Langkah | Nilai masuk | Perhitungan/keputusan | Nilai keluar |
|---|---|---|---|
| Hash | Byte PDF | SHA-256 | `0e2e...d124f` |
| Cache | Hash + mode + versi + konfigurasi | Semua harus sama | Cache v3 valid |
| Parse | PDF/cache | LlamaParse | 402 halaman |
| Preprocess | 1.310.625 karakter | Kurang 245.055 | 1.065.570 karakter |
| Chunk | Halaman bersih | 1.773 narasi + 260 tabel | 2.033 chunk |
| Target | Tabel halaman 21 | Tabel atomik | 798 karakter, 135 token unik |
| Dense | Query/chunk | Embedding 1.024-D dan cosine | Rank dense |
| Sparse | 2.033 chunk | BM25 target | 3,919420; rank 97 |
| RRF | Rank dense dan sparse | `sum 1/(60+rank)` | Skor fusi |
| Gate | Skor terbaik | Bandingkan 0,02 | Panggil/tolak LLM |
| Generator | Top-6 + pertanyaan | Prompt terikat konteks | Jawaban |
| Nilai akhir | `1.232.630.823.230` Rupiah | Tidak dikalikan | Rp1.232.630.823.230 |
| Ringkasan | Nilai penuh / 10^12 | 1,23263082323 | sekitar Rp1,23 triliun |

## 12. Evaluasi Retrieval

### 12.1 Precision@K

`Precision@K = jumlah item relevan dalam Top-K / K`

Contoh ground truth halaman `{21}` dan hasil `[166, 293, 21, 22, 51, 13]`:

- `P@1 = 0/1 = 0,0000`
- `P@3 = 1/3 = 0,3333`
- `P@6 = 1/6 = 0,1667`

### 12.2 Recall@K

`Recall@K = jumlah halaman relevan unik yang ditemukan / total halaman relevan unik`

- `R@1 = 0/1 = 0,0000`
- `R@3 = 1/1 = 1,0000`
- `R@6 = 1/1 = 1,0000`

### 12.3 Reciprocal Rank dan MRR

Halaman relevan pertama berada pada peringkat 3:

`RR = 1/3 = 0,3333`

Untuk tiga pertanyaan dengan rank relevan pertama `[3, 1, tidak ditemukan]`:

`MRR = (1/3 + 1/1 + 0) / 3`

`MRR = (0,3333 + 1 + 0) / 3 = 0,4444`

### 12.4 Unanswerable Rejection

Untuk pertanyaan tidak terjawab, kode evaluasi memberi:

`unanswerable_rejection = 1` jika retrieval kosong, selain itu `0`.

Jika 4 dari 5 pertanyaan tidak terjawab berhasil menghasilkan retrieval kosong:

`mean rejection = (1+1+1+1+0)/5 = 0,80 atau 80%`

## 13. Evaluasi Generasi Deterministik

### 13.1 Exact Match

Normalisasi mengubah huruf menjadi lowercase, mengganti tanda baca dengan spasi, lalu merapikan whitespace.

Referensi:

`Total aset perusahaan tahun 2025 adalah Rp1.232.630.823.230`

Prediksi:

`Total aset perusahaan pada 2025 adalah Rp1.232.630.823.230`

Keduanya berbeda pada token `tahun` dan `pada`, sehingga `Exact Match = 0` walaupun substansinya sama.

### 13.2 Token F1

Masing-masing kalimat memiliki 11 token setelah normalisasi. Sepuluh token sama.

`precision = 10/11 = 0,9091`

`recall = 10/11 = 0,9091`

`F1 = 2 x 0,9091 x 0,9091 / (0,9091 + 0,9091) = 0,9091`

### 13.3 Numerical Accuracy

Angka yang diekstrak dari referensi adalah `[2025, 1.232.630.823.230]`. Tahun harus sama persis. Angka keuangan memakai toleransi relatif 5%:

`relative_error = |prediction - reference| / max(|reference|, 1)`

Jika prediksi nominal sama, kedua angka cocok:

`Numerical Accuracy = 2/2 = 1,0000`

Catatan: prediksi Rp1.200.000.000.000 masih berjarak sekitar 2,65% sehingga dinilai cocok oleh toleransi, walaupun exact match gagal.

### 13.4 Citation Page Accuracy

`Citation Accuracy = 1` jika ada irisan antara halaman prediksi dan halaman referensi.

Prediksi `[166, 21]`, referensi `[21]`:

`{166,21} intersection {21} = {21}`, maka skor `1`.

Metrik ini tidak menghukum halaman tambahan yang salah; ia hanya memeriksa sedikitnya satu halaman benar.

## 14. Evaluasi RAGAS

Implementasi memakai RAGAS collections v0.4 dan mengembalikan empat skor 0-1. Skor diputuskan oleh evaluator LLM dan embedding; angka tidak boleh diklaim sebagai hasil resmi tanpa menjalankan evaluator.

| Metrik | Data yang dinilai | Interpretasi |
|---|---|---|
| Faithfulness | Jawaban dan retrieved contexts | Klaim jawaban didukung konteks |
| Answer relevancy | Pertanyaan dan jawaban | Jawaban membahas kebutuhan pengguna |
| Context precision | Pertanyaan, referensi, urutan konteks | Konteks relevan berada di posisi baik |
| Context recall | Referensi dan seluruh konteks | Informasi referensi tercakup |

Contoh manual untuk menjelaskan faithfulness: jawaban mempunyai dua klaim, yaitu nilai penuh dan nilai sekitar triliun. Keduanya didukung tabel dan aritmetika, sehingga ilustrasi manual `2/2 = 1,00`. Namun skor RAGAS resmi tetap nilai yang dikembalikan API evaluator.

## 15. Black-Box Testing

Tingkat kelulusan sederhana:

`Pass Rate = jumlah skenario lulus / total skenario x 100%`

Contoh 9 skenario lulus dari 10:

`Pass Rate = 9/10 x 100% = 90%`

Skenario minimal mencakup PDF valid, PDF rusak, cache hit, rebuild, pemilihan dokumen aktif, pertanyaan numerik, pertanyaan definisi, pertanyaan di luar dokumen, kegagalan API, sitasi halaman, dan penghapusan percakapan.

## 16. Human Evaluation

Untuk lima responden dengan skor `[5, 4, 5, 4, 5]`:

`Mean = (5+4+5+4+5)/5 = 23/5 = 4,60`

Persentase terhadap skor maksimum:

`4,60/5 x 100% = 92%`

Jika lima kriteria mempunyai mean `[4,60; 4,40; 4,80; 4,20; 4,50]` dengan bobot sama:

`Overall Mean = (4,60+4,40+4,80+4,20+4,50)/5 = 4,50`

Angka ini adalah contoh cara hitung, bukan hasil responden penelitian aktual.

## 17. Temuan yang Wajib Diperhatikan Sebelum Pengujian Skripsi

1. Ground truth Q001 saat ini menyebut Rp712.031.000.000 halaman 214, sedangkan prospektus BACH aktual menyebut Rp1.232.630.823.230 halaman 21. Ground truth harus diverifikasi ulang sebelum evaluasi.
2. Dataset memuat JELI, sedangkan registry aktif tidak memuat JELI. Evaluasi akan gagal bila dokumen tidak diindeks.
3. Tokenizer mendeduplikasi token, sehingga frekuensi kata dalam satu chunk selalu satu untuk BM25.
4. Tabel aset BACH hanya berada pada rank sparse 97 untuk query contoh; dense retrieval harus diuji agar hybrid benar-benar mengangkatnya ke Top-6.
5. Ambang 0,02 memeriksa skor terbaik keseluruhan, bukan setiap chunk.
6. Dengan k=60, skor maksimum satu stream hanya 0,016393. Ini lebih kecil dari ambang 0,02.
7. `source_pages` memuat semua halaman konteks, bukan hanya halaman yang secara langsung mendukung klaim jawaban.
8. Judul tabel halaman 21 dapat salah karena inferensi judul memakai baris terakhir seluruh halaman.
9. Nilai RAGAS dan penilaian pengguna tidak boleh diisi dengan angka contoh sebagai hasil penelitian.

## 18. Checklist Reproduksi

1. Pastikan SHA-256 PDF sama dengan yang dicatat.
2. Pastikan parser mode, cache version, dan konfigurasi preprocessing sama.
3. Rebuild indeks agar BACH masuk registry aktif.
4. Catat jumlah halaman, chunk narasi, dan chunk tabel.
5. Simpan output dense Top-10, sparse Top-10, dan RRF Top-6 per pertanyaan.
6. Simpan skor mentah, peringkat, halaman, tipe chunk, dan isi preview.
7. Tetapkan ground truth sebelum melihat keluaran model pada test split.
8. Jalankan evaluasi retrieval melalui `retrieve()` dan evaluasi end-to-end melalui `query()`.
9. Pisahkan hasil aktual, hasil hitung ulang, dan simulasi pada laporan.
10. Jangan menyimpulkan keunggulan metode dari contoh satu pertanyaan; gunakan seluruh dataset terverifikasi.

## 19. Kesimpulan Skenario

Dari PDF BACH 2026, sistem mengenali sumber dengan SHA-256, memakai cache valid 402 halaman, mengurangi 245.055 karakter artefak, dan membentuk 2.033 chunk. Tabel halaman 21 disimpan atomik sebagai chunk 798 karakter. Sparse BM25 memberi skor 3,919420 dan rank 97 pada query contoh, sehingga dense retrieval dan RRF diperlukan untuk membawa fakta ke konteks akhir. Setelah gate skor terbaik lolos, generator menerima Top-6 dan harus menghasilkan nilai penuh **Rp1.232.630.823.230**, yang setara sekitar **Rp1,23 triliun**, disertai halaman 21 sebagai sumber yang benar.
