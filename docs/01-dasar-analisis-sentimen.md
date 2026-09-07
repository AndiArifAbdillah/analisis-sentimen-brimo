# Modul 01 — Dasar Analisis Sentimen

## 1. Apa itu Analisis Sentimen?

Analisis sentimen adalah tugas **mengklasifikasikan opini dalam teks** ke dalam kategori
emosi/sikap — pada proyek ini: `positif`, `netral`, atau `negatif`.

Contoh nyata dari dataset kita:

| Ulasan (asli dari Play Store) | Sentimen |
|---|---|
| "Aplikasi Banking terbaik di Indonesia" | positif |
| "pas saya masukan KTP masa eror terus" | negatif |
| "Baru instal aplikasinya" | netral |

Kegunaannya di dunia nyata: bank bisa memantau ribuan ulasan per hari secara otomatis untuk
tahu fitur mana yang dikeluhkan (error login? verifikasi wajah?) tanpa membaca satu per satu.

## 2. Posisi dalam Peta Ilmu

```
Artificial Intelligence
└── Machine Learning          → komputer belajar pola dari data, bukan dari aturan manual
    ├── Klasik: SVM, Random Forest        (Skema 1 & 2 proyek ini)
    └── Deep Learning: neural network      (Skema 3: BiLSTM)
        
Natural Language Processing (NLP)  → cabang AI yang mengolah bahasa manusia
└── Klasifikasi Teks
    └── Analisis Sentimen      ← PROYEK INI
```

Analisis sentimen = **klasifikasi teks** yang labelnya adalah kategori sentimen.

## 3. Supervised Learning: Cara Model "Belajar"

Ketiga model di proyek ini memakai paradigma **supervised learning** (pembelajaran terarah):

1. Kita siapkan ribuan contoh berpasangan: `(teks, label)` — misalnya `("aplikasi bagus", positif)`
2. Model mencari pola: kata/kombinasi kata apa yang cenderung muncul di tiap label
3. Setelah dilatih, model bisa menebak label untuk teks **baru** yang belum pernah dilihatnya

Dari sini muncul dua kebutuhan besar yang menjadi tulang punggung proyek:

- **Butuh label** → tapi melabeli 21 ribu ulasan manual butuh berminggu-minggu.
  Solusi proyek: pelabelan otomatis berbasis kamus (Modul 04).
- **Butuh angka** → model matematika tidak bisa membaca huruf.
  Solusi proyek: ekstraksi fitur TF-IDF dan Embedding (Modul 05).

## 4. Kenapa Data Dibagi Training dan Testing?

Analogi ujian sekolah: kalau soal ujian persis sama dengan soal latihan, nilai 100 tidak
membuktikan siswa paham — mungkin cuma hafal. Maka:

- **Training set** (misalnya 80%) → "soal latihan", dipakai model untuk belajar
- **Testing set** (20%) → "soal ujian", disembunyikan selama pelatihan, dipakai sekali
  untuk mengukur kemampuan sesungguhnya

Itulah kenapa **akurasi testing lebih penting daripada akurasi training**. Lihat hasil
Random Forest di proyek ini: training 99,99% tapi testing 86,50% — model "hafal soal latihan"
(disebut **overfitting**, dibahas di Modul 06).

Di notebook, pembagian dilakukan dengan `train_test_split(..., stratify=y)`. Parameter
`stratify=y` memastikan proporsi positif/netral/negatif di training dan testing **sama** —
penting karena kelas kita tidak seimbang (66% positif, 9% netral).

## 5. Alur Lengkap Proyek (Peta Notebook)

```
[1] Scraping Play Store  →  ulasan_brimo.csv (22.668 ulasan)          → Modul 02
[2] Preprocessing 5 tahap →  kolom teks_final yang bersih             → Modul 03
[3] Pelabelan lexicon InSet → kolom label (positif/netral/negatif)    → Modul 04
[4] Ekstraksi fitur      →  TF-IDF (Skema 1-2), Tokenizer (Skema 3)   → Modul 05
[5] Pelatihan 3 skema    →  SVM | Random Forest | BiLSTM              → Modul 06
[6] Evaluasi             →  akurasi + classification report           → Modul 07
[7] Inference            →  prediksi_sentimen("kalimat baru")         → Modul 07
```

Tiga skema sengaja dibuat **bervariasi di tiga dimensi** (algoritma, ekstraksi fitur,
pembagian data) untuk membandingkan pendekatan secara adil:

| Skema | Algoritma | Fitur | Split |
|---|---|---|---|
| 1 | SVM | TF-IDF (1-2 gram) | 80/20 |
| 2 | Random Forest | TF-IDF | 70/30 |
| 3 | BiLSTM (deep learning) | Embedding | 80/20 |

## Cek Pemahaman

1. Kenapa akurasi testing dianggap lebih jujur daripada akurasi training?
2. Apa fungsi `stratify=y` dan kenapa penting untuk dataset yang 66% isinya satu kelas?
3. Sebutkan dua "jembatan" yang dibutuhkan sebelum teks mentah bisa dilatih ke model
   supervised learning.

→ Lanjut ke [Modul 02 — Scraping Data](02-scraping-data.md)
