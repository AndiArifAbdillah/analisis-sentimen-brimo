# Analisis Sentimen Ulasan Aplikasi BRImo 🏦

Proyek analisis sentimen terhadap ulasan aplikasi mobile banking **BRImo BRI** di Google Play Store, dibangun sebagai submission kelas **Belajar Fundamental Deep Learning** Dicoding — memperoleh nilai **bintang 5** ⭐⭐⭐⭐⭐.

## Ringkasan

- **Dataset:** 22.668 ulasan unik berbahasa Indonesia, hasil scraping mandiri dari Google Play Store (±36.000 ulasan sebelum pembuangan duplikat)
- **Pelabelan:** *Lexicon-based* menggunakan kamus sentimen [InSet](https://github.com/fajri91/InSet) menjadi 3 kelas: `positif`, `netral`, `negatif`
- **Preprocessing:** cleaning, case folding, normalisasi kata tidak baku ([kamus alay](https://github.com/nasalsabila/kamus-alay)), stopword removal (dengan mempertahankan kata negasi), dan stemming (Sastrawi)

## Hasil Tiga Skema Pelatihan

| Skema | Algoritma | Ekstraksi Fitur | Split | Akurasi Training | Akurasi Testing |
|---|---|---|---|---|---|
| 1 | SVM (LinearSVC) | TF-IDF (unigram+bigram) | 80/20 | 98,99% | 91,65% |
| 2 | Random Forest | TF-IDF | 70/30 | 99,99% | 86,50% |
| 3 | **Bidirectional LSTM** | Embedding + Tokenizer | 80/20 | **97,55%** | **93,22%** |

## Temuan Menarik: Kata Ganda pada Kamus InSet

Lebih dari **1.100 kata muncul di kedua file InSet** (positif dan negatif) dengan bobot yang
bertentangan — misalnya `bagus` bernilai +2 di file positif tetapi −4 di file negatif.
Penggabungan naif membuat kata-kata pujian terhitung negatif sehingga 75% ulasan terlabel
negatif secara keliru. Proyek ini menangani hal tersebut dengan **memprioritaskan bobot positif**
untuk kata ganda dan **menetralkan kata domain aplikasi** yang tidak bermakna sentimen
(mis. `aplikasi` berbobot −4 di InSet). Detail di bagian 4 notebook.

## 📚 Panduan Belajar

Repo ini dilengkapi materi belajar berbahasa Indonesia yang membedah proyek dari fundamental
sampai detail setiap keputusan teknis — cocok untuk yang ingin memahami isinya, bukan sekadar
menjalankannya:

| Modul | Topik |
|---|---|
| [01](docs/01-dasar-analisis-sentimen.md) | Dasar analisis sentimen, supervised learning, alur proyek |
| [02](docs/02-scraping-data.md) | Scraping Play Store, pagination, kenapa deduplikasi penting |
| [03](docs/03-preprocessing-teks.md) | 5 tahap preprocessing, kenapa kata negasi dipertahankan |
| [04](docs/04-pelabelan-lexicon.md) | Pelabelan lexicon InSet + studi kasus bug 1.142 kata ganda |
| [05](docs/05-ekstraksi-fitur.md) | TF-IDF vs Embedding, data leakage, padding |
| [06](docs/06-model-dan-pelatihan.md) | Cara kerja SVM, Random Forest, BiLSTM + tiap hyperparameter |
| [07](docs/07-evaluasi-dan-inference.md) | Precision/recall/f1, kenapa kelas netral tersulit, inference |
| [08](docs/08-latihan-dan-uji-pemahaman.md) | 14 soal uji pemahaman + 7 eksperimen praktik |

→ Mulai dari [daftar isi lengkap](docs/README.md)

## Struktur Proyek

```
├── notebook_analisis_sentimen.ipynb   # Notebook utama (sudah dieksekusi, output lengkap)
├── scraping_playstore.py              # Script scraping ulasan Play Store
├── ulasan_brimo.csv                   # Dataset hasil scraping
├── requirements.txt                   # Dependensi
└── docs/                              # Materi belajar 8 modul
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
python scraping_playstore.py       # opsional: scraping ulang dataset
jupyter notebook notebook_analisis_sentimen.ipynb
```

> Catatan: eksekusi penuh notebook memakan waktu ±30–40 menit pada CPU
> (tahap terlama adalah stemming Sastrawi terhadap ±11 ribu kata unik).

## Inference

Notebook menyediakan fungsi `prediksi_sentimen(teks)` yang menerima kalimat baru dan
mengembalikan kelas kategorikal dari ketiga model sekaligus, contohnya:

```
Ulasan  : Aplikasinya sangat membantu, transfer jadi cepat dan mudah digunakan
  SVM           : positif
  Random Forest : positif
  BiLSTM        : positif
```
