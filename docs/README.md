# 📚 Panduan Belajar: Analisis Sentimen Ulasan BRImo

Materi ini membedah proyek dari nol sampai Anda memahami **setiap keputusan teknis** di dalamnya —
bukan sekadar "apa yang dilakukan", tetapi **mengapa** dilakukan seperti itu, apa alternatifnya,
dan apa yang terjadi kalau dilakukan berbeda (beberapa di antaranya benar-benar terjadi selama
pengerjaan dan menjadi studi kasus di modul-modul ini).

## Peta Belajar

| Modul | Topik | Pertanyaan yang dijawab |
|---|---|---|
| [01 — Dasar Analisis Sentimen](01-dasar-analisis-sentimen.md) | Konsep fundamental | Apa itu analisis sentimen? Di mana posisinya dalam NLP dan machine learning? |
| [02 — Scraping Data](02-scraping-data.md) | Pengumpulan data | Bagaimana 22.668 ulasan diambil dari Play Store? Kenapa perlu deduplikasi? |
| [03 — Preprocessing Teks](03-preprocessing-teks.md) | Pembersihan teks | Kenapa teks harus "dimasak" 5 tahap sebelum dipakai? Kenapa kata "tidak" sengaja TIDAK dibuang? |
| [04 — Pelabelan Lexicon](04-pelabelan-lexicon.md) | Pemberian label | Bagaimana 21 ribu ulasan diberi label tanpa manusia? Apa jebakan kamus InSet yang hampir merusak proyek ini? |
| [05 — Ekstraksi Fitur](05-ekstraksi-fitur.md) | Teks → angka | Bagaimana komputer "membaca" teks? TF-IDF vs Embedding — apa bedanya? |
| [06 — Model dan Pelatihan](06-model-dan-pelatihan.md) | SVM, RF, BiLSTM | Cara kerja ketiga algoritma + arti setiap hyperparameter di notebook |
| [07 — Evaluasi dan Inference](07-evaluasi-dan-inference.md) | Mengukur & memakai model | Membaca classification report, kenapa kelas netral paling sulit, cara kerja fungsi prediksi |
| [08 — Latihan & Uji Pemahaman](08-latihan-dan-uji-pemahaman.md) | Praktik mandiri | Eksperimen yang bisa Anda coba + simulasi pertanyaan reviewer/pewawancara |

## Cara Memakai Materi Ini

1. **Baca berurutan** — tiap modul dibangun di atas modul sebelumnya, sama seperti alur notebook.
2. **Buka notebook berdampingan** ([notebook_analisis_sentimen.ipynb](../notebook_analisis_sentimen.ipynb)) —
   setiap modul merujuk ke cell tertentu; cocokkan penjelasan dengan kode aslinya.
3. **Kerjakan modul 08 tanpa menyontek** — kalau Anda bisa menjawab semua pertanyaannya,
   Anda sudah menguasai proyek ini luar-dalam.

## Prasyarat

- Python dasar (variabel, fungsi, loop, list/dict)
- pandas dasar (DataFrame, kolom, filter) — dipakai di hampir semua cell
- Tidak perlu latar belakang NLP atau deep learning — semua dijelaskan dari nol

## Ringkasan Proyek dalam Satu Paragraf

Proyek ini mengambil **22.668 ulasan unik** aplikasi BRImo dari Google Play Store secara mandiri,
membersihkannya melalui 5 tahap preprocessing, memberi label **positif/netral/negatif** secara
otomatis menggunakan kamus sentimen InSet (dengan koreksi terhadap 1.142 kata berbobot ganda),
lalu melatih **tiga model** — SVM, Random Forest, dan Bidirectional LSTM — dengan variasi
ekstraksi fitur dan pembagian data. Model terbaik (BiLSTM) mencapai akurasi **97,55% (training)**
dan **93,22% (testing)**, dan seluruh pipeline dapat dipakai ulang untuk memprediksi sentimen
kalimat baru melalui fungsi `prediksi_sentimen()`.
