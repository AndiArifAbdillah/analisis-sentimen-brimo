# Modul 08 — Latihan dan Uji Pemahaman

## Bagian A — Simulasi Pertanyaan Reviewer / Pewawancara

Jawab tanpa membuka modul lain dulu. Kunci jawaban ada di bagian bawah.

### Tingkat dasar

1. Jelaskan alur proyek ini dari data mentah sampai prediksi, dalam 5 kalimat.
2. Kenapa data dibagi menjadi training dan testing?
3. Apa itu TF-IDF dan kenapa dipakai, bukan sekadar menghitung kemunculan kata?
4. Ada berapa kelas sentimen di proyek ini, dan bagaimana menentukannya?

### Tingkat menengah

5. Kenapa kata "tidak" sengaja tidak dibuang saat stopword removal?
6. Random Forest mendapat akurasi training 99,99% tapi testing 86,50%. Apa yang terjadi?
7. Apa fungsi `stratify=y` pada `train_test_split`?
8. Kenapa `fit_transform` hanya dipakai di data training, sementara data testing memakai `transform`?
9. Apa yang dilakukan layer `Dropout` dan kenapa dibutuhkan?

### Tingkat lanjut

10. Ceritakan masalah pada kamus InSet yang Anda temukan dan bagaimana memperbaikinya.
11. Kenapa BiLSTM lebih unggul daripada SVM pada dataset ini? Sebutkan trade-off-nya.
12. Kelas netral punya f1-score terendah di ketiga model. Kenapa, dan bagaimana memperbaikinya?
13. Bagaimana Anda membuktikan bahwa pelabelan otomatis Anda valid?
14. Kalau dataset bertambah menjadi 500.000 ulasan, apa yang perlu diubah dari pipeline ini?

---

## Bagian B — Eksperimen Praktik

Semua eksperimen dijalankan di salinan notebook, bukan versi submission.

> **Peringatan waktu:** menjalankan ulang notebook dari awal memakan **±40 menit** karena
> stemming. Untuk bereksperimen, jalankan sekali sampai cell `teks_final` terbentuk, lalu
> simpan hasilnya: `df.to_csv("cache_preprocessing.csv", index=False)`. Eksperimen berikutnya
> tinggal memuat file itu dan langsung ke bagian pelatihan.

### Eksperimen 1 — Mengurangi overfitting Random Forest (mudah)

```python
model_rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=50,           # batasi kedalaman pohon
    min_samples_leaf=2,     # tiap daun minimal 2 sampel
    random_state=SEED, n_jobs=-1,
)
```
**Amati:** akurasi training pasti turun dari 99,99%. Apakah akurasi testing naik?
Kalau ya, Anda baru saja mengurangi overfitting.

### Eksperimen 2 — Efek n-gram pada SVM (mudah)

Ubah `ngram_range=(1, 2)` menjadi `(1, 1)` (unigram saja), latih ulang, bandingkan.
**Pertanyaan:** seberapa besar kontribusi bigram terhadap akurasi? Apakah sepadan dengan
bertambahnya jumlah fitur?

### Eksperimen 3 — Menyeimbangkan kelas (menengah)

Kelas positif 7× lebih banyak daripada netral. Coba beri bobot lebih pada kelas minoritas:

```python
model_svm = LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)
```
**Amati:** f1-score kelas netral biasanya naik, tapi akurasi keseluruhan bisa turun.
Diskusikan: mana yang lebih Anda butuhkan?

### Eksperimen 4 — Mempertahankan emoji (menengah)

Modifikasi `bersihkan_teks` agar emoji tidak dibuang, lalu tambahkan bobot manual:
```python
bobot_kata["😀"] = 3
bobot_kata["😡"] = -3
```
**Pertanyaan:** apakah distribusi label berubah? Apakah kelas netral berkurang?

### Eksperimen 5 — Mempercepat stemming dengan paralelisasi (menengah)

```python
from joblib import Parallel, delayed

def stem_potongan(kata_list):
    stemmer_lokal = StemmerFactory().create_stemmer()
    return {k: stemmer_lokal.stem(k) for k in kata_list}

n = 6
potongan = [list(kata_unik)[i::n] for i in range(n)]
peta_stem = {}
for hasil in Parallel(n_jobs=n)(delayed(stem_potongan)(p) for p in potongan):
    peta_stem.update(hasil)
```
Metode ini sudah diuji pada proyek: **2,5× lebih cepat dengan hasil identik**. Perhatikan
bahwa stemmer dibuat **di dalam** fungsi — objek stemmer tidak bisa dibagi antarproses.

### Eksperimen 6 — Arsitektur BiLSTM alternatif (lanjut)

```python
model_lstm = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Bidirectional(LSTM(64, return_sequences=True)),   # kembalikan seluruh urutan
    Bidirectional(LSTM(32)),                          # LSTM lapis kedua
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="softmax"),
])
```
`return_sequences=True` wajib pada LSTM pertama agar lapisan kedua menerima urutan penuh,
bukan ringkasan tunggal. **Pertanyaan:** apakah model lebih dalam selalu lebih baik pada
dataset 21 ribu sampel?

### Eksperimen 7 — Pelabelan berbasis rating (lanjut)

Ganti seluruh pendekatan pelabelan:
```python
def label_dari_rating(r):
    if r <= 2:  return "negatif"
    if r == 3:  return "netral"
    return "positif"

df["label"] = df["rating"].apply(label_dari_rating)
```
Latih ulang ketiga model. **Diskusikan:** akurasi kemungkinan **turun** — kenapa? (Petunjuk:
model membaca teks, sedangkan label berasal dari rating; ketika keduanya tidak sejalan —
"aplikasinya bagus tapi error" ⭐1 — tugas model menjadi jauh lebih sulit.) Skema ini juga
sah dipakai sebagai skema pelatihan tambahan karena berbeda di dimensi **pelabelan**.

---

## Bagian C — Kunci Jawaban Ringkas

<details>
<summary>Klik untuk membuka</summary>

1. Scraping 22.668 ulasan BRImo → preprocessing 5 tahap → pelabelan lexicon InSet 3 kelas →
   ekstraksi fitur (TF-IDF / Embedding) → latih SVM, RF, BiLSTM → evaluasi → inference.
2. Agar akurasi diukur pada data yang belum pernah dilihat model; kalau diuji dengan data latih,
   model bisa sekadar menghafal.
3. TF-IDF = frekuensi kata × kelangkaannya di seluruh dokumen. Kata yang muncul di semua dokumen
   dibobot mendekati nol karena tidak membedakan apa pun.
4. Tiga: positif, netral, negatif. Ditentukan dari jumlah bobot kata berdasarkan kamus InSet
   (>0, =0, <0).
5. Karena negasi membalik sentimen — "tidak bagus" akan berubah menjadi "bagus" kalau dibuang.
6. Overfitting: model menghafal data latih. Dikurangi dengan `max_depth`, `min_samples_leaf`,
   atau lebih banyak data.
7. Menjaga proporsi tiap kelas tetap sama di training dan testing — penting karena kelas kita
   timpang (66/25/9).
8. Agar tidak terjadi data leakage: kosakata dan nilai IDF harus dipelajari hanya dari data latih.
9. Mematikan sebagian neuron secara acak saat pelatihan agar model tidak bergantung pada
   segelintir neuron; mengurangi overfitting.
10. 1.142 kata ada di kedua file InSet dengan bobot bertentangan ("bagus" +2 vs −4). Karena
    `.update()` menimpa, semua kata ganda jadi negatif → 75% data terlabel negatif dan semua
    inference menjawab negatif. Perbaikan: muat bobot negatif dulu lalu timpa dengan positif,
    plus netralkan kata domain seperti "aplikasi" (−4 di InSet).
11. BiLSTM memahami urutan dan konteks dua arah, penting untuk negasi. Trade-off: pelatihan
    jauh lebih lama dan sulit diinterpretasi, sementara SVM selesai dalam detik.
12. Karena datanya paling sedikit (9%), definisinya berupa sisa aturan (skor tepat nol), dan
    posisinya di antara dua kelas lain. Perbaikan: `class_weight="balanced"`, tambah data netral,
    atau definisikan netral sebagai rentang skor kecil (mis. −1 sampai 1) alih-alih tepat nol.
13. Uji silang dengan rating pengguna (crosstab): proporsi label positif naik konsisten dari
    0,42 pada ⭐1 menjadi 0,82 pada ⭐5, dan negatif turun dari 0,49 ke 0,10.
14. Stemming perlu diparalelkan atau di-cache permanen; TF-IDF beralih ke `HashingVectorizer`;
    pelatihan BiLSTM sebaiknya di GPU; data dimuat bertahap (*chunking*) agar hemat memori.

</details>

---

## Bagian D — Arah Pengembangan Lanjutan

| Ide | Deskripsi | Tingkat |
|---|---|---|
| IndoBERT | Ganti BiLSTM dengan model *transformer* pra-latih bahasa Indonesia; biasanya melonjak 3–5 poin akurasi | Lanjut |
| Pelabelan hibrida | Gabungkan skor lexicon dengan rating; ambil hanya data yang keduanya sepakat sebagai label berkualitas tinggi | Menengah |
| Deteksi aspek | Bukan hanya sentimen, tapi *tentang apa* — login, transfer, tampilan (*aspect-based sentiment analysis*) | Lanjut |
| Aplikasi web | Bungkus `prediksi_sentimen()` dengan Streamlit agar bisa dicoba lewat browser | Menengah |
| Analisis tren waktu | Manfaatkan kolom `tanggal` untuk melihat pergeseran sentimen setelah pembaruan aplikasi | Menengah |

---

## Penutup

Kalau Anda bisa menjawab Bagian A tanpa membuka catatan dan sudah menjalankan minimal dua
eksperimen di Bagian B, Anda menguasai proyek ini luar-dalam — bukan sekadar tahu apa yang
dilakukan, tapi paham **kenapa** setiap keputusan diambil dan apa konsekuensinya kalau diubah.

Kembali ke [Daftar Modul](README.md)
