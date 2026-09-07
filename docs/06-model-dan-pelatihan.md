# Modul 06 — Tiga Model dan Cara Kerjanya

Bagian 5 notebook. Ketiga skema sengaja berbeda di **tiga dimensi** sekaligus (algoritma,
ekstraksi fitur, pembagian data) agar perbandingannya bermakna.

| Skema | Algoritma | Fitur | Split | Train | Test |
|---|---|---|---|---|---|
| 1 | SVM (LinearSVC) | TF-IDF 1-2 gram, 10k fitur | 80/20 | 98,99% | 91,65% |
| 2 | Random Forest | TF-IDF, 5k fitur | 70/30 | 99,99% | 86,50% |
| 3 | **BiLSTM** | Embedding 128 dim | 80/20 | **97,55%** | **93,22%** |

## 1. Skema 1 — Support Vector Machine (SVM)

### Cara kerja

SVM mencari **garis pemisah** (dalam dimensi tinggi disebut *hyperplane*) antar kelas, dengan
prinsip: pilih garis yang jaraknya ke titik data terdekat dari masing-masing kelas **sejauh
mungkin**. Jarak itu disebut *margin*, dan titik-titik terdekat yang menentukan posisi garis
disebut *support vector* — dari situlah namanya.

Kenapa margin lebar itu bagus? Karena garis yang "longgar" lebih tahan terhadap data baru yang
posisinya sedikit bergeser. Ini membuat SVM cenderung menggeneralisasi dengan baik.

SVM sangat cocok untuk teks karena TF-IDF menghasilkan **dimensi tinggi tapi jarang** (10.000
kolom, mayoritas nol) — kondisi di mana pemisahan linear justru mudah ditemukan.

```python
model_svm = LinearSVC(C=1.0, random_state=SEED)
```

- `LinearSVC` — varian SVM dengan kernel linear, jauh lebih cepat untuk data teks berdimensi besar
- `C=1.0` — parameter **regularisasi**. Nilai kecil = margin lebar tapi boleh salah lebih banyak
  (model lebih sederhana); nilai besar = memaksa semua data benar (rawan overfitting).
  1.0 adalah nilai default yang seimbang.
- `random_state=SEED` — mengunci keacakan agar hasil bisa direproduksi

**Hasil:** training 98,99%, testing 91,65% — selisih 7,3 poin, wajar.

## 2. Skema 2 — Random Forest

### Cara kerja

Bayangkan **satu pohon keputusan**: serangkaian pertanyaan bertingkat.
*"Apakah kata 'error' muncul? Ya → apakah kata 'tidak' muncul? ..."* sampai berujung pada
sebuah label. Satu pohon mudah dipahami, tapi gampang menghafal data latih.

**Random Forest** = ratusan pohon, masing-masing dilatih pada:
- sampel data yang diacak (*bootstrap*), dan
- subset fitur yang diacak

Prediksi akhir = **suara terbanyak** dari semua pohon. Kesalahan tiap pohon bersifat acak dan
saling meniadakan, sementara pola yang benar diperkuat — inilah prinsip *ensemble*.

```python
model_rf = RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1)
```

- `n_estimators=150` — jumlah pohon. Makin banyak makin stabil, tapi makin lambat
- `n_jobs=-1` — pakai semua inti CPU secara paralel

**Hasil:** training **99,99%**, testing **86,50%** — selisih **13,5 poin**.

### Studi kasus overfitting

Angka 99,99% berarti Random Forest nyaris menghafal seluruh data latih. Ini karakter khasnya:
tanpa batas kedalaman, pohon akan terus bercabang sampai tiap sampel latih masuk daun sendiri.

Yang perlu dipahami: **akurasi training tinggi bukan prestasi** — yang diuji adalah kemampuan
pada data yang belum pernah dilihat. Selisih besar antara training dan testing adalah tanda
*overfitting*.

Cara menguranginya (bahan eksperimen di Modul 08): batasi `max_depth`, naikkan
`min_samples_leaf`, atau tambah jumlah pohon.

Meski begitu, 86,50% tetap **lolos syarat minimal 85%** dan skema ini berguna justru sebagai
pembanding — menunjukkan bahwa algoritma berbeda punya kecenderungan berbeda pada data yang sama.

## 3. Skema 3 — Bidirectional LSTM (Deep Learning)

### Dari RNN ke LSTM

**RNN** (Recurrent Neural Network) memproses kata satu per satu sambil membawa "memori" dari
kata sebelumnya — cocok untuk data berurutan seperti kalimat. Kelemahannya: memori memudar
pada kalimat panjang (*vanishing gradient*).

**LSTM** (Long Short-Term Memory) memperbaikinya dengan mekanisme **gerbang** (*gate*) yang
mengatur informasi mana yang disimpan, dilupakan, dan diteruskan. Hasilnya, konteks jauh
tetap terjaga.

### Kenapa "Bidirectional"?

LSTM biasa membaca kiri → kanan. **BiLSTM** menjalankan dua LSTM sekaligus: satu maju, satu
mundur, lalu hasil keduanya digabung.

Kenapa penting? Bandingkan:

```
"tidak ada masalah sama sekali"     → positif
"ada masalah, tidak bisa dipakai"   → negatif
```

Saat membaca kata "masalah", arah maju belum tahu apa yang mengikutinya. Arah mundur sudah
"melihat" akhir kalimat. Kombinasi keduanya memberi konteks penuh dari dua sisi — inilah yang
membuat BiLSTM unggul untuk sentimen bahasa Indonesia yang kaya negasi.

### Arsitektur di notebook

```python
model_lstm = Sequential([
    Embedding(input_dim=10000, output_dim=128),   # 1. kata → vektor 128 dimensi
    Bidirectional(LSTM(64)),                      # 2. baca 2 arah, 64 unit tiap arah → 128 output
    Dropout(0.5),                                 # 3. matikan 50% neuron acak saat latih
    Dense(64, activation="relu"),                 # 4. lapisan padat 64 neuron
    Dropout(0.3),                                 # 5. matikan 30% neuron acak
    Dense(3, activation="softmax"),               # 6. 3 kelas keluaran
])
```

Penjelasan tiap lapisan:

| # | Lapisan | Fungsi |
|---|---|---|
| 1 | `Embedding` | mengubah nomor kata menjadi vektor bermakna (Modul 05) |
| 2 | `Bidirectional(LSTM(64))` | membaca urutan dari dua arah, meringkas kalimat jadi 128 angka |
| 3, 5 | `Dropout` | **anti-overfitting**: mematikan neuron acak tiap langkah latih agar model tidak bergantung pada segelintir neuron |
| 4 | `Dense(64, relu)` | mengolah ringkasan menjadi ciri tingkat tinggi. ReLU = `max(0, x)`, menambah kemampuan non-linear |
| 6 | `Dense(3, softmax)` | menghasilkan 3 probabilitas berjumlah 1, misalnya `[0,05 negatif; 0,10 netral; 0,85 positif]` |

### Kompilasi

```python
model_lstm.compile(optimizer="adam",
                   loss="categorical_crossentropy",
                   metrics=["accuracy"])
```

- **`adam`** — algoritma optimasi yang menyesuaikan laju belajar secara otomatis; default terbaik
  untuk sebagian besar kasus
- **`categorical_crossentropy`** — fungsi kerugian untuk klasifikasi multi-kelas dengan label
  one-hot. Ia menghukum keras prediksi yang **percaya diri tapi salah**
- Label diubah ke one-hot lebih dulu: `to_categorical()` mengubah `positif` menjadi `[0, 0, 1]`

### Pelatihan dan Early Stopping

```python
early_stop = EarlyStopping(monitor="val_accuracy", patience=3,
                           restore_best_weights=True, verbose=1)

riwayat = model_lstm.fit(X_train_3, y_train_3,
                         validation_split=0.1,
                         epochs=12, batch_size=256,
                         callbacks=[early_stop], verbose=2)
```

| Parameter | Arti |
|---|---|
| `validation_split=0.1` | sisihkan 10% data latih sebagai validasi — untuk memantau overfitting **tanpa** menyentuh data uji |
| `epochs=12` | maksimal 12 kali putaran seluruh data |
| `batch_size=256` | bobot diperbarui tiap 256 sampel (bukan tiap sampel) → lebih cepat & stabil |
| `patience=3` | berhenti kalau akurasi validasi tidak membaik selama 3 epoch berturut-turut |
| `restore_best_weights=True` | **kembalikan bobot terbaik**, bukan bobot terakhir |

Early stopping adalah pertahanan utama terhadap overfitting: model berhenti tepat sebelum ia
mulai menghafal. Pada run final, pelatihan berhenti sebelum epoch ke-12 dan bobot dikembalikan
ke epoch terbaik.

**Hasil:** training 97,55%, testing 93,22% — selisih hanya **4,3 poin**, terkecil dari ketiga
skema. Model ini paling seimbang sekaligus paling akurat.

## 4. Kenapa Deep Learning Menang di Sini?

| Faktor | Penjelasan |
|---|---|
| Memahami urutan | "tidak bagus" ≠ "bagus tidak"; TF-IDF hanya menambalnya lewat bigram |
| Konteks dua arah | negasi di awal maupun akhir kalimat sama-sama tertangkap |
| Embedding terlatih | kata bermakna serupa dipetakan berdekatan |
| Data cukup besar | 21.306 sampel memadai untuk melatih neural network |

Tapi perhatikan trade-off-nya: SVM selesai dilatih dalam hitungan detik dan bobot per katanya
bisa diperiksa; BiLSTM butuh belasan menit di CPU dan sulit dijelaskan isinya. Pada kasus nyata,
selisih 1,6 poin akurasi belum tentu sepadan dengan biaya itu.

## 5. Reproducibility

```python
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
```

Tiga library punya generator acak sendiri, jadi ketiganya harus dikunci. Tanpa ini, setiap
eksekusi memberi hasil sedikit berbeda dan angka di laporan tidak bisa dipertanggungjawabkan.

## Cek Pemahaman

1. Apa arti "margin" pada SVM dan kenapa margin lebar diinginkan?
2. Random Forest mencapai training 99,99% tapi testing 86,50%. Sebutkan istilahnya dan dua cara
   menguranginya.
3. Kenapa BiLSTM lebih baik daripada LSTM satu arah untuk kalimat bernegasi?
4. Apa beda `validation_split` dengan data testing?
5. Kenapa `restore_best_weights=True` penting saat memakai early stopping?

→ Lanjut ke [Modul 07 — Evaluasi dan Inference](07-evaluasi-dan-inference.md)
