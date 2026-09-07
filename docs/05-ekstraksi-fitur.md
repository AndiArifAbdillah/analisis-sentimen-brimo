# Modul 05 — Ekstraksi Fitur: Mengubah Teks Menjadi Angka

Model machine learning adalah fungsi matematika — ia hanya menerima angka. Modul ini membahas
dua cara menerjemahkan teks menjadi angka yang dipakai proyek ini: **TF-IDF** (Skema 1 & 2) dan
**Tokenizer + Embedding** (Skema 3).

## 1. TF-IDF

### Konsep dasar: Bag of Words

Cara paling sederhana: hitung berapa kali tiap kata muncul. Dengan kosakata
`[aplikasi, bagus, error]`, kalimat "aplikasi bagus" menjadi vektor `[1, 1, 0]`.

Disebut *bag of words* (kantong kata) karena urutan diabaikan — "tidak bagus" dan "bagus tidak"
menghasilkan vektor identik. Ini kelemahan mendasarnya.

### Dari hitungan mentah ke TF-IDF

Masalah hitungan mentah: kata yang muncul di **semua** dokumen (misalnya "aplikasi") mendapat
nilai besar padahal tidak membedakan apa pun. TF-IDF mengoreksinya dengan dua komponen:

**TF (Term Frequency)** — seberapa sering kata muncul dalam satu dokumen.
Makin sering → makin penting **untuk dokumen itu**.

**IDF (Inverse Document Frequency)** — seberapa langka kata di seluruh koleksi dokumen:

```
IDF(kata) = log( jumlah_dokumen / jumlah_dokumen_yang_mengandung_kata )
```

Kata yang ada di semua dokumen → rasionya mendekati 1 → log(1) = 0 → **bobot nol**.
Kata langka → rasio besar → bobot besar.

**TF-IDF = TF × IDF.** Hasilnya: kata yang sering muncul di satu ulasan tapi jarang di ulasan
lain mendapat bobot tertinggi — persis kata-kata yang membedakan satu sentimen dari lainnya.

### Implementasi di notebook

```python
# Skema 1
tfidf_1 = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

# Skema 2
tfidf_2 = TfidfVectorizer(max_features=5000)
```

| Parameter | Arti | Kenapa dipilih |
|---|---|---|
| `max_features=10000` | ambil 10.000 fitur paling sering saja | membatasi dimensi agar pelatihan cepat & hemat memori |
| `ngram_range=(1, 2)` | pakai unigram **dan** bigram | agar "tidak bagus" tertangkap sebagai satu fitur utuh |
| `max_features=5000` (Skema 2) | lebih sedikit fitur | Random Forest lambat pada dimensi tinggi; ini juga sengaja dibedakan agar skema benar-benar bervariasi |

**Bigram** adalah pasangan kata berurutan. Dengan `ngram_range=(1,2)`, kalimat "tidak bagus"
menghasilkan fitur: `["tidak", "bagus", "tidak bagus"]`. Fitur ketiga itulah yang sebagian
menambal kelemahan bag-of-words terhadap negasi.

### Aturan emas: fit hanya pada training

```python
X_train_tfidf_1 = tfidf_1.fit_transform(X_train_1)   # fit + transform
X_test_tfidf_1  = tfidf_1.transform(X_test_1)        # transform saja
```

`fit_transform` pada training = pelajari kosakata & nilai IDF, lalu ubah jadi angka.
`transform` pada testing = pakai kosakata & IDF **yang sudah dipelajari tadi**.

Kalau testing ikut di-`fit`, informasi dari data uji bocor ke proses pelatihan
(**data leakage**) dan akurasi testing menjadi tidak sah. Kata di data uji yang tak dikenal
memang akan diabaikan — dan itu justru realistis, karena di dunia nyata model pasti bertemu
kata baru.

## 2. Tokenizer + Embedding (untuk BiLSTM)

Deep learning butuh representasi berbeda: **urutan kata harus dipertahankan**, karena itulah
kekuatan utama LSTM.

### Langkah 1 — Tokenizer: kata menjadi nomor

```python
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sekuens = tokenizer.texts_to_sequences(X)
```

Setiap kata diberi nomor indeks berdasarkan frekuensi (kata tersering = nomor kecil):

```
"aplikasi bagus"  →  [3, 27]
```

`oov_token="<OOV>"` (*out of vocabulary*) adalah nomor khusus untuk kata yang tidak dikenal.
Tanpa ini, kata asing akan dilewati diam-diam sehingga kalimat bisa menyusut atau menjadi kosong.

### Langkah 2 — Padding: menyeragamkan panjang

```python
X_pad = pad_sequences(sekuens, maxlen=100, padding="post", truncating="post")
```

Neural network membutuhkan input berukuran tetap, padahal ulasan panjangnya bervariasi.
Solusinya: semua dipaksa menjadi 100 token.

- Kalimat pendek → ditambah angka 0 di belakang (`padding="post"`)
- Kalimat panjang → dipotong di belakang (`truncating="post"`)

```
[3, 27]  →  [3, 27, 0, 0, 0, ..., 0]      (total 100 angka)
```

Kenapa `maxlen=100`? Mayoritas ulasan Play Store jauh lebih pendek dari 100 kata, jadi angka ini
menampung hampir semua ulasan utuh tanpa membuat matriks terlalu besar.

### Langkah 3 — Embedding: nomor menjadi vektor bermakna

Nomor indeks masih tidak bermakna — kata nomor 50 tidak "dua kali lipat" kata nomor 25.
Layer `Embedding` mengubah tiap nomor menjadi vektor berdimensi 128 yang **dipelajari selama
pelatihan**:

```python
Embedding(input_dim=10000, output_dim=128)
```

Setelah terlatih, kata dengan makna serupa akan menempati posisi berdekatan di ruang 128 dimensi
tersebut — "bagus" dekat dengan "mantap", jauh dari "error". Inilah keunggulan embedding
dibanding TF-IDF: ia menangkap **kemiripan makna**, bukan sekadar kemunculan kata.

## 3. Perbandingan Kedua Pendekatan

| Aspek | TF-IDF | Embedding |
|---|---|---|
| Urutan kata | diabaikan (kecuali lewat n-gram) | **dipertahankan penuh** |
| Kemiripan makna | tidak ditangkap | ditangkap |
| Dimensi | besar & jarang (*sparse*), 10.000 kolom | padat (*dense*), 128 dimensi |
| Kebutuhan data | bisa bekerja dengan data kecil | butuh data banyak |
| Kecepatan latih | detik–menit | menit–jam |
| Interpretasi | mudah (bobot per kata terlihat) | sulit (kotak hitam) |

Karena itulah proyek ini memakai keduanya: TF-IDF untuk model klasik yang cepat dan mudah
dijelaskan, embedding untuk model deep learning yang lebih akurat.

## Cek Pemahaman

1. Kenapa kata yang muncul di semua dokumen mendapat bobot IDF mendekati nol?
2. Apa akibatnya kalau `tfidf.fit_transform()` dipanggil pada data testing?
3. Kenapa BiLSTM butuh padding sementara SVM tidak?
4. Sebutkan satu hal yang bisa ditangkap Embedding tapi tidak bisa ditangkap TF-IDF.

→ Lanjut ke [Modul 06 — Model dan Pelatihan](06-model-dan-pelatihan.md)
