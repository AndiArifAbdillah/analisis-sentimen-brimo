# Modul 07 — Evaluasi dan Inference

## 1. Kenapa Akurasi Saja Tidak Cukup

Akurasi = proporsi prediksi yang benar. Sederhana, tapi menyesatkan pada data tidak seimbang.

Dataset kita: 66% positif, 25% negatif, 9% netral. Model yang **selalu menjawab "positif"**
tanpa berpikir akan mendapat akurasi 66% — terdengar lumayan, padahal modelnya tidak berguna.
Karena itu kita perlu metrik per kelas.

## 2. Membaca Classification Report

Hasil BiLSTM (Skema 3) pada data uji:

```
              precision    recall  f1-score   support

     negatif       0.95      0.90      0.92      1054
      netral       0.72      0.70      0.71       382
     positif       0.95      0.98      0.97      2826

    accuracy                           0.93      4262
   macro avg       0.87      0.86      0.87      4262
weighted avg       0.93      0.93      0.93      4262
```

### Tiga metrik inti

**Precision** — dari semua yang **diprediksi** kelas X, berapa persen yang benar?
> Precision negatif 0,95 → saat model bilang "negatif", 95% kasus memang negatif.
> Menjawab: *"seberapa bisa saya percaya kalau model bilang X?"*

**Recall** — dari semua yang **sebenarnya** kelas X, berapa persen yang berhasil ditemukan?
> Recall negatif 0,90 → dari seluruh ulasan negatif asli, 90% berhasil terdeteksi (10% lolos).
> Menjawab: *"seberapa banyak kasus X yang tidak terlewat?"*

**F1-score** — rata-rata harmonik precision dan recall; ringkasan tunggal yang adil.
Nilainya tinggi hanya kalau **kedua** metrik tinggi.

**Support** — jumlah sampel asli kelas tersebut di data uji (1054 + 382 + 2826 = 4262).

### Analogi memilah email spam

- **Precision rendah** → banyak email penting masuk folder spam (model asal menuduh)
- **Recall rendah** → banyak spam lolos ke inbox (model terlalu longgar)

Mana yang lebih penting bergantung konteks. Untuk memantau keluhan nasabah, **recall negatif**
lebih kritis — lebih baik kelebihan menandai keluhan daripada melewatkan masalah nyata.

### Macro avg vs weighted avg

- **macro avg** = rata-rata biasa ketiga kelas, **tanpa** memperhitungkan jumlah sampel.
  Kelas netral yang kecil punya bobot sama dengan positif yang besar → 0,87
- **weighted avg** = rata-rata tertimbang jumlah sampel → 0,93, dekat dengan akurasi

Selisih keduanya (0,87 vs 0,93) menunjukkan ada kelas yang performanya di bawah rata-rata —
dalam hal ini netral.

## 3. Kenapa Kelas Netral Paling Sulit?

Bandingkan f1-score kelas netral di ketiga skema:

| Skema | f1 negatif | f1 netral | f1 positif |
|---|---|---|---|
| SVM | 0,88 | **0,70** | 0,95 |
| Random Forest | 0,78 | **0,69** | 0,91 |
| BiLSTM | 0,92 | **0,71** | 0,97 |

Konsisten paling rendah di semua model. Tiga penyebabnya:

1. **Datanya paling sedikit** — hanya 1.908 sampel (9%), model punya sedikit contoh untuk belajar
2. **Definisinya paling rapuh** — netral bukan "sentimen netral" melainkan sisa aturan:
   skor lexicon **tepat nol**. Bisa karena ulasan memang datar ("baru instal"), atau karena
   kata positif dan negatifnya kebetulan saling meniadakan
3. **Posisinya di tengah** — netral berbatasan dengan positif *dan* negatif, sehingga batas
   keputusannya paling kabur

Ini keterbatasan yang wajar diakui, bukan disembunyikan. Perbaikannya dibahas di Modul 08.

## 4. Perbandingan Ketiga Skema

```python
perbandingan = pd.DataFrame(
    [(nama, train, test) for nama, (train, test) in hasil_skema.items()],
    columns=["Skema Pelatihan", "Akurasi Training", "Akurasi Testing"],
)
```

| Skema | Training | Testing | Selisih | Catatan |
|---|---|---|---|---|
| SVM + TF-IDF (80/20) | 98,99% | 91,65% | 7,3 | cepat, mudah dijelaskan |
| RF + TF-IDF (70/30) | 99,99% | 86,50% | **13,5** | overfitting paling kentara |
| BiLSTM (80/20) | 97,55% | **93,22%** | **4,3** | terbaik & paling seimbang |

Cara membacanya: **selisih kecil = generalisasi baik**. BiLSTM unggul bukan hanya karena
akurasi tertinggi, tapi karena jaraknya ke akurasi training paling sempit.

## 5. Inference: Memakai Model pada Kalimat Baru

Inference adalah tujuan akhir — model dipakai memprediksi teks yang belum pernah dilihat.

```python
def preprocessing_inference(teks):
    teks = bersihkan_teks(teks)
    teks = normalisasi_slang(teks)
    teks = hapus_stopword(teks)
    return " ".join(peta_stem.get(kata, stemmer.stem(kata)) for kata in teks.split())
```

**Prinsip terpenting:** kalimat baru **wajib** melewati preprocessing yang **persis sama**
dengan data latih. Kalau model dilatih pada teks yang sudah di-stem, lalu diberi teks mentah,
kosakatanya tidak akan cocok dan prediksinya kacau.

Perhatikan `peta_stem.get(kata, stemmer.stem(kata))`: pakai cache kalau kata pernah muncul saat
pelatihan; kalau kata benar-benar baru, stem saat itu juga.

```python
def prediksi_sentimen(teks):
    teks_olahan = preprocessing_inference(teks)

    pred_svm = model_svm.predict(tfidf_1.transform([teks_olahan]))[0]
    pred_rf  = model_rf.predict(tfidf_2.transform([teks_olahan]))[0]

    sekuens_baru = tokenizer.texts_to_sequences([teks_olahan])
    pad_baru = pad_sequences(sekuens_baru, maxlen=100, padding="post", truncating="post")
    probabilitas = model_lstm.predict(pad_baru, verbose=0)
    pred_lstm = label_encoder.classes_[np.argmax(probabilitas)]

    return {"SVM": pred_svm, "Random Forest": pred_rf, "BiLSTM": pred_lstm}
```

Tiap model memakai transformer miliknya sendiri (`tfidf_1` untuk SVM, `tfidf_2` untuk RF,
`tokenizer` untuk BiLSTM) — tidak boleh tertukar, karena masing-masing punya kosakata berbeda.

Baris `label_encoder.classes_[np.argmax(probabilitas)]` menerjemahkan output softmax kembali
menjadi teks: `np.argmax` mengambil indeks probabilitas tertinggi, lalu indeks itu dipetakan
ke nama kelas.

## 6. Hasil Inference dan Cara Menyikapinya

| Kalimat uji | SVM | RF | BiLSTM |
|---|---|---|---|
| "Aplikasinya sangat membantu, transfer jadi cepat dan mudah digunakan" | positif | positif | positif |
| "Aplikasi sering error dan lambat, saldo tidak muncul, sangat mengecewakan" | negatif | negatif | negatif |
| "Baru instal aplikasinya, belum tahu bagaimana cara pakainya" | positif | positif | positif |
| "Fitur pembayarannya lengkap dan tampilannya bagus sekali" | positif | positif | positif |
| "Sudah update tapi masih gagal login terus, tolong diperbaiki" | negatif | negatif | **netral** |

**Yang benar:** kalimat 1, 2, dan 4 diprediksi tepat oleh ketiga model — pujian → positif,
keluhan → negatif. Bandingkan dengan versi sebelum perbaikan pelabelan (Modul 04), di mana
kelima kalimat dijawab "negatif".

**Yang perlu dikritisi — kalimat 3.** *"Baru instal aplikasinya, belum tahu cara pakainya"*
idealnya **netral**, tapi ketiganya menjawab positif. Kenapa? Setelah preprocessing tersisa
kata seperti "instal", "belum", "tahu", "cara" — hampir semuanya netral atau tidak ada di
kamus, sehingga skornya tidak jelas. Ditambah bias data (66% positif), model condong menebak
positif saat ragu.

**Yang menarik — kalimat 5.** BiLSTM menjawab "netral" sementara dua model lain "negatif".
Kalimat ini memang campuran: ada keluhan ("gagal login") sekaligus harapan sopan
("tolong diperbaiki"). Ketidaksepakatan antarmodel pada kalimat ambigu adalah hal normal dan
justru informatif.

Menyajikan hasil apa adanya — termasuk yang meleset — jauh lebih kredibel daripada hanya
memamerkan contoh yang berhasil.

## Cek Pemahaman

1. Model selalu menjawab "positif" pada dataset kita. Berapa akurasinya, dan kenapa angka itu menipu?
2. Jelaskan beda precision dan recall dengan kalimat Anda sendiri.
3. Kenapa macro avg (0,87) lebih rendah daripada weighted avg (0,93)?
4. Apa yang terjadi kalau kalimat baru langsung diprediksi tanpa preprocessing?
5. Kenapa `tfidf_1` tidak boleh dipakai untuk model Random Forest?

→ Lanjut ke [Modul 08 — Latihan & Uji Pemahaman](08-latihan-dan-uji-pemahaman.md)
