# Modul 04 — Pelabelan Data dengan Lexicon

Bagian 4 notebook. Ini modul terpenting — di sinilah proyek hampir gagal total, dan cara
memperbaikinya adalah nilai jual utama submission ini.

## 1. Masalah: Dari Mana Label Berasal?

Supervised learning butuh pasangan `(teks, label)`. Kita punya 21.306 teks, tapi **nol label**.
Tiga pilihan:

| Pendekatan | Cara kerja | Kelemahan |
|---|---|---|
| Manual | manusia membaca & melabeli satu per satu | 21.306 ulasan × 10 detik ≈ **59 jam** |
| Berdasarkan rating | bintang 1–2 = negatif, 3 = netral, 4–5 = positif | rating sering tidak sesuai isi teks ("bagus tapi sering error" ⭐1) |
| **Lexicon-based** ✅ | jumlahkan bobot sentimen tiap kata dari kamus | bergantung kualitas kamus |

Proyek ini memilih **lexicon-based** dengan kamus [InSet](https://github.com/fajri91/InSet)
(Koto & Rahmaningtyas, 2017) — kamus sentimen bahasa Indonesia berisi kata beserta bobot
numeriknya: `positive.tsv` (3.609 kata) dan `negative.tsv` (6.609 kata).

## 2. Aturan Pelabelan

```python
def beri_label(teks):
    skor = sum(bobot_kata.get(kata, 0) for kata in teks.split())
    if skor > 0:
        return "positif"
    if skor < 0:
        return "negatif"
    return "netral"
```

Skor sebuah ulasan = **jumlah bobot semua katanya**. Kata di luar kamus dihitung 0
(lagi-lagi trik `.get(kata, 0)`). Contoh:

```
"aplikasi bagus sekali"  →  0 + 2 + 1  =  +3   →  positif
"error terus kecewa"     → -5 + 0 - 4  =  -9   →  negatif
```

## 3. ⚠️ Jebakan InSet: 1.142 Kata Berbobot Ganda

Versi pertama notebook menggabungkan kedua kamus dengan cara paling intuitif:

```python
bobot_kata = dict(zip(lexicon_positif["word"], lexicon_positif["weight"]))
bobot_kata.update(dict(zip(lexicon_negatif["word"], lexicon_negatif["weight"])))   # ❌
```

Hasilnya bencana. Distribusi label yang keluar:

```
negatif    15.971   (75%)
positif     3.523
netral      1.812
```

Dan saat inference, **kelima kalimat uji diprediksi negatif** — termasuk
*"Aplikasinya sangat membantu, transfer jadi cepat dan mudah digunakan"*.

### Penyebabnya

Pemeriksaan menemukan **1.142 kata muncul di kedua file sekaligus** dengan bobot bertentangan:

| Kata | di positive.tsv | di negative.tsv | Hasil `.update()` |
|---|---|---|---|
| bagus | **+2** | −4 | **−4** ❌ |
| membantu | **+4** | −4 | **−4** ❌ |
| cepat | **+3** | −3 | **−3** ❌ |
| sangat | **+3** | −5 | **−5** ❌ |
| mudah | **+4** | −1 | **−1** ❌ |

Method `.update()` pada dictionary **menimpa** nilai yang sudah ada. Karena kamus negatif
dimuat belakangan, semua kata ganda mengambil bobot negatif. Akibatnya kalimat pujian murni
seperti *"aplikasi sangat bantu transfer cepat mudah guna"* memperoleh skor **−18** → dilabeli
negatif. Model lalu belajar dengan patuh dari label yang salah itu — sampah masuk, sampah keluar.

### Perbaikan

```python
# Bobot negatif dimuat lebih dulu, lalu DITIMPA bobot positif
bobot_kata = dict(zip(lexicon_negatif["word"], lexicon_negatif["weight"]))
bobot_kata.update(dict(zip(lexicon_positif["word"], lexicon_positif["weight"])))   # ✅

# Netralkan kata domain aplikasi yang tidak bermakna sentimen
kata_domain_netral = ["aplikasi", "bayar", "tampil", "muncul", "instal", "buka", "pakai"]
for kata in kata_domain_netral:
    bobot_kata[kata] = 0
```

Dua perubahan, dua alasan berbeda:

1. **Urutan dibalik** — untuk kata ganda, bobot positif yang menang. Pilihan ini masuk akal
   pada domain ulasan aplikasi: "bagus", "membantu", "mudah" hampir selalu dipakai sebagai
   pujian, bukan sarkasme.
2. **Netralisasi kata domain** — kata `aplikasi` punya bobot **−4** di InSet (kemungkinan dari
   konteks lain). Padahal kata ini muncul di hampir setiap ulasan, positif maupun negatif.
   Membiarkannya berarti memberi setiap ulasan "hadiah" −4 tanpa alasan. Enam kata lain
   (`bayar`, `tampil`, `muncul`, `instal`, `buka`, `pakai`) bernasib sama.

### Hasil setelah perbaikan

```
positif    14.130   (66%)
negatif     5.268   (25%)
netral      1.908   ( 9%)
```

Dan kalimat pujian tadi kini diprediksi **positif** oleh ketiga model.

## 4. Validasi: Apakah Label Ini Bisa Dipercaya?

Pelabelan otomatis wajib diverifikasi. Di sinilah kolom `rating` yang disimpan sejak scraping
(Modul 02) berguna — kita uji silang label lexicon dengan bintang yang diberikan pengguna:

```python
pd.crosstab(df["rating"], df["label"], normalize="index").round(2)
```

Hasilnya:

| Rating | negatif | netral | positif |
|---|---|---|---|
| ⭐1 | 0,49 | 0,09 | 0,42 |
| ⭐2 | 0,46 | 0,09 | 0,46 |
| ⭐3 | 0,43 | 0,09 | 0,48 |
| ⭐4 | 0,27 | 0,10 | 0,63 |
| ⭐5 | **0,10** | 0,09 | **0,82** |

**Cara membacanya:** setiap baris berjumlah 1,00 (karena `normalize="index"`). Baris ⭐5 berarti:
dari semua ulasan bintang 5, sebanyak 82% dilabeli positif dan hanya 10% dilabeli negatif.

Polanya **konsisten dan monoton**: makin tinggi rating, makin besar proporsi label positif
(0,42 → 0,46 → 0,48 → 0,63 → 0,82) dan makin kecil proporsi negatif (0,49 → 0,10). Ini bukti
kuat bahwa pelabelan menangkap sentimen nyata, bukan angka acak.

**Kenapa tidak sempurna?** Ulasan ⭐1 yang dilabeli positif (42%) sebagian besar adalah keluhan
yang ditulis dengan kata-kata sopan/positif: *"aplikasinya bagus tapi tolong perbaiki verifikasi
wajahnya"*. Ini **keterbatasan wajar** pendekatan lexicon — ia menjumlahkan kata tanpa memahami
struktur kalimat (kontras, sarkasme, kondisi). Menyadari dan mengakui keterbatasan ini penting
saat mempresentasikan proyek.

## 5. Kenapa Ini Layak Diceritakan

Bug ini tidak menyebabkan error — notebook berjalan mulus sampai akhir dengan akurasi tinggi.
Yang menyingkapnya adalah **memeriksa output inference**: model yang akurasinya 94% ternyata
menjawab "negatif" untuk semua kalimat. Akurasi tinggi terhadap label yang salah tetap saja salah.

Pelajarannya: **selalu uji model dengan contoh yang Anda tahu jawabannya**, jangan berhenti di
angka akurasi.

## Cek Pemahaman

1. Kenapa `dict.update()` menyebabkan kata "bagus" berbobot negatif? Perbaikannya bagaimana?
2. Kenapa kata `aplikasi` perlu dinetralkan padahal ada di kamus sentimen resmi?
3. Pada tabel crosstab, apa arti angka 0,82 di baris ⭐5 kolom positif?
4. Sebutkan satu jenis kalimat yang pasti salah dilabeli oleh pendekatan lexicon, dan jelaskan
   kenapa.

→ Lanjut ke [Modul 05 — Ekstraksi Fitur](05-ekstraksi-fitur.md)
