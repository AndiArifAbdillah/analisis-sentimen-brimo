# Modul 03 — Preprocessing Teks

Bagian 3 notebook. Input: kolom `ulasan` (teks mentah). Output: kolom `teks_final` (teks bersih siap dilatih).

## 1. Kenapa Teks Harus "Dimasak" Dulu?

Bagi manusia, tiga teks ini sama maknanya:

```
"Aplikasinya BAGUS bgt!!! 😍"
"aplikasi nya bagus banget"
"Aplikasi'y bagus bgt 100%"
```

Bagi komputer, ketiganya adalah **tiga string yang sama sekali berbeda**. Kalau langsung dilatih,
model harus belajar "bgt", "banget", "bgt!!!" sebagai tiga kata terpisah — pemborosan besar dan
membuat model lemah. Preprocessing menyeragamkan variasi tersebut menjadi satu bentuk baku.

Efeknya nyata di proyek ini: dari 22.668 ulasan, setelah pembersihan tersisa **21.326** baris
(ada yang jadi kosong setelah emoji dibuang, ada yang jadi duplikat setelah diseragamkan),
lalu **21.306** setelah stemming.

## 2. Tahap 1 — Cleaning dan Case Folding

```python
def bersihkan_teks(teks):
    teks = str(teks)
    teks = re.sub(r"@[\w]+", " ", teks)                 # hapus mention
    teks = re.sub(r"#[\w]+", " ", teks)                 # hapus hashtag
    teks = re.sub(r"https?://\S+|www\.\S+", " ", teks)  # hapus URL
    teks = re.sub(r"\d+", " ", teks)                    # hapus angka
    teks = re.sub(r"[^\w\s]", " ", teks)                # hapus tanda baca & emoji
    teks = teks.replace("_", " ")
    teks = re.sub(r"\s+", " ", teks).strip()            # rapikan spasi ganda
    return teks.lower()                                 # case folding
```

**Membaca regex-nya:**

| Pola | Arti | Contoh yang kena |
|---|---|---|
| `@[\w]+` | `@` diikuti satu huruf/angka atau lebih | `@brimo` |
| `https?://\S+` | `http` atau `https` (`s?` = s opsional) diikuti karakter non-spasi | `https://bri.co.id` |
| `\d+` | satu digit atau lebih | `100`, `2024` |
| `[^\w\s]` | apa pun yang **bukan** (`^`) huruf/angka/underscore (`\w`) dan bukan spasi (`\s`) | `!`, `.`, `😍` |
| `\s+` | satu spasi/tab/newline atau lebih | spasi ganda hasil penghapusan di atas |

**Kenapa emoji dibuang?** Emoji sebenarnya sinyal sentimen yang kuat (😍 vs 😡). Tapi kamus
sentimen InSet yang kita pakai (Modul 04) hanya berisi kata, bukan emoji — jadi emoji tidak akan
berkontribusi ke skor label. Membuangnya membuat pipeline konsisten. Ini **trade-off sadar**,
bukan kelalaian; menyimpan emoji adalah salah satu ide pengembangan di Modul 08.

**Kenapa angka dibuang?** Pada ulasan aplikasi, angka umumnya berupa versi ("versi 2.1"), kode
error, atau nominal — jarang membawa sentimen, tapi menambah ribuan token unik yang tidak berguna.

**Kenapa `.lower()`?** Agar "BAGUS", "Bagus", dan "bagus" dihitung sebagai satu kata yang sama.

## 3. Tahap 2 — Normalisasi Kata Tidak Baku (Slang)

Bahasa ulasan Indonesia penuh singkatan: "gk", "bgt", "gmn", "trs". Kita memakai kamus
[colloquial-indonesian-lexicon](https://github.com/nasalsabila/kamus-alay) berisi **4.331 entri**
pemetaan slang → bentuk baku.

```python
peta_slang = dict(zip(kamus_slang["slang"], kamus_slang["formal"]))

def normalisasi_slang(teks):
    return " ".join(peta_slang.get(kata, kata) for kata in teks.split())
```

Kunci pemahaman ada di `peta_slang.get(kata, kata)`: method `.get()` pada dictionary menerima
**nilai default** sebagai argumen kedua. Jadi baris ini berarti *"ganti kata dengan bentuk
bakunya kalau ada di kamus; kalau tidak ada, biarkan kata itu apa adanya"*. Tanpa default,
kata yang tidak ada di kamus akan menjadi `None` dan merusak kalimat.

## 4. Tahap 3 — Stopword Removal (dengan Pengecualian Penting)

**Stopword** adalah kata yang sangat sering muncul tapi minim makna: "yang", "di", "dan", "saya".
Membuangnya mengurangi noise dan mempercepat pelatihan.

```python
kata_negasi = {"tidak", "bukan", "jangan", "belum", "kurang", "tanpa", "gagal"}
daftar_stopword = set(StopWordRemoverFactory().get_stop_words()) - kata_negasi
```

Baris kedua adalah **keputusan desain paling penting di modul ini**. Daftar stopword bawaan
Sastrawi memasukkan "tidak" dan "belum" sebagai kata tak bermakna. Untuk analisis sentimen itu
bencana:

```
"aplikasi ini tidak bagus"  →  buang "tidak"  →  "aplikasi bagus"   ❌ makna terbalik!
```

Operasi `set(...) - kata_negasi` adalah **selisih himpunan** (set difference): ambil semua
stopword Sastrawi, lalu keluarkan 7 kata negasi dari daftar tersebut sehingga kata-kata itu
selamat dari penghapusan. Negasi adalah pembalik polaritas sentimen — harus dipertahankan.

## 5. Tahap 4 — Stemming

**Stemming** mengubah kata berimbuhan ke kata dasarnya, memakai library
[Sastrawi](https://github.com/sastrawi/sastrawi) (khusus bahasa Indonesia):

```
"menggunakan", "digunakan", "kegunaan"  →  "guna"
"perhatikan", "memperhatikan"           →  "perhati"
"pembayarannya"                         →  "bayar"
```

Tanpa stemming, model harus belajar "membantu", "dibantu", "bantuan" sebagai tiga fitur berbeda
padahal maknanya satu. Stemming memadatkan kosakata sehingga model belajar lebih efisien dari
data yang sama.

### Optimasi yang wajib dipahami

```python
kata_unik = set()
for teks in df["teks_tanpa_stopword"]:
    kata_unik.update(teks.split())          # → 11.240 kata unik

peta_stem = {kata: stemmer.stem(kata) for kata in kata_unik}   # stem SEKALI per kata unik

def stemming_teks(teks):
    return " ".join(peta_stem.get(kata, kata) for kata in teks.split())
```

Kenapa tidak langsung `df["teks"].apply(stemmer.stem)` saja? Karena 21.306 ulasan mengandung
**sekitar 300.000 kata** secara total, tetapi hanya **11.240 kata unik**. Kata "aplikasi" muncul
belasan ribu kali — percuma di-stem berulang-ulang dengan hasil yang selalu sama.

Angka nyatanya: pada laptop yang dipakai, Sastrawi butuh **±0,2 detik per kata**. Dengan cache:
11.240 × 0,2 detik ≈ **38 menit**. Tanpa cache: 300.000 × 0,2 detik ≈ **16 jam**. Cache dictionary
sederhana ini adalah pembeda antara proyek yang selesai dan yang tidak.

> **Catatan lapangan:** stemming adalah tahap paling lambat di seluruh notebook. Kalau Anda
> menjalankan ulang, bersiaplah menunggu ±40 menit di cell ini — itu normal, bukan macet.
> Alternatif yang sudah teruji: paralelisasi dengan `joblib` (2,5× lebih cepat, hasil identik).

## 6. Ringkasan Transformasi

Satu ulasan asli melewati kelima tahap:

| Tahap | Hasil |
|---|---|
| Asli | `Sudah berulang kali saya mencoba melakukan verifikasi wajah` |
| Cleaning | `sudah berulang kali saya mencoba melakukan verifikasi wajah` |
| Slang | *(tidak berubah — semua kata sudah baku)* |
| Stopword | `berulang kali mencoba melakukan verifikasi wajah` |
| Stemming | `ulang kali coba laku verifikasi wajah` |

## Cek Pemahaman

1. Apa yang terjadi pada kalimat "aplikasi ini tidak bagus" kalau kata negasi ikut dibuang?
   Kenapa itu fatal untuk analisis sentimen?
2. Jelaskan arti argumen kedua pada `peta_slang.get(kata, kata)`.
3. Hitung sendiri: kalau ada 500.000 total kata dengan 15.000 kata unik dan stemming butuh
   0,2 detik/kata, berapa selisih waktu antara memakai cache dan tidak?

→ Lanjut ke [Modul 04 — Pelabelan Lexicon](04-pelabelan-lexicon.md)
