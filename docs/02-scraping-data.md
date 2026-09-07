# Modul 02 — Scraping Data dari Google Play Store

File terkait: [`scraping_playstore.py`](../scraping_playstore.py) → menghasilkan [`ulasan_brimo.csv`](../ulasan_brimo.csv)

## 1. Kenapa Scraping Mandiri?

Syarat submission Dicoding melarang dataset open source yang tinggal unduh — data harus
dikumpulkan sendiri. Selain itu, scraping mandiri berarti kita mengontrol penuh: aplikasi mana,
bahasa apa, berapa banyak, dan seberapa segar datanya.

**Kenapa BRImo?** Aplikasi mobile banking dengan jutaan ulasan berbahasa Indonesia — dijamin
cukup untuk target 10.000+ sampel, opininya beragam (pujian kemudahan transfer vs keluhan
error login), dan temanya relevan dengan konteks perbankan Indonesia.

## 2. Library: google-play-scraper

Kita memakai library `google-play-scraper` (Python) yang mengambil data ulasan melalui
endpoint yang sama dengan yang dipakai halaman web Play Store — tanpa login, tanpa API key.

Fungsi kuncinya:

```python
hasil, token_lanjutan = reviews(
    app_id,                 # "id.co.bri.brimo" — ID unik aplikasi (lihat URL Play Store)
    lang="id",              # ulasan berbahasa Indonesia
    country="id",           # dari Play Store region Indonesia
    sort=Sort.NEWEST,       # urut dari terbaru
    count=2000,             # maksimal per permintaan
    continuation_token=token_lanjutan,   # "penanda halaman" untuk permintaan berikutnya
)
```

## 3. Konsep Penting: Pagination dengan Continuation Token

Play Store tidak memberikan 35.000 ulasan sekaligus — data diambil **per halaman** (batch)
maksimal ±2.000 ulasan. Setiap panggilan mengembalikan `continuation_token`, semacam
"pembatas buku" yang menandai posisi terakhir. Loop di script:

```python
while len(semua_ulasan) < jumlah:
    hasil, token_lanjutan = reviews(..., continuation_token=token_lanjutan)
    if not hasil:          # tidak ada data lagi → berhenti
        break
    semua_ulasan.extend(hasil)
```

Panggilan pertama `token_lanjutan = None` (mulai dari awal); panggilan berikutnya meneruskan
token dari hasil sebelumnya. Ini pola standar untuk mengambil data besar dari API mana pun.

## 4. Dari 36.000 Menjadi 22.668: Kenapa Menyusut?

Setelah terkumpul ±36.000 ulasan, script melakukan dua pembersihan:

```python
df = df.dropna(subset=["ulasan"])          # buang ulasan kosong
df = df.drop_duplicates(subset=["ulasan"]) # buang teks yang persis sama
```

Hasilnya tinggal **22.668 baris**. Sepertiga data hilang karena banyak pengguna menulis ulasan
identik: "Bagus", "Mantap", "Sangat membantu". **Kenapa duplikat harus dibuang?** Dua alasan:

1. **Kebocoran data (data leakage):** kalau teks "bagus" muncul 500 kali lalu data dibagi
   80/20, teks yang sama ada di training DAN testing → model seolah "sudah melihat soal ujian"
   → akurasi testing menggelembung palsu.
2. **Bias frekuensi:** model jadi terlalu condong ke pola ulasan-ulasan pendek yang berulang.

Target scraping sengaja dilebihkan ke 35.000 justru untuk mengantisipasi penyusutan ini agar
hasil akhirnya tetap jauh di atas syarat 10.000 sampel.

## 5. Kolom yang Disimpan

| Kolom | Isi | Dipakai untuk |
|---|---|---|
| `nama_pengguna` | penulis ulasan | dokumentasi (tidak dipakai model) |
| `rating` | bintang 1–5 | **validasi pelabelan** di Modul 04 |
| `tanggal` | waktu ulasan | dokumentasi |
| `ulasan` | teks ulasan | **bahan utama** seluruh pipeline |

Catatan penting: `rating` sengaja **tidak** dipakai sebagai label, tetapi disimpan sebagai
alat uji silang — nanti kita cek apakah label dari kamus sentimen "nyambung" dengan bintang
yang diberikan pengguna. Ini dibahas tuntas di Modul 04.

## 6. Etika Scraping

- Data yang diambil adalah **ulasan publik** yang memang ditampilkan terbuka di Play Store
- Tidak ada login, tidak menembus proteksi, tidak mengambil data pribadi tersembunyi
- Volume permintaan wajar (belasan panggilan, bukan ribuan per detik)
- Dipakai untuk tujuan edukasi/riset

## Cek Pemahaman

1. Apa fungsi `continuation_token` dan kenapa scraping tidak bisa sekali panggil langsung 35.000?
2. Jelaskan bagaimana duplikat bisa membuat akurasi testing "bohong".
3. Kenapa kolom `rating` tetap disimpan padahal bukan label?

→ Lanjut ke [Modul 03 — Preprocessing Teks](03-preprocessing-teks.md)
