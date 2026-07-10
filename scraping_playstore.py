"""
Scraping ulasan aplikasi BRImo (BRI Mobile Banking) dari Google Play Store.

Script ini mengambil ulasan berbahasa Indonesia menggunakan library
google-play-scraper, lalu menyimpannya ke dalam file CSV yang akan
digunakan pada tahap pelabelan dan pelatihan model analisis sentimen.
"""

import pandas as pd
from google_play_scraper import Sort, reviews

# ID aplikasi BRImo di Google Play Store
APP_ID = "id.co.bri.brimo"
# Jumlah ulasan yang ingin dikumpulkan
TARGET_JUMLAH = 35000
# Nama file output
OUTPUT_CSV = "ulasan_brimo.csv"


def scrape_ulasan(app_id: str, jumlah: int) -> pd.DataFrame:
    """Mengambil ulasan aplikasi dari Play Store hingga mencapai jumlah target."""
    semua_ulasan = []
    token_lanjutan = None

    while len(semua_ulasan) < jumlah:
        hasil, token_lanjutan = reviews(
            app_id,
            lang="id",              # ulasan berbahasa Indonesia
            country="id",           # dari Play Store region Indonesia
            sort=Sort.NEWEST,       # diurutkan dari yang terbaru
            count=min(2000, jumlah - len(semua_ulasan)),
            continuation_token=token_lanjutan,
        )
        if not hasil:
            break
        semua_ulasan.extend(hasil)
        print(f"Terkumpul {len(semua_ulasan)} ulasan...")

    df = pd.DataFrame(semua_ulasan)
    # Ambil kolom yang relevan untuk analisis sentimen
    df = df[["userName", "score", "at", "content"]]
    df = df.rename(
        columns={
            "userName": "nama_pengguna",
            "score": "rating",
            "at": "tanggal",
            "content": "ulasan",
        }
    )
    return df


def main():
    print(f"Memulai scraping ulasan aplikasi {APP_ID}...")
    df = scrape_ulasan(APP_ID, TARGET_JUMLAH)

    # Buang ulasan kosong dan duplikat
    df = df.dropna(subset=["ulasan"])
    df = df.drop_duplicates(subset=["ulasan"])

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Selesai. {len(df)} ulasan tersimpan di {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
