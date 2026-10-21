"""
Langkah 1 - Data sintetis 1000 baris dari 2 data contoh (Bedah Kasus 3: Pengamatan Cuaca).

Skema:
  Kota           (Kategorik Nominal)
  Suhu (°C)      (Numerik Interval)  -> 0°C bukan "tidak ada suhu"; rasio tidak berlaku
  Kelembaban (%) (Numerik Rasio)     -> 0% = benar-benar tidak ada kelembaban
  Kondisi        (Kategorik Nominal)

Dua data contoh:
  Jakarta | 32.5 | 75 | Cerah
  Bandung | 24.0 | 85 | Hujan

Logika sintetis (agar realistis):
  1. Kondisi dipilih dulu dengan peluang berbeda per kota (Bandung/Medan lebih sering hujan).
  2. Suhu = suhu dasar kota + efek kondisi (cerah lebih panas, hujan/badai lebih sejuk) + noise.
  3. Kelembaban = dasar kota + efek kondisi - 1,6 x (selisih suhu dari dasar) + noise  (korelasi negatif suhu-kelembaban).
"""
import numpy as np
import pandas as pd

SEED, N = 42, 1000
rng = np.random.default_rng(SEED)

KONDISI = ["Cerah", "Berawan", "Hujan", "Badai Petir"]
# kota: (suhu dasar, kelembaban dasar, bobot kemunculan, peluang [Cerah, Berawan, Hujan, Badai Petir])
KOTA = {
    "Jakarta":    (31.0, 72, 0.18, [0.38, 0.30, 0.25, 0.07]),
    "Bandung":    (25.0, 80, 0.15, [0.22, 0.30, 0.42, 0.06]),
    "Surabaya":   (32.0, 70, 0.14, [0.45, 0.30, 0.20, 0.05]),
    "Medan":      (30.5, 78, 0.12, [0.28, 0.27, 0.35, 0.10]),
    "Semarang":   (31.5, 72, 0.12, [0.40, 0.30, 0.24, 0.06]),
    "Yogyakarta": (29.0, 74, 0.11, [0.35, 0.33, 0.28, 0.04]),
    "Makassar":   (31.5, 70, 0.10, [0.46, 0.28, 0.20, 0.06]),
    "Denpasar":   (30.0, 74, 0.08, [0.42, 0.30, 0.24, 0.04]),
}
EFEK_SUHU = {"Cerah": +1.5, "Berawan": 0.0, "Hujan": -2.5, "Badai Petir": -3.2}
EFEK_RH   = {"Cerah": -9,   "Berawan": 0,   "Hujan": +11,  "Badai Petir": +14}

nama = list(KOTA)
bobot = np.array([KOTA[k][2] for k in nama]); bobot /= bobot.sum()
n = N - 2
kota = rng.choice(nama, size=n, p=bobot)

kondisi = np.array([rng.choice(KONDISI, p=KOTA[k][3]) for k in kota])
dasar_t = np.array([KOTA[k][0] for k in kota]); dasar_rh = np.array([KOTA[k][1] for k in kota])
dt = np.array([EFEK_SUHU[c] for c in kondisi]) + rng.normal(0, 1.4, n)
suhu = np.round(dasar_t + dt, 1)
rh = dasar_rh + np.array([EFEK_RH[c] for c in kondisi]) - 1.6 * (dt - np.array([EFEK_SUHU[c] for c in kondisi])) + rng.normal(0, 4.5, n)
rh = np.clip(np.round(rh), 30, 100).astype(int)

df = pd.DataFrame({"Kota": kota, "Suhu (°C)": suhu, "Kelembaban (%)": rh, "Kondisi": kondisi})
asli = pd.DataFrame({"Kota": ["Jakarta", "Bandung"], "Suhu (°C)": [32.5, 24.0],
                     "Kelembaban (%)": [75, 85], "Kondisi": ["Cerah", "Hujan"]})
df = pd.concat([asli, df], ignore_index=True)
df.insert(0, "ID", [f"OBS{i:04d}" for i in range(1, len(df) + 1)])
df.to_csv("data_cuaca.csv", index=False)

print(f"Data tersimpan: data_cuaca.csv ({len(df)} baris)")
print(df.head(8).to_string(index=False))
