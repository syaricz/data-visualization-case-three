"""
Langkah 2 - Analisis detail + visualisasi (Bedah Kasus 3: Pengamatan Cuaca).
  Kota, Kondisi (Nominal)  -> Bar Chart / Pie
  Suhu (Interval)          -> Histogram, Box Plot  (rata-rata & selisih bermakna, RASIO tidak)
  Kelembaban (Rasio)       -> Histogram, Box Plot  (rata-rata, selisih & rasio bermakna)
Hasil gambar disimpan di ./hasil_analisis
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", palette="deep")
OUT = "hasil_analisis"; os.makedirs(OUT, exist_ok=True)
def judul(t): print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)

df = pd.read_csv("data_cuaca.csv")
T, RH = "Suhu (°C)", "Kelembaban (%)"

# fitur turunan: Discomfort Index (Thom) = T - 0.55*(1 - RH/100)*(T - 14.5)
df["DI"] = df[T] - 0.55 * (1 - df[RH] / 100) * (df[T] - 14.5)
URUT_DI = ["<21 (nyaman)", "21-24", "24-27", "27-29", "29-32", ">=32 (darurat)"]
df["Kenyamanan"] = pd.cut(df["DI"], [-99, 21, 24, 27, 29, 32, 99], labels=URUT_DI, right=False)

# ------------------------------------------------------------ 1. KUALITAS
judul("1. KUALITAS DATA")
print(f"Baris/kolom: {df.shape}")
print("\nMissing value:\n", df[["Kota", T, RH, "Kondisi"]].isna().sum().to_string())
print(f"\nDuplikat ID: {df['ID'].duplicated().sum()}")
print(f"Kelembaban di luar 0-100: {((df[RH] < 0) | (df[RH] > 100)).sum()}")
print(f"Suhu di luar 15-40 °C (tak wajar untuk kota di Indonesia): {((df[T] < 15) | (df[T] > 40)).sum()}")
print(f"Kelembaban = 100% (jenuh/tersensor): {(df[RH] == 100).sum()}")

# ------------------------------------------------------------ 2. SKALA
judul("2. KLASIFIKASI SKALA DATA")
print(pd.DataFrame({
    "Variabel": ["Kota", "Suhu (°C)", "Kelembaban (%)", "Kondisi"],
    "Jenis": ["Kategorik", "Numerik", "Numerik", "Kategorik"],
    "Skala": ["Nominal", "Interval", "Rasio", "Nominal"],
    "Nol": ["-", "Arbitrer (0°C ≠ tak ada suhu)", "Absolut (0% = tak ada uap air)", "-"],
    "Operasi sah": ["=, ≠, modus", "+, -, mean, SD (bukan rasio)", "+, -, ×, ÷, mean, rasio", "=, ≠, modus"]}).to_string(index=False))

judul("3. DEMONSTRASI: MENGAPA RASIO SUHU CELSIUS TIDAK BERMAKNA")
for a, b in [(20, 10), (30, 15)]:
    print(f"{a}°C vs {b}°C: rasio Celsius = {a/b:.2f}x | dalam Kelvin ({a+273.15:.2f} K vs {b+273.15:.2f} K) rasio = {(a+273.15)/(b+273.15):.3f}x")
print("-> Rasio berubah bila satuan diganti, maka '2x lebih panas' tidak valid. Selisih (10°C) tetap sah.")
print("Kelembaban: 80% adalah 2x dari 40% (kandungan uap relatif), valid karena nol absolut.")

# ------------------------------------------------------------ 4. DESKRIPTIF
judul("4. STATISTIK DESKRIPTIF NUMERIK")
num = df[[T, RH, "DI"]]
d = num.describe().T
d["skewness"], d["kurtosis"], d["CV (%)"] = num.skew(), num.kurt(), num.std() / num.mean() * 100
d["IQR"] = d["75%"] - d["25%"]
print(d.round(3).to_string())
print("\nCatatan: CV (%) hanya bermakna untuk skala rasio (Kelembaban). CV Suhu °C tidak valid karena nol arbitrer.")

judul("5. STATISTIK DESKRIPTIF KATEGORIK")
for k in ["Kota", "Kondisi"]:
    fr = df[k].value_counts()
    print(f"\n{k}: modus = {fr.index[0]}\n", pd.DataFrame({"Frekuensi": fr, "Persen (%)": (fr / len(df) * 100).round(1)}).to_string())
print("\nTabel silang Kota x Kondisi (% baris):\n", (pd.crosstab(df["Kota"], df["Kondisi"], normalize="index") * 100).round(1).to_string())

# ------------------------------------------------------------ 6. PER KELOMPOK
judul("6. SUHU & KELEMBABAN PER KOTA")
pk = df.groupby("Kota").agg(n=("ID", "count"), Suhu_mean=(T, "mean"), Suhu_sd=(T, "std"), Suhu_max=(T, "max"),
                            RH_mean=(RH, "mean"), RH_sd=(RH, "std"), Hujan_pct=("Kondisi", lambda x: x.isin(["Hujan", "Badai Petir"]).mean() * 100),
                            DI_mean=("DI", "mean")).round(2)
print(pk.sort_values("Suhu_mean", ascending=False).to_string())

judul("7. SUHU & KELEMBABAN PER KONDISI CUACA")
pc = df.groupby("Kondisi").agg(n=("ID", "count"), Suhu_mean=(T, "mean"), Suhu_median=(T, "median"),
                               RH_mean=(RH, "mean"), RH_median=(RH, "median")).round(2)
print(pc.to_string())

# ------------------------------------------------------------ 8. UJI
judul("8. UJI NORMALITAS")
for c in [T, RH]:
    w, p = stats.shapiro(df[c].sample(500, random_state=1))
    print(f"{c:<15}: Shapiro-Wilk W = {w:.3f}, p = {p:.3g} -> {'normal' if p > .05 else 'tidak normal'}; skew = {df[c].skew():.2f}")

judul("9. KORELASI SUHU vs KELEMBABAN")
r, pr = stats.pearsonr(df[T], df[RH]); rho, ps = stats.spearmanr(df[T], df[RH])
print(f"Pearson r = {r:.3f} (p={pr:.3g}); Spearman rho = {rho:.3f} (p={ps:.3g}); R² = {r**2:.3f}")
reg = stats.linregress(df[T], df[RH])
print(f"Regresi: Kelembaban = {reg.intercept:.2f} + ({reg.slope:.3f}) x Suhu  -> tiap +1°C, kelembaban turun ~{abs(reg.slope):.2f} poin persen")
print("\nKorelasi di dalam tiap kota (mengontrol perbedaan iklim antar kota):")
print(pd.DataFrame({k: [stats.pearsonr(g[T], g[RH])[0]] for k, g in df.groupby("Kota")}, index=["Pearson r"]).T.round(3).to_string())
print("\nKorelasi di dalam tiap kondisi:")
print(pd.DataFrame({k: [stats.pearsonr(g[T], g[RH])[0]] for k, g in df.groupby("Kondisi")}, index=["Pearson r"]).T.round(3).to_string())

judul("10. UJI PERBEDAAN ANTAR KELOMPOK")
for var in [T, RH]:
    for kel in ["Kota", "Kondisi"]:
        grp = [g[var].values for _, g in df.groupby(kel)]
        F, pF = stats.f_oneway(*grp); H, pH = stats.kruskal(*grp)
        k = len(grp); eta = (F * (k - 1)) / (F * (k - 1) + len(df) - k)
        print(f"{var:<15} antar {kel:<8}: ANOVA F = {F:7.2f} (p={pF:.3g}), eta² = {eta:.3f} | Kruskal H = {H:7.2f} (p={pH:.3g})")

print("\nUji lanjut Tukey HSD - Suhu antar Kondisi:")
th = stats.tukey_hsd(*[g[T].values for _, g in df.groupby("Kondisi")])
nm = sorted(df["Kondisi"].unique())
rows = [(nm[i], nm[j], df[df.Kondisi == nm[i]][T].mean() - df[df.Kondisi == nm[j]][T].mean(), th.pvalue[i, j])
        for i in range(len(nm)) for j in range(i + 1, len(nm))]
print(pd.DataFrame(rows, columns=["A", "B", "Selisih mean A-B (°C)", "p"]).round(4).to_string(index=False))

hj = df["Kondisi"].isin(["Hujan", "Badai Petir"])
u, pu = stats.mannwhitneyu(df.loc[hj, RH], df.loc[~hj, RH])
print(f"\nKelembaban saat hujan/badai vs tidak: median {df.loc[hj, RH].median():.0f}% vs {df.loc[~hj, RH].median():.0f}% (Mann-Whitney p = {pu:.3g})")
ct = pd.crosstab(df["Kota"], df["Kondisi"]); chi, pc2, dof, _ = stats.chi2_contingency(ct)
v = np.sqrt(chi / (len(df) * (min(ct.shape) - 1)))
print(f"\nKota x Kondisi: Chi-square = {chi:.2f}, dof = {dof}, p = {pc2:.3g}, Cramér's V = {v:.3f} -> "
      f"{'ada asosiasi' if pc2 < .05 else 'tidak ada asosiasi signifikan'} (kekuatan {'lemah' if v < .2 else 'sedang' if v < .4 else 'kuat'})")

judul("11. OUTLIER (IQR) per variabel")
for c in [T, RH]:
    q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1; lo, hi = q1 - 1.5 * i, q3 + 1.5 * i
    print(f"{c:<15}: batas [{lo:.1f}, {hi:.1f}] -> rendah={int((df[c] < lo).sum())}, tinggi={int((df[c] > hi).sum())}")
print("\nOutlier suhu dalam konteks kota (IQR per kota):")
for k, g in df.groupby("Kota"):
    q1, q3 = g[T].quantile([.25, .75]); i = q3 - q1
    print(f"  {k:<11}: {int(((g[T] < q1 - 1.5*i) | (g[T] > q3 + 1.5*i)).sum())} outlier")

judul("12. TINGKAT KENYAMANAN (Discomfort Index)")
kn = df["Kenyamanan"].value_counts().reindex(URUT_DI)
print(pd.DataFrame({"Jumlah": kn, "Persen (%)": (kn / len(df) * 100).round(1)}).to_string())
print("\n% observasi dengan DI >= 29 (mayoritas orang sangat tidak nyaman) per kota:\n", (df.assign(x=df["DI"] >= 29).groupby("Kota")["x"].mean() * 100).round(1).sort_values(ascending=False).to_string())

# ------------------------------------------------------------ VISUALISASI
fk, fc = df["Kota"].value_counts(), df["Kondisi"].value_counts()
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(x=fk.values, y=fk.index, ax=ax[0], hue=fk.index, legend=False); ax[0].set_title("Observasi per Kota (Bar)")
ax[1].pie(fc.values, labels=fc.index, autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white"}); ax[1].set_title("Proporsi Kondisi Cuaca (Pie)")
plt.tight_layout(); plt.savefig(f"{OUT}/01_kota_kondisi.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df[T], bins=30, kde=True, ax=ax[0]); ax[0].axvline(df[T].mean(), color="red", ls="--"); ax[0].set_title("Histogram Suhu (Interval)")
sns.histplot(df[RH], bins=30, kde=True, ax=ax[1], color="tab:orange"); ax[1].axvline(df[RH].mean(), color="red", ls="--"); ax[1].set_title("Histogram Kelembaban (Rasio)")
plt.tight_layout(); plt.savefig(f"{OUT}/02_histogram.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
o = pk.sort_values("Suhu_mean").index
sns.boxplot(data=df, x=T, y="Kota", order=o, ax=ax[0]); ax[0].set_title("Suhu per Kota")
sns.boxplot(data=df, x=RH, y="Kota", order=o, ax=ax[1], color="tab:orange"); ax[1].set_title("Kelembaban per Kota")
plt.tight_layout(); plt.savefig(f"{OUT}/03_box_kota.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ok = ["Cerah", "Berawan", "Hujan", "Badai Petir"]
sns.violinplot(data=df, x="Kondisi", y=T, order=ok, ax=ax[0], inner="quartile", hue="Kondisi", legend=False); ax[0].set_title("Suhu per Kondisi")
sns.violinplot(data=df, x="Kondisi", y=RH, order=ok, ax=ax[1], inner="quartile", hue="Kondisi", legend=False); ax[1].set_title("Kelembaban per Kondisi")
plt.tight_layout(); plt.savefig(f"{OUT}/04_violin_kondisi.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=df, x=T, y=RH, hue="Kondisi", hue_order=ok, alpha=.65, ax=ax)
xs = np.linspace(df[T].min(), df[T].max(), 50); ax.plot(xs, reg.intercept + reg.slope * xs, "k--", label=f"Regresi (r={r:.2f})")
ax.legend(); ax.set_title("Suhu vs Kelembaban"); plt.tight_layout(); plt.savefig(f"{OUT}/05_scatter.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(pd.crosstab(df["Kota"], df["Kondisi"], normalize="index")[ok] * 100, annot=True, fmt=".0f", cmap="Blues", ax=ax)
ax.set_title("Kondisi Cuaca per Kota (% baris)"); plt.tight_layout(); plt.savefig(f"{OUT}/06_heatmap_kota_kondisi.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df[[T, RH, "DI"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax); ax.set_title("Korelasi Pearson")
plt.tight_layout(); plt.savefig(f"{OUT}/07_korelasi.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=kn.index, y=kn.values, ax=ax, color="tab:red"); ax.tick_params(axis="x", rotation=20); ax.set_title("Tingkat Kenyamanan (Discomfort Index)")
plt.tight_layout(); plt.savefig(f"{OUT}/08_kenyamanan.png", dpi=130); plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
sns.kdeplot(data=df, x=T, hue="Kota", common_norm=False, ax=ax); ax.set_title("Kepadatan Suhu per Kota")
plt.tight_layout(); plt.savefig(f"{OUT}/09_kde_suhu_kota.png", dpi=130); plt.close()

# ------------------------------------------------------------ INSIGHT
judul("13. RINGKASAN INSIGHT")
print(f"1. Kota terpanas: {pk['Suhu_mean'].idxmax()} ({pk['Suhu_mean'].max():.1f}°C); terdingin: {pk['Suhu_mean'].idxmin()} ({pk['Suhu_mean'].min():.1f}°C); paling lembab: {pk['RH_mean'].idxmax()} ({pk['RH_mean'].max():.0f}%).")
print(f"2. Kondisi dominan: {fc.index[0]} ({fc.iloc[0]/len(df):.1%}); hujan+badai = {hj.mean():.1%} observasi.")
print(f"3. Suhu & kelembaban berkorelasi negatif kuat (r = {r:.2f}); +1°C ≈ {reg.slope:.2f} poin persen kelembaban.")
print(f"4. Hari hujan/badai rata-rata {df.loc[~hj, T].mean() - df.loc[hj, T].mean():.1f}°C lebih sejuk dan {df.loc[hj, RH].mean() - df.loc[~hj, RH].mean():.0f} poin lebih lembab.")
print(f"5. Asosiasi Kota-Kondisi {'signifikan' if pc2 < .05 else 'tidak signifikan'} (Cramér's V = {v:.2f}).")
print(f"\nGrafik tersimpan di '{OUT}/'")
