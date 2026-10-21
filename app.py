"""
Langkah 3 - Dashboard Streamlit: Pengamatan Cuaca (Bedah Kasus 3)
Jalankan:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

st.set_page_config(page_title="Dashboard Pengamatan Cuaca", page_icon="🌦️", layout="wide")

T, RH = "Suhu (°C)", "Kelembaban (%)"
KONDISI = ["Cerah", "Berawan", "Hujan", "Badai Petir"]
WARNA = {"Cerah": "#F4B400", "Berawan": "#9AA0A6", "Hujan": "#4285F4", "Badai Petir": "#7B1FA2"}
URUT_DI = ["<21 (nyaman)", "21-24", "24-27", "27-29", "29-32", ">=32 (darurat)"]

@st.cache_data
def load(path="data_cuaca.csv"):
    if not os.path.exists(path):
        import subprocess, sys
        subprocess.run([sys.executable, "1_generate_data.py"], check=True)
    d = pd.read_csv(path)
    d["DI"] = d[T] - 0.55 * (1 - d[RH] / 100) * (d[T] - 14.5)   # Discomfort Index (Thom)
    d["Kenyamanan"] = pd.cut(d["DI"], [-99, 21, 24, 27, 29, 32, 99], labels=URUT_DI, right=False)
    return d

df = load()

# ------------------------------------------------------------- SIDEBAR
st.sidebar.header("🔎 Filter Data")
kota = st.sidebar.multiselect("Kota", sorted(df["Kota"].unique()), default=sorted(df["Kota"].unique()))
kond = st.sidebar.multiselect("Kondisi", KONDISI, default=KONDISI)
t_r = st.sidebar.slider("Rentang suhu (°C)", float(df[T].min()), float(df[T].max()), (float(df[T].min()), float(df[T].max())), step=0.1)
h_r = st.sidebar.slider("Rentang kelembaban (%)", int(df[RH].min()), int(df[RH].max()), (int(df[RH].min()), int(df[RH].max())))

f = df[df["Kota"].isin(kota) & df["Kondisi"].isin(kond) & df[T].between(*t_r) & df[RH].between(*h_r)].copy()

st.caption("Nama: Syamsurizal NIM: 241051004")
st.title("🌦️ Dashboard Pengamatan Cuaca")
st.caption("Bedah Kasus 3 · Data sintetis 1.000 pengamatan (dikembangkan dari 2 data contoh: Jakarta & Bandung)")
if f.empty:
    st.warning("Tidak ada data untuk filter yang dipilih."); st.stop()

# ------------------------------------------------------------- KPI
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jumlah Pengamatan", f"{len(f):,}".replace(",", "."))
c2.metric("Rata-rata Suhu", f"{f[T].mean():.1f} °C")
c3.metric("Rata-rata Kelembaban", f"{f[RH].mean():.1f} %")
c4.metric("Hari Hujan/Badai", f"{f['Kondisi'].isin(['Hujan', 'Badai Petir']).mean():.1%}")
c5.metric("Kondisi Dominan", f["Kondisi"].mode().iloc[0])

t1, t2, t3, t4, t5, t6 = st.tabs(["🏷️ Kota & Kondisi (Nominal)", "🌡️ Suhu (Interval)", "💧 Kelembaban (Rasio)",
                                  "🔗 Hubungan", "🧪 Analisis Statistik", "📚 Skala Data"])

# ------------------------------------------------------------- NOMINAL
with t1:
    a, b = st.columns(2)
    fk = f["Kota"].value_counts().rename_axis("Kota").reset_index(name="Jumlah").sort_values("Jumlah")
    a.plotly_chart(px.bar(fk, x="Jumlah", y="Kota", orientation="h", text="Jumlah", color="Kota",
                          title="Pengamatan per Kota (Bar Chart)").update_layout(showlegend=False), use_container_width=True)
    fc = f["Kondisi"].value_counts().rename_axis("Kondisi").reset_index(name="Jumlah")
    b.plotly_chart(px.pie(fc, names="Kondisi", values="Jumlah", hole=.45, color="Kondisi", color_discrete_map=WARNA,
                          title="Proporsi Kondisi Cuaca (Pie/Donut)"), use_container_width=True)
    a, b = st.columns(2)
    ct = pd.crosstab(f["Kota"], f["Kondisi"], normalize="index").mul(100).reindex(columns=[k for k in KONDISI if k in f["Kondisi"].unique()])
    a.plotly_chart(px.bar(ct.reset_index().melt("Kota"), x="Kota", y="value", color="Kondisi", color_discrete_map=WARNA,
                          labels={"value": "% pengamatan"}, title="Komposisi Kondisi per Kota (100% stacked)"), use_container_width=True)
    b.plotly_chart(px.imshow(ct, text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
                             title="Heatmap Kondisi per Kota (% baris)"), use_container_width=True)
    pk = f.groupby("Kota").agg(n=("ID", "count"), Suhu_Mean=(T, "mean"), Kelembaban_Mean=(RH, "mean"),
                               Hujan_Badai_pct=("Kondisi", lambda x: x.isin(["Hujan", "Badai Petir"]).mean() * 100)).round(2).reset_index()
    st.dataframe(pk.sort_values("Suhu_Mean", ascending=False), use_container_width=True, hide_index=True)

# ------------------------------------------------------------- SUHU
with t2:
    a, b = st.columns(2)
    h = px.histogram(f, x=T, nbins=30, marginal="rug", title="Histogram Suhu")
    h.add_vline(x=f[T].mean(), line_dash="dash", line_color="red", annotation_text="Mean")
    h.add_vline(x=f[T].median(), line_dash="dash", line_color="green", annotation_text="Median", annotation_position="bottom right")
    a.plotly_chart(h, use_container_width=True)
    by = b.radio("Box plot menurut", ["Kota", "Kondisi"], horizontal=True, key="by_t")
    b.plotly_chart(px.box(f, x=by, y=T, color=by, points="outliers", color_discrete_map=WARNA if by == "Kondisi" else None,
                          title=f"Suhu per {by}").update_layout(showlegend=False), use_container_width=True)
    st.plotly_chart(px.violin(f, x="Kota", y=T, color="Kondisi", box=True, color_discrete_map=WARNA,
                              title="Suhu per Kota & Kondisi (Violin)"), use_container_width=True)
    d = f[T].describe().to_frame().T; d["skewness"] = f[T].skew(); d["IQR"] = d["75%"] - d["25%"]
    st.dataframe(d.round(2), use_container_width=True)
    st.caption("Skala interval: selisih antar suhu bermakna, rata-rata & simpangan baku sah; rasio ('2× lebih panas') dan koefisien variasi tidak sah.")

# ------------------------------------------------------------- KELEMBABAN
with t3:
    a, b = st.columns(2)
    h = px.histogram(f, x=RH, nbins=30, marginal="rug", color_discrete_sequence=["#EA8600"], title="Histogram Kelembaban")
    h.add_vline(x=f[RH].mean(), line_dash="dash", line_color="red", annotation_text="Mean")
    h.add_vline(x=f[RH].median(), line_dash="dash", line_color="green", annotation_text="Median", annotation_position="bottom right")
    a.plotly_chart(h, use_container_width=True)
    by = b.radio("Box plot menurut", ["Kota", "Kondisi"], horizontal=True, key="by_h")
    b.plotly_chart(px.box(f, x=by, y=RH, color=by, points="outliers", color_discrete_map=WARNA if by == "Kondisi" else None,
                          title=f"Kelembaban per {by}").update_layout(showlegend=False), use_container_width=True)
    d = f[RH].describe().to_frame().T; d["skewness"] = f[RH].skew(); d["CV (%)"] = f[RH].std() / f[RH].mean() * 100
    st.dataframe(d.round(2), use_container_width=True)
    st.caption("Skala rasio: nol absolut → semua operasi sah, termasuk rasio (80% adalah 2× dari 40%) dan koefisien variasi.")

# ------------------------------------------------------------- HUBUNGAN
with t4:
    a, b = st.columns([3, 2])
    sc = px.scatter(f, x=T, y=RH, color="Kondisi", color_discrete_map=WARNA, opacity=.7, hover_data=["Kota"],
                    title="Suhu vs Kelembaban")
    if len(f) > 5:
        reg = stats.linregress(f[T], f[RH]); xs = np.linspace(f[T].min(), f[T].max(), 50)
        sc.add_trace(go.Scatter(x=xs, y=reg.intercept + reg.slope * xs, mode="lines", line=dict(color="black", dash="dash"),
                                name=f"Regresi (r={reg.rvalue:.2f})"))
    a.plotly_chart(sc, use_container_width=True)
    corr = f[[T, RH, "DI"]].corr()
    b.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Korelasi Pearson"),
                   use_container_width=True)
    kn = f["Kenyamanan"].value_counts().reindex(URUT_DI, fill_value=0).rename_axis("DI").reset_index(name="Jumlah")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(kn, x="DI", y="Jumlah", text="Jumlah", color_discrete_sequence=["#D93025"],
                          title="Tingkat Kenyamanan (Discomfort Index = T − 0,55·(1−RH)·(T−14,5))")
                   .update_xaxes(categoryorder="array", categoryarray=URUT_DI), use_container_width=True)
    b.plotly_chart(px.scatter(f, x=T, y="DI", color="Kota", opacity=.6, title="Suhu vs Discomfort Index"), use_container_width=True)

# ------------------------------------------------------------- STATISTIK
with t5:
    if len(f) < 20:
        st.info("Data terlalu sedikit untuk uji statistik.")
    else:
        a, b = st.columns(2)
        with a:
            st.markdown("### Korelasi Suhu vs Kelembaban")
            r, p = stats.pearsonr(f[T], f[RH]); rho, ps = stats.spearmanr(f[T], f[RH])
            st.write(f"Pearson r = `{r:.3f}` (p = `{p:.3g}`) · Spearman ρ = `{rho:.3f}` (p = `{ps:.3g}`) · R² = `{r**2:.3f}`")
            reg = stats.linregress(f[T], f[RH])
            st.write(f"Kelembaban = `{reg.intercept:.2f}` + (`{reg.slope:.3f}`) × Suhu → tiap +1 °C, kelembaban berubah `{reg.slope:.2f}` poin persen.")
            st.markdown("### Normalitas (Shapiro-Wilk)")
            smp = f.sample(min(500, len(f)), random_state=1)
            for v in [T, RH]:
                w, pw = stats.shapiro(smp[v])
                st.write(f"**{v}**: W = `{w:.3f}`, p = `{pw:.3g}` → {'normal' if pw > .05 else 'tidak normal'} (skew = `{f[v].skew():.2f}`)")
        with b:
            st.markdown("### Perbedaan antar kelompok")
            for v in [T, RH]:
                for kel in ["Kota", "Kondisi"]:
                    grp = [g[v].values for _, g in f.groupby(kel) if len(g) > 1]
                    if len(grp) >= 2:
                        F, pF = stats.f_oneway(*grp); H, pH = stats.kruskal(*grp)
                        eta = F * (len(grp) - 1) / (F * (len(grp) - 1) + len(f) - len(grp))
                        st.write(f"**{v} antar {kel}**: ANOVA F = `{F:.1f}` (p = `{pF:.3g}`), η² = `{eta:.2f}` · Kruskal H = `{H:.1f}` (p = `{pH:.3g}`)")
            ctab = pd.crosstab(f["Kota"], f["Kondisi"])
            if ctab.shape[0] > 1 and ctab.shape[1] > 1:
                chi, pc, dof, _ = stats.chi2_contingency(ctab)
                V = np.sqrt(chi / (len(f) * (min(ctab.shape) - 1)))
                st.write(f"**Kota × Kondisi**: χ² = `{chi:.1f}`, dof = `{dof}`, p = `{pc:.3g}`, Cramér's V = `{V:.2f}`")
        st.markdown("### Korelasi Suhu–Kelembaban di dalam tiap kota")
        ck = pd.DataFrame([(k, stats.pearsonr(g[T], g[RH])[0], len(g)) for k, g in f.groupby("Kota") if len(g) > 10], columns=["Kota", "Pearson r", "n"])
        if not ck.empty:
            st.plotly_chart(px.bar(ck.sort_values("Pearson r"), x="Pearson r", y="Kota", orientation="h", color="Pearson r",
                                   color_continuous_scale="RdBu_r", range_color=[-1, 1]), use_container_width=True)

# ------------------------------------------------------------- SKALA DATA
with t6:
    st.markdown("### Interval vs Rasio: mengapa rasio suhu °C tidak bermakna")
    st.table(pd.DataFrame({
        "Variabel": ["Kota", "Suhu (°C)", "Kelembaban (%)", "Kondisi"],
        "Skala": ["Nominal", "Interval", "Rasio", "Nominal"],
        "Nol": ["-", "Arbitrer (0 °C ≠ tidak ada suhu)", "Absolut (0% = tidak ada uap air)", "-"],
        "Statistik yang sah": ["Modus, frekuensi", "Mean, SD, selisih", "Mean, SD, selisih, rasio, CV", "Modus, frekuensi"],
        "Visualisasi": ["Bar / Pie", "Histogram / Box Plot", "Histogram / Box Plot", "Bar / Pie"]}))
    a, b = st.columns(2)
    x = a.number_input("Suhu A (°C)", value=20.0, step=1.0)
    y = b.number_input("Suhu B (°C)", value=10.0, step=1.0)
    if y != 0 and x + 273.15 > 0 and y + 273.15 > 0:
        r1 = pd.DataFrame({"Satuan": ["Celsius", "Kelvin", "Fahrenheit"],
                           "Suhu A": [x, x + 273.15, x * 9 / 5 + 32], "Suhu B": [y, y + 273.15, y * 9 / 5 + 32]})
        r1["Rasio A/B"] = r1["Suhu A"] / r1["Suhu B"].replace(0, np.nan)
        st.dataframe(r1.round(3), use_container_width=True, hide_index=True)
        st.warning("Rasio A/B berbeda di tiap satuan → '2× lebih panas' tidak punya arti fisik pada skala Celsius. Selisih (A − B) tetap bermakna.")
    st.success("Sebaliknya, kelembaban 80% memang dua kali kelembaban 40% pada skala rasio, apa pun cara mengukurnya.")

with st.expander("📄 Lihat & unduh data (hasil filter)"):
    st.dataframe(f.drop(columns=["Kenyamanan"]), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Unduh CSV", f.drop(columns=["Kenyamanan"]).to_csv(index=False).encode("utf-8"), "cuaca_filter.csv", "text/csv")
