"""
STUDI KASUS 3: DATASET PENGAMATAN CUACA
Dashboard Streamlit - Klasifikasi Tipe Data & Pemilihan Visualisasi

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# IDENTITAS MAHASISWA  --> WAJIB DIISI SEBELUM DIKUMPULKAN
# =============================================================================
NIM = "ISI_NIM_KAMU"
NAMA = "ISI_NAMA_LENGKAP_KAMU"
# =============================================================================

# Warna mengikuti skema pada materi: biru = kategorik, oranye = numerik
BIRU = "#2B8FD0"
BIRU_PIE = ["#0d366b", "#184f95", "#2a78d6", "#5598e7", "#86b6ef", "#b7d3f6"]
ABU = "#9CA3AF"
ORANYE = "#E8890C"

K_KOTA = "Kota"
K_SUHU = "Suhu (°C)"
K_LEMBAB = "Kelembaban (%)"
K_KOND = "Kondisi"

# Warna kondisi cuaca tetap per kategori (urutan slot palet yang sudah divalidasi), plus simbol
# berbeda sebagai penanda kedua agar tidak bergantung pada warna saja.
URUT_KOND = ["Hujan", "Cerah", "Berawan", "Badai Petir"]
WARNA_KOND = {"Hujan": "#2a78d6", "Cerah": "#eb6834", "Berawan": "#1baf7a", "Badai Petir": "#eda100"}
SIMBOL_KOND = {"Hujan": "circle", "Cerah": "diamond", "Berawan": "square", "Badai Petir": "triangle-up"}

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cuaca.csv")

st.set_page_config(
    page_title=f"Case 3 | {NIM} | {NAMA}",
    page_icon="🌦️",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Fungsi bantu
# -----------------------------------------------------------------------------
def angka(v, d=0):
    """Format angka gaya Indonesia: titik ribuan, koma desimal (32,5)."""
    s = f"{v:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def urut_kondisi(seri):
    """Urutan kondisi yang tetap: kondisi di luar daftar ditaruh di belakang."""
    ada = set(seri.unique())
    return [k for k in URUT_KOND if k in ada] + sorted(ada - set(URUT_KOND))


def warna_k(k):
    return WARNA_KOND.get(k, "#6B7280")


def simbol_k(k):
    return SIMBOL_KOND.get(k, "circle")


@st.cache_data
def muat_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def gaya(fig, tinggi=380):
    """Tampilan grafik yang bersih: grid tipis, tanpa legenda ganda."""
    fig.update_layout(
        height=tinggi,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


def tampil(fig, key):
    if fig is None:
        st.info("Data terlalu sedikit untuk membuat grafik ini. Longgarkan filter di sidebar.")
    else:
        st.plotly_chart(fig, key=key)


def hist_kontinu(seri, label, d_bin, nbins):
    """Histogram untuk data numerik kontinu."""
    x = seri.to_numpy(dtype=float)
    if x.size < 2 or x.min() == x.max():
        return None
    counts, batas = np.histogram(x, bins=nbins)
    pos = (batas[:-1] + batas[1:]) / 2
    lebar = np.diff(batas) * 0.94
    teks = [f"{angka(a, d_bin)} – {angka(b, d_bin)}" for a, b in zip(batas[:-1], batas[1:])]
    fig = go.Figure(
        go.Bar(
            x=pos,
            y=counts,
            width=lebar,
            marker_color=ORANYE,
            customdata=teks,
            hovertemplate="%{customdata}<br>Frekuensi: %{y:,} pengamatan<extra></extra>",
        )
    )
    fig.update_xaxes(title=label)
    fig.update_yaxes(title="Frekuensi (jumlah pengamatan)")
    fig.update_layout(title=f"Histogram {label}")
    return gaya(fig, 340)


def box_kontinu(seri, label):
    """Box Plot horizontal; titik di luar whisker = outlier."""
    if seri.size < 2:
        return None
    fig = go.Figure(
        go.Box(
            x=seri.to_numpy(dtype=float),
            orientation="h",
            name="",
            marker=dict(color=ORANYE),
            boxpoints="outliers",
            hovertemplate=f"{label}: %{{x}}<extra></extra>",
        )
    )
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title=label)
    fig.update_layout(title=f"Box Plot {label}")
    return gaya(fig, 230)


def ringkas(seri, d):
    """Ringkasan lima angka + jumlah outlier menurut aturan 1,5 x IQR."""
    q1, med, q3 = seri.quantile(0.25), seri.median(), seri.quantile(0.75)
    iqr = q3 - q1
    bawah, atas = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((seri < bawah) | (seri > atas)).sum())
    return (
        f"Q1 = {angka(q1, d)} | Median = {angka(med, d)} | Q3 = {angka(q3, d)} | "
        f"Rata-rata = {angka(seri.mean(), d)}  \n"
        f"Batas outlier (1,5 × IQR): di bawah {angka(bawah, d)} atau di atas {angka(atas, d)} "
        f"→ **{n_out} outlier**"
    )


# -----------------------------------------------------------------------------
# Header + identitas (WAJIB tampil di dashboard)
# -----------------------------------------------------------------------------
st.title("🌦️ Bedah Kasus 3: Dataset Pengamatan Cuaca")
st.markdown(f"**NIM:** {NIM} &nbsp;&nbsp;|&nbsp;&nbsp; **Nama Mahasiswa:** {NAMA}")

if NIM.startswith("ISI_") or NAMA.startswith("ISI_"):
    st.warning("NIM dan NAMA belum diisi. Buka app.py lalu ganti nilai NIM dan NAMA di bagian atas file.")

# Angka KPI jangan terpotong titik-titik di layar yang sempit
st.markdown(
    """
<style>
[data-testid="stMetricValue"] { font-size: 1.7rem; }
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
</style>
""",
    unsafe_allow_html=True,
)

if not os.path.exists(DATA_PATH):
    st.error("File data_cuaca.csv tidak ditemukan. Letakkan di folder yang sama dengan app.py.")
    st.stop()

df = muat_data(DATA_PATH)

# -----------------------------------------------------------------------------
# Sidebar: identitas + filter
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Identitas")
    st.write(f"**NIM:** {NIM}")
    st.write(f"**Nama:** {NAMA}")
    st.divider()

    st.header("Filter Data")
    semua_kota = sorted(df[K_KOTA].unique())
    pilih_kota = st.multiselect("Kota", semua_kota, default=semua_kota)
    semua_kond = urut_kondisi(df[K_KOND])
    pilih_kond = st.multiselect("Kondisi", semua_kond, default=semua_kond)

dff = df[df[K_KOTA].isin(pilih_kota) & df[K_KOND].isin(pilih_kond)]

if dff.empty:
    st.warning("Tidak ada data untuk filter yang dipilih. Ubah filter di sidebar.")
    st.stop()

# -----------------------------------------------------------------------------
# KPI
# -----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Jumlah Pengamatan", angka(len(dff)))
k2.metric("Rata-rata Suhu", f"{angka(dff[K_SUHU].mean(), 1)} °C")
k3.metric("Rata-rata Kelembaban", f"{angka(dff[K_LEMBAB].mean(), 1)} %")
k4.metric("Kondisi Terbanyak", dff[K_KOND].value_counts().index[0])

st.caption(
    f"Menampilkan {angka(len(dff))} dari {angka(len(df))} baris. "
    "Data berupa contoh (simulasi) yang mengikuti struktur pada materi; dua baris pertama sama dengan contoh di slide."
)

tab_data, tab_tipe, tab_kat, tab_num, tab_hub = st.tabs(
    ["1. Dataset", "2. Klasifikasi Tipe Data", "3. Kategorik", "4. Numerik", "5. Hubungan Antar Variabel"]
)

# -----------------------------------------------------------------------------
# Tab 1: Dataset
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset Pengamatan Cuaca")
    st.dataframe(
        dff,
        hide_index=True,
        column_config={
            K_SUHU: st.column_config.NumberColumn("Suhu (°C)", format="%.1f"),
            K_LEMBAB: st.column_config.NumberColumn("Kelembaban (%)", format="%d"),
        },
    )
    st.caption("Tabel mengikuti filter di sidebar.")

    st.subheader("Kualitas Data (file asli, sebelum filter)")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Jumlah baris", angka(len(df)))
    q2.metric("Jumlah kolom", angka(df.shape[1]))
    q3.metric("Nilai kosong", angka(int(df.isna().sum().sum())))
    q4.metric("Baris duplikat persis", angka(int(df.duplicated().sum())))
    st.caption(
        "Dataset ini tidak punya kolom tanggal atau jam pengamatan, sehingga tidak ada data waktu "
        "dan Line Chart untuk tren waktu tidak dipakai."
    )

    with st.expander("Statistik deskriptif (Suhu dan Kelembaban)"):
        st.dataframe(dff[[K_SUHU, K_LEMBAB]].describe().round(2))

# -----------------------------------------------------------------------------
# Tab 2: Klasifikasi tipe data
# -----------------------------------------------------------------------------
with tab_tipe:
    st.subheader("Klasifikasi Tipe Data per Kolom")
    klasifikasi = pd.DataFrame(
        {
            "Kolom": [K_KOTA, K_SUHU, K_LEMBAB, K_KOND],
            "Tipe Data": [
                "Kategorik (Nominal)",
                "Numerik - Interval",
                "Numerik - Rasio",
                "Kategorik (Nominal)",
            ],
            "Alasan": [
                "Label teks tanpa urutan",
                "Jarak antar suhu sama, tetapi 0 °C bukan berarti \"tidak ada suhu\". Aturan rasio tidak berlaku "
                "(20 °C bukan 2 kali lebih panas dari 10 °C)",
                "0% berarti benar-benar tidak ada kelembaban. Memiliki nol absolut",
                "Label teks tanpa urutan (Cerah, Berawan, Hujan, Badai Petir)",
            ],
            "Visualisasi": [
                "Bar Chart / Pie Chart",
                "Histogram, Box Plot, atau Scatter Plot",
                "Histogram, Box Plot, atau Scatter Plot",
                "Bar Chart / Pie Chart",
            ],
        }
    )
    st.table(klasifikasi.set_index("Kolom"))

    st.subheader("Interval vs Rasio: apa bedanya?")
    banding = pd.DataFrame(
        {
            "Pertanyaan": [
                "Apakah ada nol absolut?",
                "Apakah selisih nilai bermakna?",
                "Apakah perbandingan \"kali lipat\" bermakna?",
            ],
            "Suhu (°C) - Interval": [
                "Tidak. 0 °C bukan berarti tidak ada suhu",
                "Ya. 30 °C naik dari 20 °C berarti selisih 10 °C",
                "Tidak. 20 °C bukan 2 kali lebih panas dari 10 °C (dalam Kelvin: 293,15 K dibanding 283,15 K, "
                "hanya sekitar 1,04 kali)",
            ],
            "Kelembaban (%) - Rasio": [
                "Ya. 0% berarti tidak ada kelembaban sama sekali",
                "Ya. 80% naik dari 60% berarti selisih 20 poin persen",
                "Ya. Kelembaban 80% adalah 2 kali kelembaban 40%",
            ],
        }
    )
    st.table(banding.set_index("Pertanyaan"))

    st.markdown(
        """
- **Nominal**: label tanpa urutan (Kota, Kondisi). Cocok dengan Bar Chart atau Pie Chart.
- **Interval**: angka dengan jarak antar nilai yang sama, tetapi **tanpa nol absolut** (Suhu Celsius). Boleh dijumlah dan dikurangkan, tetapi tidak boleh dibandingkan dengan "kali lipat".
- **Rasio**: angka dengan jarak sama **dan nol absolut** (Kelembaban), sehingga semua operasi matematika, termasuk perbandingan "kali lipat", bermakna.
"""
    )
    st.caption(
        "Kondisi (Cerah, Berawan, Hujan, Badai Petir) tidak punya urutan tingkatan alami sehingga "
        "diklasifikasikan Nominal, sama seperti Kota."
    )

# -----------------------------------------------------------------------------
# Tab 3: Kategorik -> Bar Chart / Pie Chart
# -----------------------------------------------------------------------------
with tab_kat:
    st.subheader("Kota: Kategorik Nominal (Bar Chart)")
    ukuran = st.radio(
        "Ukuran pada Bar Chart",
        ["Jumlah Pengamatan", "Rata-rata Suhu (°C)", "Rata-rata Kelembaban (%)"],
        index=1,
        horizontal=True,
    )
    grup_kota = dff.groupby(K_KOTA)
    if ukuran == "Jumlah Pengamatan":
        nilai, fmt_u = grup_kota.size(), (lambda v: angka(v))
    elif ukuran == "Rata-rata Suhu (°C)":
        nilai, fmt_u = grup_kota[K_SUHU].mean(), (lambda v: f"{angka(v, 1)} °C")
    else:
        nilai, fmt_u = grup_kota[K_LEMBAB].mean(), (lambda v: f"{angka(v, 1)} %")
    nilai = nilai.sort_values(ascending=True)

    fig = go.Figure(
        go.Bar(
            x=nilai.to_numpy(),
            y=nilai.index.to_numpy(),
            orientation="h",
            marker_color=BIRU,
            text=[fmt_u(v) for v in nilai],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>" + ukuran + ": %{text}<extra></extra>",
        )
    )
    fig.update_layout(title=f"Bar Chart: {ukuran} per Kota")
    gaya(fig, 420)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=10, r=90, t=50, b=10))
    st.plotly_chart(fig, key="bar_kota")
    st.caption(
        "Kota tidak memiliki urutan, jadi batang diurutkan dari nilai terbesar agar mudah dibandingkan. "
        "Pada ukuran Suhu, bacalah selisih antar kota, bukan \"kali lipat\", karena Suhu berskala Interval."
    )

    st.subheader("Kondisi: Kategorik Nominal (Pie Chart dan Bar Chart)")
    urut_k = urut_kondisi(dff[K_KOND])
    per_kond = dff[K_KOND].value_counts().reindex(urut_k)
    kiri, kanan = st.columns(2)

    with kiri:
        fig = go.Figure(
            go.Pie(
                labels=per_kond.index.to_list(),
                values=per_kond.to_numpy(),
                hole=0.4,
                sort=False,
                marker=dict(colors=[warna_k(k) for k in per_kond.index], line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                textposition="outside",
                automargin=True,
            )
        )
        fig.update_layout(title="Pie Chart: Proporsi Kondisi Cuaca")
        gaya(fig, 400)
        fig.update_layout(margin=dict(l=70, r=70, t=60, b=40))
        st.plotly_chart(fig, key="pie_kond")

    with kanan:
        fig = go.Figure(
            go.Bar(
                x=per_kond.index.to_list(),
                y=per_kond.to_numpy(),
                marker_color=[warna_k(k) for k in per_kond.index],
                text=[angka(v) for v in per_kond],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>Jumlah pengamatan: %{y}<extra></extra>",
            )
        )
        fig.update_layout(title="Bar Chart: Jumlah Pengamatan per Kondisi")
        fig.update_xaxes(title="Kondisi")
        fig.update_yaxes(title="Jumlah pengamatan", showgrid=False)
        st.plotly_chart(gaya(fig, 400), key="bar_kond")

    st.subheader("Kondisi pada setiap Kota (Stacked Bar Chart)")
    tabel_silang = pd.crosstab(dff[K_KOTA], dff[K_KOND]).reindex(columns=urut_k, fill_value=0)
    kunci = tabel_silang["Hujan"] if "Hujan" in tabel_silang.columns else tabel_silang.sum(axis=1)
    tabel_silang = tabel_silang.loc[kunci.sort_values(ascending=True).index]
    fig = go.Figure()
    for k in urut_k:
        fig.add_trace(
            go.Bar(
                y=tabel_silang.index.to_list(),
                x=tabel_silang[k].to_numpy(),
                name=k,
                orientation="h",
                marker=dict(color=warna_k(k), line=dict(color="#FFFFFF", width=1.5)),
                hovertemplate="%{y}<br>" + k + ": %{x} pengamatan<extra></extra>",
            )
        )
    fig.update_layout(title="Stacked Bar Chart: Jumlah Pengamatan per Kondisi pada setiap Kota", barmode="stack")
    gaya(fig, 440)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", y=1.08, x=0, title_text="", traceorder="normal"),
    )
    fig.update_xaxes(title="Jumlah pengamatan")
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, key="stack_kota_kond")
    st.caption("Kota diurutkan dari jumlah pengamatan Hujan terbanyak (paling atas).")

# -----------------------------------------------------------------------------
# Tab 4: Numerik -> Histogram / Box Plot
# -----------------------------------------------------------------------------
with tab_num:
    st.subheader("Data Numerik: Histogram (distribusi) dan Box Plot (sebaran dan outlier)")
    nbins = st.slider("Jumlah bin histogram", 5, 30, 15)

    kolom_num = [
        (K_SUHU, "Interval", "suhu"),
        (K_LEMBAB, "Rasio", "lembab"),
    ]
    for kol, (nama_k, skala, key) in zip(st.columns(2), kolom_num):
        with kol:
            st.markdown(f"**{nama_k}** - Numerik {skala}")
            tampil(hist_kontinu(dff[nama_k], nama_k, 1, nbins), f"hist_{key}")
            tampil(box_kontinu(dff[nama_k], nama_k), f"box_{key}")
            st.caption(ringkas(dff[nama_k], 1))

    st.caption(
        "Histogram menjawab \"bagaimana sebaran nilainya?\", sedangkan Box Plot menjawab "
        "\"di mana nilai tengahnya dan mana yang tergolong outlier?\". Pada Suhu (Interval), rata-rata dan "
        "selisih bermakna, tetapi perbandingan \"kali lipat\" tidak."
    )

# -----------------------------------------------------------------------------
# Tab 5: Hubungan antar variabel
# -----------------------------------------------------------------------------
with tab_hub:
    st.subheader("Kelembaban vs Suhu: Scatter Plot (dua data numerik)")
    fig = go.Figure()
    for k in urut_kondisi(dff[K_KOND]):
        d = dff[dff[K_KOND] == k]
        fig.add_trace(
            go.Scatter(
                x=d[K_LEMBAB],
                y=d[K_SUHU],
                mode="markers",
                name=k,
                marker=dict(
                    color=warna_k(k),
                    symbol=simbol_k(k),
                    size=9,
                    opacity=0.8,
                    line=dict(color="#FFFFFF", width=1),
                ),
                customdata=d[[K_KOTA]].to_numpy(),
                hovertemplate=(
                    "%{customdata[0]} - " + k + "<br>Kelembaban: %{x}%<br>Suhu: %{y} °C<extra></extra>"
                ),
            )
        )
    fig.update_layout(title="Scatter Plot: Kelembaban vs Suhu, dibedakan menurut Kondisi")
    gaya(fig, 480)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0, title_text=""))
    fig.update_xaxes(title="Kelembaban (%)")
    fig.update_yaxes(title="Suhu (°C)")
    st.plotly_chart(fig, key="scatter")
    if dff[K_SUHU].nunique() > 1 and dff[K_LEMBAB].nunique() > 1:
        r = dff[K_SUHU].corr(dff[K_LEMBAB], method="spearman")
        st.caption(
            f"Korelasi Spearman Suhu dan Kelembaban: {angka(r, 2)}. Tanda negatif berarti makin lembab, "
            "suhu cenderung makin rendah (sejalan dengan kondisi hujan yang lebih lembab dan lebih sejuk). "
            "Bentuk titik juga dibedakan, bukan hanya warna."
        )

    st.subheader("Sebaran per Kelompok: Box Plot")
    c1, c2 = st.columns(2)
    var_box = c1.selectbox("Variabel", [K_SUHU, K_LEMBAB])
    grup_box = c2.radio("Kelompokkan berdasarkan", [K_KOTA, K_KOND], horizontal=True)

    if grup_box == K_KOTA:
        urut = dff.groupby(K_KOTA)[var_box].median().sort_values(ascending=False).index.tolist()
        fig = px.box(
            dff,
            x=K_KOTA,
            y=var_box,
            points="outliers",
            category_orders={K_KOTA: urut},
            color_discrete_sequence=[ORANYE],
        )
    else:
        urut = urut_kondisi(dff[K_KOND])
        fig = px.box(
            dff,
            x=K_KOND,
            y=var_box,
            color=K_KOND,
            points="outliers",
            category_orders={K_KOND: urut},
            color_discrete_map={k: warna_k(k) for k in urut},
        )
    fig.update_layout(title=f"Box Plot: {var_box} per {grup_box}")
    fig.update_xaxes(title=grup_box)
    fig.update_yaxes(title=var_box)
    st.plotly_chart(gaya(fig, 440), key="box_kel")

# -----------------------------------------------------------------------------
# Footer identitas
# -----------------------------------------------------------------------------
st.divider()
st.caption(f"Studi Kasus 3 - Dikerjakan oleh {NAMA} (NIM {NIM})")
