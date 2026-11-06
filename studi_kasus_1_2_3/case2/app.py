"""
STUDI KASUS 2: DATASET PROFIL MAHASISWA
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

K_JUR = "Jurusan"
K_SEM = "Semester"
K_IPK = "IPK"
K_USIA = "Usia"

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_mahasiswa.csv")

st.set_page_config(
    page_title=f"Case 2 | {NIM} | {NAMA}",
    page_icon="🎓",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Fungsi bantu
# -----------------------------------------------------------------------------
def angka(v, d=0):
    """Format angka gaya Indonesia: titik ribuan, koma desimal (3,85)."""
    s = f"{v:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


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
            hovertemplate="%{customdata}<br>Frekuensi: %{y:,} mahasiswa<extra></extra>",
        )
    )
    fig.update_xaxes(title=label)
    fig.update_yaxes(title="Frekuensi (jumlah mahasiswa)")
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
st.title("🎓 Bedah Kasus 2: Dataset Profil Mahasiswa")
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
    st.error("File data_mahasiswa.csv tidak ditemukan. Letakkan di folder yang sama dengan app.py.")
    st.stop()

df = muat_data(DATA_PATH)

# -----------------------------------------------------------------------------
# Sidebar: identitas + filter
# -----------------------------------------------------------------------------
sem_min, sem_max = int(df[K_SEM].min()), int(df[K_SEM].max())

with st.sidebar:
    st.header("Identitas")
    st.write(f"**NIM:** {NIM}")
    st.write(f"**Nama:** {NAMA}")
    st.divider()

    st.header("Filter Data")
    semua_jur = sorted(df[K_JUR].unique())
    pilih_jur = st.multiselect("Jurusan", semua_jur, default=semua_jur)
    rentang_sem = st.slider("Rentang Semester", sem_min, sem_max, (sem_min, sem_max))

dff = df[df[K_JUR].isin(pilih_jur) & df[K_SEM].between(rentang_sem[0], rentang_sem[1])]

if dff.empty:
    st.warning("Tidak ada data untuk filter yang dipilih. Ubah filter di sidebar.")
    st.stop()

# -----------------------------------------------------------------------------
# KPI
# -----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Jumlah Mahasiswa", angka(len(dff)))
k2.metric("Rata-rata IPK", angka(dff[K_IPK].mean(), 2))
k3.metric("Median Usia", f"{angka(dff[K_USIA].median(), 1)} tahun")
k4.metric("Jumlah Jurusan", angka(dff[K_JUR].nunique()))

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
    st.subheader("Dataset Profil Mahasiswa")
    st.dataframe(
        dff,
        hide_index=True,
        column_config={K_IPK: st.column_config.NumberColumn("IPK", format="%.2f")},
    )
    st.caption("Tabel mengikuti filter di sidebar.")

    st.subheader("Kualitas Data (file asli, sebelum filter)")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Jumlah baris", angka(len(df)))
    q2.metric("Jumlah kolom", angka(df.shape[1]))
    q3.metric("Nilai kosong", angka(int(df.isna().sum().sum())))
    q4.metric("Baris duplikat persis", angka(int(df.duplicated().sum())))
    st.caption(
        "Dataset ini tidak punya kolom pengenal (misalnya NIM mahasiswa), sehingga baris yang sama "
        "bisa jadi mahasiswa yang berbeda dan tidak otomatis dihapus."
    )

    with st.expander("Statistik deskriptif (IPK dan Usia)"):
        st.dataframe(dff[[K_IPK, K_USIA]].describe().round(2))

# -----------------------------------------------------------------------------
# Tab 2: Klasifikasi tipe data
# -----------------------------------------------------------------------------
with tab_tipe:
    st.subheader("Klasifikasi Tipe Data per Kolom")
    klasifikasi = pd.DataFrame(
        {
            "Kolom": [K_JUR, K_SEM, K_IPK, K_USIA],
            "Tipe Data": [
                "Kategorik (Nominal)",
                "Kategorik (Ordinal)",
                "Numerik (Kontinu)",
                "Numerik (Kontinu)",
            ],
            "Alasan": [
                "Hanya berupa label pembeda, tidak ada urutan",
                "Ada tingkatan dan urutan yang jelas (1 → 2 → 3), tetapi jarak antar nilai belum tentu sama",
                "Berupa rentang/desimal hasil pengukuran",
                "Berupa rentang hasil pengukuran (lama hidup dalam tahun)",
            ],
            "Visualisasi": [
                "Bar Chart / Pie Chart",
                "Bar Chart Berurutan",
                "Histogram (melihat distribusi) atau Box Plot (melihat sebaran dan outlier)",
                "Histogram (melihat distribusi) atau Box Plot (melihat sebaran dan outlier)",
            ],
        }
    )
    st.table(klasifikasi.set_index("Kolom"))

    st.markdown(
        """
- **Nominal**: label tanpa urutan (contoh: Jurusan). Cocok dengan Bar Chart atau Pie Chart.
- **Ordinal**: ada urutan/tingkatan, tetapi jarak antar nilai belum tentu sama (contoh: Semester). Cocok dengan Bar Chart yang **berurutan** sesuai tingkatannya, bukan diurutkan dari yang terbanyak.
- **Kontinu**: nilai berada dalam rentang tertentu dan bisa berupa pecahan/desimal (contoh: IPK). Cocok dengan Histogram, Box Plot, atau Scatter Plot.
"""
    )
    st.info(
        "Semester tampak seperti angka, tetapi diperlakukan sebagai kategorik ordinal. Semester 4 tidak bisa "
        "dikatakan \"dua kali\" semester 2, dan rata-rata semester tidak bermakna sebagai ukuran. "
        "Karena itu Semester digambar dengan Bar Chart berurutan, bukan Histogram."
    )
    st.caption(
        "Catatan: pada data ini Usia dicatat dalam tahun bulat (19, 20, ...). Mengikuti materi, Usia tetap "
        "diklasifikasikan Numerik Kontinu karena usia pada dasarnya hasil pengukuran waktu."
    )

# -----------------------------------------------------------------------------
# Tab 3: Kategorik -> Bar Chart / Pie Chart / Bar Chart Berurutan
# -----------------------------------------------------------------------------
with tab_kat:
    st.subheader("Jurusan: Kategorik Nominal (Bar Chart dan Pie Chart)")
    kiri, kanan = st.columns(2)

    vc = dff[K_JUR].value_counts()

    with kiri:
        asc = vc.sort_values(ascending=True)
        fig = go.Figure(
            go.Bar(
                x=asc.to_numpy(),
                y=asc.index.to_numpy(),
                orientation="h",
                marker_color=BIRU,
                text=[angka(v) for v in asc],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>Jumlah mahasiswa: %{text}<extra></extra>",
            )
        )
        fig.update_layout(title="Bar Chart: Jumlah Mahasiswa per Jurusan")
        gaya(fig, 400)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(margin=dict(l=10, r=70, t=50, b=10))
        st.plotly_chart(fig, key="bar_jur")

    with kanan:
        if len(vc) > 6:
            lain = f"Lainnya ({len(vc) - 5})"
            vc_pie = pd.concat([vc.iloc[:5], pd.Series({lain: vc.iloc[5:].sum()})])
            warna = BIRU_PIE[:5] + [ABU]
        else:
            vc_pie, warna = vc, BIRU_PIE[: len(vc)]
        fig = go.Figure(
            go.Pie(
                labels=vc_pie.index.to_list(),
                values=vc_pie.to_numpy(),
                hole=0.4,
                sort=False,
                marker=dict(colors=warna, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                textposition="outside",
                automargin=True,
            )
        )
        fig.update_layout(title="Pie Chart: Proporsi Mahasiswa per Jurusan")
        gaya(fig, 400)
        fig.update_layout(margin=dict(l=70, r=70, t=60, b=40))
        st.plotly_chart(fig, key="pie_jur")

    st.caption(
        f"Jurusan terbanyak: **{vc.index[0]}** ({angka(int(vc.iloc[0]))} mahasiswa, "
        f"{angka(vc.iloc[0] / len(dff) * 100, 1)}%). Jurusan tidak memiliki urutan, jadi batang diurutkan dari "
        "yang terbanyak agar mudah dibandingkan."
    )

    st.subheader("Semester: Kategorik Ordinal (Bar Chart Berurutan)")
    urut_sem = list(range(sem_min, sem_max + 1))
    per_sem = dff[K_SEM].value_counts().reindex(urut_sem, fill_value=0)
    fig = go.Figure(
        go.Bar(
            x=[str(s) for s in urut_sem],
            y=per_sem.to_numpy(),
            marker_color=BIRU,
            text=[angka(v) for v in per_sem],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="Semester %{x}<br>Jumlah mahasiswa: %{y}<extra></extra>",
        )
    )
    fig.update_layout(title="Bar Chart Berurutan: Jumlah Mahasiswa per Semester")
    fig.update_xaxes(title="Semester", type="category", categoryorder="array", categoryarray=[str(s) for s in urut_sem])
    fig.update_yaxes(title="Jumlah mahasiswa", showgrid=False)
    st.plotly_chart(gaya(fig, 360), key="bar_sem")
    st.caption(
        "Sumbu X mengikuti urutan semester (1 → 8), bukan diurutkan dari yang terbanyak, "
        "karena Semester adalah data ordinal."
    )

# -----------------------------------------------------------------------------
# Tab 4: Numerik -> Histogram / Box Plot
# -----------------------------------------------------------------------------
with tab_num:
    st.subheader("Data Numerik Kontinu: Histogram (distribusi) dan Box Plot (sebaran dan outlier)")
    nbins = st.slider("Jumlah bin histogram", 5, 30, 12)

    kolom_num = [
        (K_IPK, 2, "ipk"),
        (K_USIA, 1, "usia"),
    ]
    for kol, (nama_k, d, key) in zip(st.columns(2), kolom_num):
        with kol:
            tampil(hist_kontinu(dff[nama_k], nama_k, d, nbins), f"hist_{key}")
            tampil(box_kontinu(dff[nama_k], nama_k), f"box_{key}")
            st.caption(ringkas(dff[nama_k], d))

    st.caption(
        "Histogram menjawab \"bagaimana sebaran nilainya?\", sedangkan Box Plot menjawab "
        "\"di mana nilai tengahnya dan mana yang tergolong outlier?\"."
    )

# -----------------------------------------------------------------------------
# Tab 5: Hubungan antar variabel
# -----------------------------------------------------------------------------
with tab_hub:
    st.subheader("Sebaran IPK per Jurusan: Box Plot")
    urut_jur = dff.groupby(K_JUR)[K_IPK].median().sort_values(ascending=True).index.tolist()
    fig = px.box(
        dff,
        x=K_IPK,
        y=K_JUR,
        orientation="h",
        points="outliers",
        category_orders={K_JUR: urut_jur},
        color_discrete_sequence=[BIRU],
        title="Box Plot: IPK per Jurusan (diurutkan dari median tertinggi di atas)",
    )
    fig.update_xaxes(title="IPK")
    fig.update_yaxes(title="")
    st.plotly_chart(gaya(fig, 420), key="box_jur")

    st.subheader("Sebaran IPK per Semester: Box Plot Berurutan")
    fig = px.box(
        dff,
        x=K_SEM,
        y=K_IPK,
        points="outliers",
        category_orders={K_SEM: urut_sem},
        color_discrete_sequence=[BIRU],
        title="Box Plot: IPK per Semester",
    )
    fig.update_xaxes(type="category", title="Semester")
    fig.update_yaxes(title="IPK")
    st.plotly_chart(gaya(fig, 400), key="box_sem")

    st.subheader("Usia vs IPK: Scatter Plot (dua data numerik kontinu)")
    fig = px.scatter(
        dff,
        x=K_USIA,
        y=K_IPK,
        opacity=0.55,
        hover_data=[K_JUR, K_SEM],
        color_discrete_sequence=[ORANYE],
        title="Scatter Plot: Usia vs IPK",
    )
    fig.update_traces(marker=dict(size=8))
    fig.update_xaxes(title="Usia (tahun)", dtick=1)
    fig.update_yaxes(title="IPK")
    st.plotly_chart(gaya(fig, 420), key="scatter")
    if dff[K_USIA].nunique() > 1 and dff[K_IPK].nunique() > 1:
        r = dff[K_USIA].corr(dff[K_IPK], method="spearman")
        st.caption(
            f"Korelasi Spearman Usia dan IPK: {angka(r, 2)} (mendekati 0 berarti hampir tidak berhubungan). "
            "Titik tampak berbaris vertikal karena Usia dicatat dalam tahun bulat."
        )

# -----------------------------------------------------------------------------
# Footer identitas
# -----------------------------------------------------------------------------
st.divider()
st.caption(f"Studi Kasus 2 - Dikerjakan oleh {NAMA} (NIM {NIM})")
