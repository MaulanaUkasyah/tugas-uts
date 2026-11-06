"""
STUDI KASUS 1: DATASET HARGA RUMAH
Dashboard Streamlit - Klasifikasi Tipe Data & Pemilihan Visualisasi

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""

import math
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
BIRU_PIE = ["#0B3D6B", "#14588F", "#1F74B3", "#2B8FD0", "#5AAAE0", "#8CC4EC"]
ABU = "#9CA3AF"
ORANYE = "#E8890C"
DIVERGING = [[0.0, "#E8890C"], [0.5, "#F3F4F6"], [1.0, "#2B8FD0"]]

K_HARGA = "Harga"
K_KT = "Kamar Tidur"
K_KM = "Kamar Mandi"
K_GAR = "Garasi"
K_LT = "Luas Tanah"
K_LB = "Luas Bangunan"
K_KEC = "Kecamatan"

NUMERIK = [K_HARGA, K_KT, K_KM, K_GAR, K_LT, K_LB]
DISKRIT = [K_KT, K_KM, K_GAR]

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_rumah.csv")

st.set_page_config(
    page_title=f"Case 1 | {NIM} | {NAMA}",
    page_icon="🏠",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Fungsi bantu: format angka gaya Indonesia (titik ribuan, koma desimal)
# -----------------------------------------------------------------------------
def angka(v, d=0):
    s = f"{v:,.{d}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def rp(v, d=1):
    """Rupiah singkat: Rp 1,4 miliar / Rp 830 juta."""
    if v >= 1e12:
        return f"Rp {angka(v / 1e12, d)} triliun"
    if v >= 1e9:
        return f"Rp {angka(v / 1e9, d)} miliar"
    if v >= 1e6:
        return f"Rp {angka(v / 1e6, d)} juta"
    return f"Rp {angka(v)}"


def m2(v, d=0):
    return f"{angka(v, d)} m²"


def tick_angka(v, d=0):
    return angka(v, d)


def rp_tick(v, d=0):
    """Label sumbu yang sangat singkat: 10 jt, 1 M (jt = juta, M = miliar)."""
    if v >= 1e12:
        return f"{angka(v / 1e12, d)} T"
    if v >= 1e9:
        return f"{angka(v / 1e9, d)} M"
    if v >= 1e6:
        return f"{angka(v / 1e6, d)} jt"
    return angka(v)


@st.cache_data
def muat_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


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


def tick_log(fig, sumbu, seri, fmt):
    """Label sumbu logaritmik: pangkat 10 ditulis sebagai Rp 1 juta, 10 juta, dst."""
    e0 = math.floor(math.log10(seri.min()))
    e1 = math.ceil(math.log10(seri.max()))
    nilai = [10.0**e for e in range(e0, e1 + 1)]
    update = fig.update_xaxes if sumbu == "x" else fig.update_yaxes
    update(tickvals=nilai, ticktext=[fmt(v, 0) for v in nilai])


def hist_kontinu(seri, label, fmt, d_bin, fmt_tick, log, nbins):
    """Histogram untuk data numerik kontinu. Mode log: lebar bin sama di skala log10."""
    x = seri.to_numpy(dtype=float)
    if x.size < 2 or x.min() == x.max():
        return None
    if log:
        e = np.linspace(np.log10(x.min()), np.log10(x.max()), nbins + 1)
        counts, _ = np.histogram(np.log10(x), bins=e)
        batas = 10.0**e
        pos, lebar = (e[:-1] + e[1:]) / 2, np.diff(e) * 0.94
    else:
        counts, batas = np.histogram(x, bins=nbins)
        pos, lebar = (batas[:-1] + batas[1:]) / 2, np.diff(batas) * 0.94
    teks = [f"{fmt(a, d_bin)} – {fmt(b, d_bin)}" for a, b in zip(batas[:-1], batas[1:])]
    fig = go.Figure(
        go.Bar(
            x=pos,
            y=counts,
            width=lebar,
            marker_color=ORANYE,
            customdata=teks,
            hovertemplate="%{customdata}<br>Frekuensi: %{y:,} rumah<extra></extra>",
        )
    )
    if log:
        # label di 1 dan 3 kali pangkat 10 (misalnya 100 jt, 300 jt, 1 M, 3 M), hanya yang ada di rentang data
        tv = [
            p + math.log10(k)
            for p in range(math.floor(e[0]), math.ceil(e[-1]) + 1)
            for k in (1, 3)
            if e[0] - 0.02 <= p + math.log10(k) <= e[-1] + 0.02
        ]
        fig.update_xaxes(tickvals=tv, ticktext=[fmt_tick(round(10.0**t, 6), 0) for t in tv])
    else:
        fig.update_xaxes(tickformat=",.0f")
    fig.update_xaxes(tickangle=0, title=label + (", skala log" if log else ""))
    fig.update_yaxes(title="Frekuensi (jumlah rumah)")
    fig.update_layout(title=f"Histogram {label}")
    return gaya(fig, 340)


def hist_diskrit(seri, label):
    """Histogram untuk data numerik diskrit: satu batang untuk tiap nilai bulat."""
    vc = seri.value_counts().sort_index()
    fig = go.Figure(
        go.Bar(
            x=vc.index.to_numpy(),
            y=vc.to_numpy(),
            width=0.85,
            marker_color=ORANYE,
            hovertemplate=f"{label}: %{{x}}<br>Frekuensi: %{{y:,}} rumah<extra></extra>",
        )
    )
    if vc.index.max() - vc.index.min() <= 20:
        fig.update_xaxes(dtick=1)
    fig.update_xaxes(title=label)
    fig.update_yaxes(title="Frekuensi (jumlah rumah)")
    fig.update_layout(title=f"Histogram {label}")
    return gaya(fig, 320)


# -----------------------------------------------------------------------------
# Header + identitas (WAJIB tampil di dashboard)
# -----------------------------------------------------------------------------
st.title("🏠 Bedah Kasus 1: Dataset Harga Rumah")
st.markdown(f"**NIM:** {NIM} &nbsp;&nbsp;|&nbsp;&nbsp; **Nama Mahasiswa:** {NAMA}")

if NIM.startswith("ISI_") or NAMA.startswith("ISI_"):
    st.warning("NIM dan NAMA belum diisi. Buka app.py lalu ganti nilai NIM dan NAMA di bagian atas file.")

# Angka KPI jangan terpotong titik-titik (contoh: "Rp 1,35 m...") di layar yang sempit
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
    st.error("File data_rumah.csv tidak ditemukan. Letakkan di folder yang sama dengan app.py.")
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
    semua_kec = sorted(df[K_KEC].unique())
    pilih_kec = st.multiselect("Kecamatan", semua_kec, default=semua_kec)

    st.subheader("Nilai ekstrem")
    sembunyi = st.checkbox(
        "Sembunyikan nilai ekstrem (outlier)",
        value=True,
        help="Menyembunyikan rumah dengan nilai tidak wajar: Harga di luar batas bawah-atas persentil, "
        "atau kolom numerik lain di atas batas persentil. File data asli tidak diubah.",
    )
    persen = st.select_slider(
        "Batas persentil",
        options=[95.0, 97.5, 99.0, 99.5],
        value=99.0,
        format_func=lambda v: f"P{angka(v, 1)}",
        disabled=not sembunyi,
    )
    hapus_dup = st.checkbox("Hapus baris duplikat persis", value=False)

# Batas outlier dihitung dari SELURUH data, supaya tidak bergeser saat filter diubah.
# Batas atas berlaku untuk semua kolom numerik. Batas bawah hanya untuk Harga, karena ada harga
# yang tidak wajar (misalnya Rp 1,3 juta untuk rumah 92 m²).
batas = df[NUMERIK].quantile(persen / 100)
batas_bawah_harga = df[K_HARGA].quantile(1 - persen / 100)
wajar = (df[NUMERIK] <= batas).all(axis=1) & (df[K_HARGA] >= batas_bawah_harga)

dff = df
if pilih_kec:
    dff = dff[dff[K_KEC].isin(pilih_kec)]
else:
    dff = dff.iloc[0:0]
if hapus_dup:
    dff = dff.drop_duplicates()
if sembunyi:
    dff = dff[wajar.loc[dff.index]]

if dff.empty:
    st.warning("Tidak ada data untuk filter yang dipilih. Ubah filter di sidebar.")
    st.stop()

# -----------------------------------------------------------------------------
# KPI
# -----------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Jumlah Rumah", angka(len(dff)))
k2.metric("Median Harga", rp(dff[K_HARGA].median(), 2))
k3.metric("Rata-rata Harga", rp(dff[K_HARGA].mean(), 2))
k4.metric("Median Luas Tanah", m2(dff[K_LT].median()))
k5.metric("Median Luas Bangunan", m2(dff[K_LB].median()))

catatan = f"Menampilkan {angka(len(dff))} dari {angka(len(df))} baris."
if sembunyi:
    catatan += (
        f" {angka(int((~wajar).sum()))} baris dengan nilai ekstrem disembunyikan "
        f"(Harga di luar P{angka(100 - persen, 1)}-P{angka(persen, 1)}, "
        f"atau kolom numerik lain di atas P{angka(persen, 1)})."
    )
st.caption(catatan)

tab_data, tab_tipe, tab_kat, tab_num, tab_hub = st.tabs(
    ["1. Dataset", "2. Klasifikasi Tipe Data", "3. Kategorik", "4. Numerik", "5. Hubungan Antar Variabel"]
)

# -----------------------------------------------------------------------------
# Tab 1: Dataset + kualitas data
# -----------------------------------------------------------------------------
with tab_data:
    st.subheader("Dataset Harga Rumah")
    st.dataframe(
        dff,
        hide_index=True,
        column_config={
            K_HARGA: st.column_config.NumberColumn("Harga (Rp)", format="localized"),
            K_LT: st.column_config.NumberColumn("Luas Tanah (m²)", format="localized"),
            K_LB: st.column_config.NumberColumn("Luas Bangunan (m²)", format="localized"),
        },
    )
    st.caption("Tabel mengikuti filter di sidebar.")

    st.subheader("Kualitas Data (file asli, sebelum filter)")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Jumlah baris", angka(len(df)))
    q2.metric("Jumlah kolom", angka(df.shape[1]))
    q3.metric("Nilai kosong", angka(int(df.isna().sum().sum())))
    q4.metric("Baris duplikat persis", angka(int(df.duplicated().sum())))

    def hitung_luar(c):
        luar = df[c] > batas[c]
        if c == K_HARGA:
            luar = luar | (df[c] < batas_bawah_harga)
        return int(luar.sum())

    kol_bawah = f"Batas bawah P{angka(100 - persen, 1)}"
    kol_atas = f"Batas atas P{angka(persen, 1)}"
    outlier = pd.DataFrame(
        {
            "Kolom": NUMERIK,
            "Nilai minimum": [angka(df[c].min()) for c in NUMERIK],
            "Nilai maksimum": [angka(df[c].max()) for c in NUMERIK],
            kol_bawah: [angka(batas_bawah_harga) if c == K_HARGA else "-" for c in NUMERIK],
            kol_atas: [angka(batas[c]) for c in NUMERIK],
            "Baris di luar batas": [angka(hitung_luar(c)) for c in NUMERIK],
        }
    )
    st.dataframe(outlier, hide_index=True)
    st.caption(
        "Nilai yang jauh di luar batas menandakan data tidak wajar, misalnya luas tanah ratusan juta m², "
        "rumah 43 m² berharga ratusan miliar, atau rumah ratusan m² berharga hanya beberapa juta rupiah. "
        "Nilai tersebut disembunyikan dari grafik secara default agar bentuk sebaran terlihat jelas."
    )

# -----------------------------------------------------------------------------
# Tab 2: Klasifikasi tipe data
# -----------------------------------------------------------------------------
with tab_tipe:
    st.subheader("Klasifikasi Tipe Data per Kolom")
    klasifikasi = pd.DataFrame(
        {
            "Kolom": [K_HARGA, K_KT, K_KM, K_GAR, K_LT, K_LB, K_KEC],
            "Tipe Data": [
                "Numerik (Kontinu) - Rasio",
                "Numerik (Diskrit) - Rasio",
                "Numerik (Diskrit) - Rasio",
                "Numerik (Diskrit) - Rasio",
                "Numerik (Kontinu) - Rasio",
                "Numerik (Kontinu) - Rasio",
                "Kategorik (Nominal)",
            ],
            "Alasan": [
                "Hasil pengukuran nilai uang, rentang lebar, memiliki nol absolut (Rp 0 = gratis)",
                "Hasil menghitung ruangan (bilangan bulat), memiliki nol absolut",
                "Hasil menghitung ruangan (bilangan bulat), memiliki nol absolut",
                "Hasil menghitung jumlah garasi (bilangan bulat); 0 berarti tidak ada garasi",
                "Hasil pengukuran luas (m²), bisa pecahan, 0 m² berarti tidak ada tanah",
                "Hasil pengukuran luas (m²), bisa pecahan, 0 m² berarti tidak ada bangunan",
                "Nama wilayah/label tanpa urutan",
            ],
            "Visualisasi": [
                "Histogram",
                "Histogram (satu batang per nilai)",
                "Histogram (satu batang per nilai)",
                "Histogram (satu batang per nilai)",
                "Histogram",
                "Histogram",
                "Bar Chart / Pie Chart",
            ],
        }
    )
    st.table(klasifikasi.set_index("Kolom"))

    st.markdown(
        """
- **Nominal**: data kelompok/label. Tidak ada urutan dan tidak bisa dijumlahkan atau dirata-ratakan (contoh: Kecamatan).
- **Rasio**: data angka yang punya **nol absolut**, sehingga perbandingan seperti "2 kali lebih mahal" atau "2 kali lebih luas" bermakna.
- **Kontinu vs Diskrit**: harga dan luas hasil pengukuran sehingga bisa bernilai pecahan (kontinu); kamar tidur, kamar mandi, dan garasi hasil menghitung sehingga selalu bilangan bulat (diskrit).
"""
    )
    st.info(
        "Dataset ini tidak memiliki kolom tanggal/waktu, sehingga tidak ada data bertipe Waktu - Interval "
        "dan Line Chart untuk tren waktu tidak dipakai. Sebagai gantinya, tab 5 menampilkan hubungan antar variabel."
    )

# -----------------------------------------------------------------------------
# Tab 3: Kategorik (Kecamatan) -> Bar Chart / Pie Chart
# -----------------------------------------------------------------------------
with tab_kat:
    st.subheader("Data Kategorik (Nominal): Kecamatan")
    ukuran = st.radio(
        "Ukuran pada Bar Chart",
        ["Jumlah Rumah", "Median Harga", "Median Luas Bangunan"],
        horizontal=True,
    )

    grup = dff.groupby(K_KEC)
    if ukuran == "Jumlah Rumah":
        nilai, fmt_u = grup.size(), (lambda v: angka(v))
    elif ukuran == "Median Harga":
        nilai, fmt_u = grup[K_HARGA].median(), (lambda v: rp(v, 2))
    else:
        nilai, fmt_u = grup[K_LB].median(), (lambda v: m2(v))
    nilai = nilai.sort_values(ascending=True)

    kiri, kanan = st.columns(2)

    with kiri:
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
        fig.update_layout(title=f"Bar Chart: {ukuran} per Kecamatan")
        gaya(fig, 440)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(margin=dict(l=10, r=100, t=50, b=10))
        st.plotly_chart(fig, key="bar_kec")

    with kanan:
        vc = dff[K_KEC].value_counts()
        if len(vc) > 7:
            lain = f"Lainnya ({len(vc) - 6})"
            vc = pd.concat([vc.iloc[:6], pd.Series({lain: vc.iloc[6:].sum()})])
            warna = BIRU_PIE + [ABU]
        else:
            warna = BIRU_PIE[: len(vc)]
        fig = go.Figure(
            go.Pie(
                labels=vc.index.to_list(),
                values=vc.to_numpy(),
                hole=0.4,
                sort=False,
                marker=dict(colors=warna, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                textposition="outside",
                automargin=True,
            )
        )
        fig.update_layout(title="Pie Chart: Proporsi Jumlah Rumah per Kecamatan")
        gaya(fig, 440)
        fig.update_layout(margin=dict(l=70, r=70, t=60, b=40))
        st.plotly_chart(fig, key="pie_kec")
        if len(dff[K_KEC].unique()) > 7:
            st.caption("Angka dalam kurung pada \"Lainnya\" = jumlah kecamatan yang digabung.")

    jumlah = dff[K_KEC].value_counts()
    median_harga = dff.groupby(K_KEC)[K_HARGA].median().sort_values(ascending=False)
    st.caption(
        f"Rumah terbanyak ada di **{jumlah.index[0]}** ({angka(int(jumlah.iloc[0]))} rumah). "
        f"Median harga tertinggi ada di **{median_harga.index[0]}** ({rp(median_harga.iloc[0], 2)}). "
        "Bar Chart lebih mudah dibaca untuk banyak kategori; Pie Chart menunjukkan proporsi terhadap keseluruhan."
    )

# -----------------------------------------------------------------------------
# Tab 4: Numerik -> Histogram
# -----------------------------------------------------------------------------
with tab_num:
    st.subheader("Data Numerik Kontinu (Rasio): Histogram")
    c1, c2 = st.columns([1, 1])
    log_skala = c1.checkbox(
        "Skala logaritmik (disarankan, rentang harga dan luas sangat lebar)", value=True
    )
    nbins = c2.slider("Jumlah bin", 10, 40, 25)

    kolom_kontinu = [
        (K_HARGA, "Harga (Rp)", rp, 2, rp_tick, "hist_harga"),
        (K_LT, "Luas Tanah (m²)", m2, 0, tick_angka, "hist_lt"),
        (K_LB, "Luas Bangunan (m²)", m2, 0, tick_angka, "hist_lb"),
    ]
    for kol, (nama_k, judul_x, fmt, d_bin, fmt_tick, key) in zip(st.columns(3), kolom_kontinu):
        with kol:
            tampil(hist_kontinu(dff[nama_k], judul_x, fmt, d_bin, fmt_tick, log_skala, nbins), key)
            if nama_k == K_HARGA:
                st.caption(f"Median: {rp(dff[nama_k].median(), 2)} | Rata-rata: {rp(dff[nama_k].mean(), 2)}")
            else:
                st.caption(f"Median: {m2(dff[nama_k].median())} | Rata-rata: {m2(dff[nama_k].mean(), 1)}")

    st.caption("Keterangan sumbu Harga: jt = juta, M = miliar.")

    st.subheader("Data Numerik Diskrit (Rasio): Histogram")
    for kol, nama_k in zip(st.columns(3), DISKRIT):
        with kol:
            tampil(hist_diskrit(dff[nama_k], nama_k), f"hist_{nama_k}")
            st.caption(f"Median: {angka(dff[nama_k].median(), 1)} | Terbanyak: {int(dff[nama_k].mode().iloc[0])}")

    with st.expander("Statistik deskriptif"):
        st.dataframe(dff[NUMERIK].describe().round(1))

# -----------------------------------------------------------------------------
# Tab 5: Hubungan antar variabel
# -----------------------------------------------------------------------------
with tab_hub:
    st.subheader("Hubungan Luas dengan Harga: Scatter Plot")
    s1, s2 = st.columns([2, 1])
    xvar = s1.selectbox("Sumbu X", [K_LB, K_LT])
    log_sc = s2.checkbox("Skala logaritmik", value=True, key="log_sc")

    sampel = dff.sample(min(len(dff), 4000), random_state=42)
    fig = px.scatter(
        sampel,
        x=xvar,
        y=K_HARGA,
        log_x=log_sc,
        log_y=log_sc,
        opacity=0.35,
        hover_data=[K_KEC, K_KT, K_KM, K_GAR],
        color_discrete_sequence=[BIRU],
        title=f"Scatter Plot: {xvar} vs Harga",
    )
    fig.update_traces(marker=dict(size=5))
    if log_sc:
        tick_log(fig, "x", dff[xvar], m2)
        tick_log(fig, "y", dff[K_HARGA], rp)
    else:
        fig.update_xaxes(tickformat=",.0f")
        fig.update_yaxes(tickformat=",.0f")
    fig.update_xaxes(title=f"{xvar} (m²)")
    fig.update_yaxes(title="Harga (Rp)")
    st.plotly_chart(gaya(fig, 460), key="scatter")
    st.caption(f"Menampilkan {angka(len(sampel))} titik sampel acak dari {angka(len(dff))} rumah agar grafik tetap ringan.")

    st.subheader("Sebaran Harga per Kelompok: Box Plot")
    pilih_grup = st.selectbox("Kelompokkan berdasarkan", [K_KEC, K_KT, K_KM, K_GAR])
    if pilih_grup == K_KEC:
        urut = dff.groupby(K_KEC)[K_HARGA].median().sort_values(ascending=False).index.tolist()
    else:
        urut = sorted(dff[pilih_grup].unique().tolist())
    fig = px.box(
        dff,
        x=pilih_grup,
        y=K_HARGA,
        log_y=True,
        points=False,
        category_orders={pilih_grup: urut},
        color_discrete_sequence=[BIRU],
        title=f"Box Plot: Harga per {pilih_grup} (skala log)",
    )
    fig.update_xaxes(type="category")
    tick_log(fig, "y", dff[K_HARGA], rp)
    fig.update_yaxes(title="Harga (Rp)")
    st.plotly_chart(gaya(fig, 440), key="box")
    st.caption("Garis tengah kotak = median, kotak = 50% data di tengah (kuartil 1 sampai 3).")

    st.subheader("Korelasi Antar Variabel Numerik: Heatmap")
    korelasi = dff[NUMERIK].corr(method="spearman")
    fig = px.imshow(
        korelasi,
        text_auto=".2f",
        color_continuous_scale=DIVERGING,
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Heatmap Korelasi Spearman",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Korelasi"))
    st.plotly_chart(gaya(fig, 460), key="heatmap")
    st.caption(
        "Korelasi Spearman dipakai karena tidak terlalu terpengaruh nilai ekstrem. "
        "Nilai mendekati 1 berarti dua variabel cenderung naik bersamaan; mendekati 0 berarti hampir tidak berhubungan."
    )

# -----------------------------------------------------------------------------
# Footer identitas
# -----------------------------------------------------------------------------
st.divider()
st.caption(f"Studi Kasus 1 - Dikerjakan oleh {NAMA} (NIM {NIM})")
