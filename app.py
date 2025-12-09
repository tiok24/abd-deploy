# app.py
# ============================================
# Aplikasi Analisis Skripsi Hukum (Streamlit)
# ============================================

import streamlit as st
import pandas as pd
import ast
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# ------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------
st.set_page_config(
    page_title="Analisis Skripsi Hukum",
    layout="wide"
)

# ------------------------------------------------
# Fungsi utilitas
# ------------------------------------------------

def parse_tokens(x):
    """Konversi kolom tokens (string) menjadi list Python."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x)
    try:
        if s.strip().startswith("[") and s.strip().endswith("]"):
            val = ast.literal_eval(s)
            if isinstance(val, list):
                return [str(t) for t in val]
    except Exception:
        pass
    return s.split()


@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)

    # normalisasi kolom prodi (kalau ada)
    if "prodi" in df.columns:
        df["prodi"] = (
            df["prodi"]
            .astype(str)
            .str.strip()
            .replace({"nan": None})
        )

    # pastikan tokens_parsed ada
    if "tokens" in df.columns:
        df["tokens_parsed"] = df["tokens"].apply(parse_tokens)
    else:
        df["tokens_parsed"] = [[] for _ in range(len(df))]

    return df


def get_top_terms(df_sub, n=10):
    """Ambil top-n kata dari kolom tokens_parsed."""
    counter = Counter()
    for toks in df_sub["tokens_parsed"]:
        counter.update(toks)
    return counter.most_common(n)


def plot_wordcloud(df_sub, title="Wordcloud"):
    """Buat wordcloud dari subset dokumen."""
    tokens = []
    for toks in df_sub["tokens_parsed"]:
        tokens.extend(toks)

    if not tokens:
        st.warning("Tidak ada token untuk dibuat wordcloud.")
        return

    text = " ".join(tokens)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.subheader(title)
    st.pyplot(fig)


def apply_filters(df):
    """Terapkan filter berdasarkan sidebar."""
    df_filtered = df.copy()

    # ---------- FILTER PRODI ----------
    if "prodi" in df.columns:
        all_prodi = (
            df["prodi"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )
        all_prodi = sorted(all_prodi)
        selected_prodi = st.sidebar.multiselect("Program Studi", all_prodi)
        if selected_prodi:
            df_filtered = df_filtered[df_filtered["prodi"].isin(selected_prodi)]

    # ---------- FILTER TAHUN ----------
    if "tahun" in df_filtered.columns:
        all_years = (
            df_filtered["tahun"]
            .dropna()
            .astype(str)
            .unique()
        )
        all_years = sorted(all_years)
        selected_years = st.sidebar.multiselect("Tahun", all_years)
        if selected_years:
            df_filtered = df_filtered[
                df_filtered["tahun"].astype(str).isin(selected_years)
            ]

    # ---------- FILTER PENULIS ----------
    if "penulis" in df_filtered.columns:
        all_authors = (
            df_filtered["penulis"]
            .dropna()
            .astype(str)
            .unique()
        )
        all_authors = sorted(all_authors)
        selected_authors = st.sidebar.multiselect("Penulis", all_authors)
        if selected_authors:
            df_filtered = df_filtered[df_filtered["penulis"].isin(selected_authors)]

    # ---------- FILTER CLUSTER LDA ----------
    if "lda_cluster" in df_filtered.columns:
        lda_opts = sorted(df_filtered["lda_cluster"].dropna().unique())
        selected_lda = st.sidebar.multiselect("Cluster LDA", lda_opts)
        if selected_lda:
            df_filtered = df_filtered[df_filtered["lda_cluster"].isin(selected_lda)]

    # ---------- FILTER CLUSTER HDBSCAN ----------
    if "hdbscan_cluster" in df_filtered.columns:
        hdb_opts = sorted(df_filtered["hdbscan_cluster"].dropna().unique())
        selected_hdb = st.sidebar.multiselect("Cluster HDBSCAN", hdb_opts)
        if selected_hdb:
            df_filtered = df_filtered[df_filtered["hdbscan_cluster"].isin(selected_hdb)]

    # ---------- FILTER KATA KUNCI ----------
    keyword = st.sidebar.text_input("Filter Kata Kunci (judul/abstrak)")
    if keyword:
        kw = keyword.lower()
        mask = False
        if "judul" in df_filtered.columns:
            mask = df_filtered["judul"].str.lower().str.contains(kw, na=False)
        if "clean" in df_filtered.columns:
            mask = mask | df_filtered["clean"].str.lower().str.contains(kw, na=False)
        df_filtered = df_filtered[mask]

    return df_filtered


def show_topic_block(df_topic, topic_label):
    """Tampilkan ringkasan satu topik LDA."""
    st.markdown(f"### Topik {topic_label}")
    st.write(f"Jumlah dokumen: **{len(df_topic)}**")

    # Top kata
    st.markdown("**Top 10 kata dalam topik ini:**")
    top_terms = get_top_terms(df_topic, n=10)
    if top_terms:
        term_df = pd.DataFrame(top_terms, columns=["Kata", "Frekuensi"])
        st.table(term_df)
    else:
        st.info("Belum ada token yang bisa dihitung untuk topik ini.")

    # Judul
    st.markdown("**Daftar judul skripsi:**")
    if "judul" in df_topic.columns:
        for _, row in df_topic.iterrows():
            st.markdown(
                f"- **{row.get('judul', '(tanpa judul)')}**  "
                f"(_{row.get('penulis', '-')}_)"
            )
    else:
        st.info("Kolom 'judul' tidak ditemukan di dataset.")


# ------------------------------------------------
# Load data
# ------------------------------------------------

# GANTI path ini sesuai nama file di repo
DATA_PATH = "hasil_preprocessing_enriched.csv"

df = load_data(DATA_PATH)

# ------------------------------------------------
# Sidebar: navigasi & filter
# ------------------------------------------------
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman",
    (
        "Halaman Utama",
        "Document Filtering & Visualisasi",
        "Wordcloud & Top Terms per Cluster",
        "Halaman Analisis Topik (LDA)",
        "Ranking Dosen Pembimbing 1",
    )
)

st.sidebar.header("Filter Data")
df_filtered = apply_filters(df)

# ------------------------------------------------
# Halaman: Utama
# ------------------------------------------------
if page == "Halaman Utama":
    st.title("📘 Aplikasi Analisis Skripsi Hukum")

    st.markdown(
        """
        Aplikasi ini digunakan untuk mengeksplorasi kumpulan skripsi hukum.

        **Fitur utama:**
        - 🔍 *Document Filtering*: filter berdasarkan Program Studi, Tahun, Penulis, Kata Kunci, Cluster LDA & HDBSCAN  
        - 🗺️ *Visualisasi Interaktif*: UMAP + HDBSCAN  
        - ☁️ *Wordcloud per Cluster*: lihat kata dominan di tiap cluster/topik  
        - 🧠 *Analisis Topik (LDA)*: ringkasan topik beserta judul dokumen  
        - 🏅 *Ranking Dosen Pembimbing 1*: dosen yang paling banyak membimbing skripsi
        """
    )

    st.subheader("Preview Dataset")
    st.dataframe(df.head())


# ------------------------------------------------
# Halaman: Document Filtering & Visualisasi
# ------------------------------------------------
elif page == "Document Filtering & Visualisasi":
    st.title("🔍 Document Filtering & Visualisasi UMAP + HDBSCAN")

    st.write(f"Jumlah dokumen hasil filter: **{len(df_filtered)}**")

    cols_show = [c for c in ["penulis", "judul", "prodi", "tahun"] if c in df_filtered.columns]
    if cols_show:
        st.dataframe(df_filtered[cols_show].head(50))

    st.markdown("---")
    st.subheader("Visualisasi UMAP + HDBSCAN")

    required_cols = {"umap_x", "umap_y", "hdbscan_cluster"}
    if required_cols.issubset(df_filtered.columns):
        fig = px.scatter(
            df_filtered,
            x="umap_x",
            y="umap_y",
            color="hdbscan_cluster",
            hover_data=[c for c in ["judul", "penulis", "prodi", "tahun"] if c in df_filtered.columns],
            title="UMAP Embedding (diwarnai berdasarkan Cluster HDBSCAN)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        missing = required_cols - set(df_filtered.columns)
        st.warning(
            "Kolom berikut belum ada di dataset sehingga scatter UMAP "
            f"tidak bisa ditampilkan: {', '.join(missing)}.\n\n"
            "Pastikan file CSV sudah berisi hasil UMAP dan label cluster HDBSCAN."
        )


# ------------------------------------------------
# Halaman: Wordcloud & Top Terms per Cluster
# ------------------------------------------------
elif page == "Wordcloud & Top Terms per Cluster":
    st.title("☁️ Wordcloud & Top Terms per Cluster")

    st.markdown(
        """
        Pilih jenis cluster (HDBSCAN atau LDA), lalu pilih cluster tertentu untuk:
        - melihat wordcloud
        - top keyword
        - jumlah dokumen & judul-judulnya
        """
    )

    cluster_type = st.radio("Pilih jenis cluster:", ("HDBSCAN", "LDA"))

    cluster_col = "hdbscan_cluster" if cluster_type == "HDBSCAN" else "lda_cluster"

    if cluster_col not in df_filtered.columns:
        st.warning(f"Kolom '{cluster_col}' belum ada di dataset.")
    else:
        clusters = sorted(df_filtered[cluster_col].dropna().unique())
        if not clusters:
            st.warning("Tidak ada nilai cluster pada subset data ini.")
        else:
            selected_cluster = st.selectbox(
                f"Pilih cluster {cluster_type}:",
                clusters
            )

            df_cluster = df_filtered[df_filtered[cluster_col] == selected_cluster]

            st.markdown(
                f"### Ringkasan Cluster {cluster_type} = {selected_cluster}  \n"
                f"Jumlah dokumen: **{len(df_cluster)}**"
            )

            # Wordcloud
            plot_wordcloud(df_cluster, title=f"Wordcloud Cluster {cluster_type} = {selected_cluster}")

            # Top terms
            st.subheader("Top 10 kata di cluster ini")
            top_terms = get_top_terms(df_cluster, n=10)
            if top_terms:
                term_df = pd.DataFrame(top_terms, columns=["Kata", "Frekuensi"])
                st.table(term_df)
            else:
                st.info("Belum ada token yang bisa dihitung untuk cluster ini.")

            # List judul
            st.subheader("Daftar Judul dalam Cluster ini")
            if "judul" in df_cluster.columns:
                for _, row in df_cluster.iterrows():
                    st.markdown(
                        f"- **{row.get('judul', '(tanpa judul)')}**  "
                        f"(_{row.get('penulis', '-')}_)"
                    )
            else:
                st.info("Kolom 'judul' tidak ditemukan di dataset.")


# ------------------------------------------------
# Halaman: Analisis Topik (LDA)
# ------------------------------------------------
elif page == "Halaman Analisis Topik (LDA)":
    st.title("🧠 Analisis Topik (LDA)")

    if "lda_cluster" not in df_filtered.columns:
        st.warning(
            "Kolom 'lda_cluster' belum ada di dataset. "
            "Silakan tambahkan hasil pemodelan LDA (nomor topik per dokumen)."
        )
    else:
        st.markdown(
            """
            Halaman ini menampilkan:
            - daftar topik LDA
            - top 10 kata per topik
            - jumlah dokumen
            - daftar judul skripsi per topik
            """
        )

        topic_list = sorted(df_filtered["lda_cluster"].dropna().unique())
        st.markdown(f"**Jumlah topik LDA terdeteksi: {len(topic_list)}**")

        selected_topic = st.selectbox(
            "Pilih topik untuk difokuskan:",
            ["(Semua Topik)"] + list(topic_list)
        )

        if selected_topic == "(Semua Topik)":
            for topic in topic_list:
                df_topic = df_filtered[df_filtered["lda_cluster"] == topic]
                with st.expander(f"Topik {topic}  •  {len(df_topic)} dokumen"):
                    show_topic_block(df_topic, topic)
        else:
            df_topic = df_filtered[df_filtered["lda_cluster"] == selected_topic]
            show_topic_block(df_topic, selected_topic)


# ------------------------------------------------
# Halaman: Ranking Dosen Pembimbing 1
# ------------------------------------------------
elif page == "Ranking Dosen Pembimbing 1":
    st.title("🏅 Ranking Dosen Pembimbing 1")

    st.markdown(
        """
        Halaman ini menampilkan peringkat **dosen pembimbing 1** berdasarkan
        jumlah skripsi dalam dataset (setelah filter di sidebar diterapkan).
        """
    )

    # Coba deteksi nama kolom dospem 1
    possible_supervisor_cols = [
        "dospem_1",
        "pembimbing_pertama",
        "pembimbing_1",
        "pembimbing1"
    ]
    supervisor_col = None
    for c in possible_supervisor_cols:
        if c in df_filtered.columns:
            supervisor_col = c
            break

    if supervisor_col is None:
        st.error(
            "Kolom untuk dosen pembimbing 1 tidak ditemukan.\n\n"
            "Pastikan salah satu nama kolom berikut ada di CSV:\n"
            f"{', '.join(possible_supervisor_cols)}"
        )
    else:
        st.info(f"Menggunakan kolom dosen pembimbing 1: **{supervisor_col}**")

        df_sup = df_filtered.dropna(subset=[supervisor_col]).copy()

        if df_sup.empty:
            st.warning("Tidak ada data dosen pembimbing 1 di subset dokumen yang terfilter.")
        else:
            # Ranking dosen
            rank_df = (
                df_sup
                .groupby(supervisor_col)
                .agg(jumlah_skripsi=("judul", "count"))
                .sort_values("jumlah_skripsi", ascending=False)
                .reset_index()
            )

            st.subheader("Top Dosen Pembimbing 1 (berdasarkan jumlah skripsi)")
            st.dataframe(rank_df.head(20))

            # Bar chart top 10
            st.subheader("Visualisasi Top 10 Dosen Pembimbing 1")
            top_n = min(10, len(rank_df))
            rank_top = rank_df.head(top_n)

            fig = px.bar(
                rank_top,
                x="jumlah_skripsi",
                y=supervisor_col,
                orientation="h",
                title="Top 10 Dosen Pembimbing 1 berdasarkan jumlah skripsi",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

            # Detail per dosen
            st.subheader("Detail Skripsi per Dosen Pembimbing 1")
            selected_dosen = st.selectbox(
                "Pilih dosen pembimbing 1",
                rank_df[supervisor_col].tolist()
            )

            df_dosen = df_sup[df_sup[supervisor_col] == selected_dosen]

            st.markdown(
                f"**{selected_dosen}** membimbing **{len(df_dosen)}** skripsi "
                "pada subset data saat ini."
            )

            if "prodi" in df_dosen.columns:
                prodi_counts = df_dosen["prodi"].value_counts()
                st.markdown("Sebaran per Program Studi:")
                st.table(prodi_counts.to_frame("jumlah_skripsi"))

            st.markdown("### Daftar Judul Skripsi yang Dibimbing")
            cols_show = [c for c in ["judul", "penulis", "prodi", "tahun"] if c in df_dosen.columns]
            if cols_show:
                st.dataframe(df_dosen[cols_show])
            else:
                for _, row in df_dosen.iterrows():
                    st.markdown(
                        f"- **{row.get('judul', '(tanpa judul)')}** — "
                        f"_{row.get('penulis', '-')}_"
                    )
