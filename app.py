"""
============================================================
APLIKASI ANALISIS SENTIMEN ULASAN PINJAMAN ONLINE (VERSI ONLINE)
Aspect-Based Sentiment Analysis (ABSA)
Model: Fine-Tuned IndoBERT (di-load dari Hugging Face Hub)
============================================================

CARA DEPLOY:
1. Upload model ke Hugging Face Hub dulu (pakai upload_to_huggingface.py)
2. Ganti HF_MODEL_REPO di bawah dengan repo_id model-mu
3. Push app.py + requirements.txt + dataset_pinjol_with_aspects.csv ke GitHub
4. Deploy di share.streamlit.io, hubungkan ke repo GitHub tersebut
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import re
import string
from transformers import AutoTokenizer, BertForSequenceClassification

# ============================================================
# KONFIGURASI - GANTI INI DENGAN REPO HUGGING FACE-MU
# ============================================================
HF_MODEL_REPO = "hanwho/indobert-pinjol-sentiment"  # <-- GANTI!

st.set_page_config(
    page_title="Analisis Sentimen Pinjol",
    page_icon=":bar_chart:",
    layout="wide"
)

# ============================================================
# LOAD MODEL DARI HUGGING FACE HUB (sekali saja, di-cache)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Memuat model dari Hugging Face Hub. Di-cache supaya hanya download/load sekali."""
    # torch.set_num_threads(1) membantu hemat memori di server gratis
    torch.set_num_threads(1)

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
    model = BertForSequenceClassification.from_pretrained(
        HF_MODEL_REPO,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True  # penting untuk hemat RAM saat loading
    )
    model.eval()

    # Label encoder: karena hanya 2 kelas (Positif/Negatif), didefinisikan manual
    # supaya tidak perlu upload file .pkl terpisah (lebih simpel untuk online)
    label_map = {0: "Negatif", 1: "Positif"}  # sesuaikan urutan dengan le.classes_ aslimu!

    return tokenizer, model, label_map

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("dataset_pinjol_with_aspects.csv")
    except FileNotFoundError:
        return None

# ============================================================
# PREPROCESSING & PREDIKSI
# ============================================================
def preprocess_ringkas(teks):
    if not isinstance(teks, str):
        return ""
    teks = teks.lower()
    teks = re.sub(r"http\S+|www\S+|https\S+", "", teks)
    teks = re.sub(r"@\w+|#\w+", "", teks)
    teks = teks.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    teks = re.sub(r"\d+", "", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks

def prediksi_sentimen(teks, tokenizer, model, label_map, max_len=128):
    teks_bersih = preprocess_ringkas(teks)
    inputs = tokenizer(
        teks_bersih, padding="max_length", truncation=True,
        max_length=max_len, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    label = label_map[pred_idx]
    return label, confidence

# ============================================================
# HEADER
# ============================================================
st.title("Analisis Sentimen Ulasan Aplikasi Pinjaman Online")
st.markdown("**Aspect-Based Sentiment Analysis menggunakan Fine-Tuned IndoBERT**")
st.caption("Model di-load dari Hugging Face Hub. Pemuatan pertama mungkin butuh waktu lebih lama.")
st.markdown("---")

menu = st.sidebar.radio(
    "Pilih Menu:",
    ["Cek Satu Ulasan", "Analisis Banyak Ulasan (CSV)", "Dashboard Dataset"]
)

# Load model dengan penanganan error yang jelas (penting untuk online, RAM terbatas)
model_ready = False
if menu != "Dashboard Dataset":
    with st.spinner("Memuat model IndoBERT dari Hugging Face Hub... (mohon tunggu, ~30-60 detik pertama kali)"):
        try:
            tokenizer, model, label_map = load_model()
            model_ready = True
        except Exception as e:
            st.error(
                "Gagal memuat model. Kemungkinan penyebab: "
                "(1) REPO_ID Hugging Face salah, (2) server kehabisan memori (RAM), "
                "atau (3) koneksi terputus saat download model. "
                f"Detail error: {e}"
            )
            st.info("Coba refresh halaman. Jika tetap gagal, server mungkin kehabisan RAM "
                    "saat memuat model berukuran besar.")

# ============================================================
# MENU 1: CEK SATU ULASAN
# ============================================================
if menu == "Cek Satu Ulasan":
    st.header("Cek Sentimen Satu Ulasan")
    st.write("Ketik atau tempel ulasan pengguna, lalu klik tombol analisis.")

    contoh = st.selectbox(
        "Atau pilih contoh ulasan:",
        ["(ketik sendiri)",
         "Proses cepat banget, langsung cair dalam 5 menit. Sangat membantu!",
         "Bunganya kemahalan, penagihan kasar banget sampai teror keluarga.",
         "Aplikasi sering error, login gagal terus, data saya juga diminta macam-macam."]
    )

    teks_input = st.text_area(
        "Ulasan:",
        value="" if contoh == "(ketik sendiri)" else contoh,
        height=120
    )

    if st.button("Analisis Sentimen", type="primary"):
        if not model_ready:
            st.warning("Model belum siap. Coba refresh halaman.")
        elif teks_input.strip() == "":
            st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
        else:
            label, conf = prediksi_sentimen(teks_input, tokenizer, model, label_map)
            col1, col2 = st.columns(2)
            with col1:
                if label.lower() == "positif":
                    st.success(f"Sentimen: **{label.upper()}**")
                else:
                    st.error(f"Sentimen: **{label.upper()}**")
            with col2:
                st.metric("Tingkat Keyakinan Model", f"{conf*100:.1f}%")

# ============================================================
# MENU 2: ANALISIS BANYAK ULASAN (CSV)
# ============================================================
elif menu == "Analisis Banyak Ulasan (CSV)":
    st.header("Analisis Banyak Ulasan Sekaligus")
    st.warning("Untuk versi online, disarankan upload maksimal 200-300 baris agar tidak timeout.")
    uploaded = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded is not None and model_ready:
        df_upload = pd.read_csv(uploaded)
        st.write("Pratinjau data:")
        st.dataframe(df_upload.head())

        kolom_teks = st.selectbox("Pilih kolom yang berisi teks ulasan:", df_upload.columns)

        if st.button("Proses Semua Ulasan", type="primary"):
            if len(df_upload) > 500:
                st.error("Terlalu banyak baris untuk versi online (maks 500). Gunakan versi offline untuk dataset besar.")
            else:
                progress = st.progress(0)
                hasil_label, hasil_conf = [], []
                total = len(df_upload)

                for i, teks in enumerate(df_upload[kolom_teks].astype(str)):
                    label, conf = prediksi_sentimen(teks, tokenizer, model, label_map)
                    hasil_label.append(label)
                    hasil_conf.append(round(conf * 100, 1))
                    progress.progress((i + 1) / total)

                df_upload["Prediksi_Sentimen"] = hasil_label
                df_upload["Keyakinan_%"] = hasil_conf

                st.success(f"Selesai menganalisis {total} ulasan!")
                st.dataframe(df_upload)

                col1, col2 = st.columns(2)
                with col1:
                    counts = df_upload["Prediksi_Sentimen"].value_counts()
                    st.bar_chart(counts)
                with col2:
                    for lbl, cnt in counts.items():
                        st.write(f"**{lbl}**: {cnt} ulasan ({cnt/total*100:.1f}%)")

                csv_hasil = df_upload.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Hasil (CSV)", csv_hasil,
                    "hasil_analisis_sentimen.csv", "text/csv"
                )

# ============================================================
# MENU 3: DASHBOARD DATASET (tidak butuh model, jadi selalu ringan)
# ============================================================
elif menu == "Dashboard Dataset":
    st.header("Dashboard Analisis Dataset")
    df = load_dataset()

    if df is None:
        st.warning("File 'dataset_pinjol_with_aspects.csv' tidak ditemukan di repository.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Ulasan", f"{len(df):,}")
        if "platform" in df.columns:
            col2.metric("Jumlah Aplikasi", df["platform"].nunique())
        if "sentiment" in df.columns:
            neg_pct = (df["sentiment"].str.lower() == "negatif").mean() * 100
            col3.metric("Proporsi Negatif", f"{neg_pct:.1f}%")

        st.markdown("---")

        if "sentiment" in df.columns:
            st.subheader("Distribusi Sentimen")
            st.bar_chart(df["sentiment"].value_counts())

        if "platform" in df.columns and "sentiment" in df.columns:
            st.subheader("Sentimen per Aplikasi")
            st.bar_chart(pd.crosstab(df["platform"], df["sentiment"]))

        if "aspek" in df.columns:
            st.subheader("Distribusi Aspek Keluhan")
            st.bar_chart(df["aspek"].value_counts())

        if "aspek" in df.columns and "sentiment" in df.columns:
            st.subheader("Tabel Aspek vs Sentimen")
            st.dataframe(pd.crosstab(df["aspek"], df["sentiment"], margins=True))

st.sidebar.markdown("---")
st.sidebar.caption("Model: Fine-Tuned IndoBERT | Akurasi 96,16%")
