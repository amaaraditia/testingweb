import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load model
loaded_model = joblib.load("/full/path/to/random_forest_model.pkl")

# Sidebar menu
menu = st.sidebar.selectbox("Navigasi", ["Dashboard", "Prediksi", "Visualisasi"])

if menu == "Dashboard":
    st.title("Dashboard Prediksi Cuaca")
    st.markdown("""
    Selamat datang di sistem prediksi kelas cuaca berbasis model *Random Forest*.

    Silakan pilih menu di sidebar untuk:
    - Melakukan **Prediksi** berdasarkan input parameter iklim.
    - Melihat **Visualisasi** data atau hasil prediksi.
    """)

    # Load dataset tren cuaca
    df = pd.read_csv("Dataset Tren Cuaca.csv")

    # Ubah kolom 'date' menjadi format datetime
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

    # Tambahkan kolom bulan
    df['month'] = df['date'].dt.month

    st.subheader("Visualisasi Tren Cuaca")

    # Daftar parameter yang bisa divisualisasikan
    parameter_options = [
        "temp_average",
        "temp_max",
        "temp_min",
        "curah_hujan",
        "penyinaran_matahari",
        "tekanan_udara",
        "kelembaban_average",
        "kec_angin_average",
        "kec_angin_high",
        "arah_angin_most",
        "arah_angin",
        "weather_encoded"
    ]

    # Dropdown untuk memilih parameter
    parameter = st.selectbox("Pilih Parameter Cuaca untuk Ditampilkan", parameter_options)

    # Cek apakah kolom 'weather' tersedia, jika tidak map dari 'weather_encoded'
    if 'weather' not in df.columns and 'weather_encoded' in df.columns:
        df['weather'] = df['weather_encoded'].map({
            0: 'CLOUDINESS',
            1: 'RAIN',
            2: 'THUNDERSTORM'
        })

    # Visualisasi data
    fig, ax = plt.subplots(figsize=(10, 6))

    if parameter == "weather_encoded":
        # Hitung distribusi kategori weather_encoded berdasarkan bulan
        weather_monthly_dist = df.groupby(['month', 'weather_encoded']).size().unstack(fill_value=0)

        # Visualisasi distribusi kategori weather_encoded berdasarkan bulan
        weather_monthly_dist.plot(kind='bar', stacked=True, ax=ax, color=['tab:blue', 'tab:orange', 'tab:green'])

        ax.set_title("Distribusi Weather Encoded per Bulan")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Frekuensi Kejadian")
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)

        st.info("""
        **Keterangan Weather Encoded:**
        - 0 = Cloudiness
        - 1 = Rain
        - 2 = Thunderstorm
        """)

    else:
        # Visualisasi histogram stacked berdasarkan kelas cuaca
        weather_classes = df['weather'].unique()
        colors = plt.cm.tab20.colors  # Maksimum 20 warna

        for i, weather in enumerate(sorted(weather_classes)):
            subset = df[df['weather'] == weather]
            ax.hist(subset[parameter], bins=20, alpha=0.8, label=weather, stacked=True, color=colors[i % len(colors)])

        ax.set_title(f"Distribusi {parameter.replace('_', ' ').title()} berdasarkan Kelas Cuaca")
        ax.set_xlabel(parameter.replace('_', ' ').title())
        ax.set_ylabel("Frekuensi")
        ax.legend(title="Weather", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

    st.pyplot(fig)

    # Tampilkan tabel data tren parameter
    st.subheader(f"Tabel Data Tren {parameter.replace('_', ' ').title()}")
    st.dataframe(df[['date', parameter]].sort_values(by='date').reset_index(drop=True))


# Visualisasi
elif menu == "Visualisasi":
    st.title("Visualisasi Hasil Evaluasi Model dan Korelasi Cuaca")

    st.subheader("1. Perbandingan Kinerja Semua MOdel")
    image_nn = Image.open("Perbandingan Model.png")
    st.image(image_nn, caption="Perbandingan Akurasi, Presisi, Recall, dan F1-Score 6 Model", use_container_width=True)

    st.subheader("2. Confusion Matrix dari Masing-masing Model")
    col1, col2 = st.columns(2)

    with col1:
        st.image(Image.open("CM NN.png"), caption="Confusion Matrix - Neural Network", use_container_width=True)
        st.image(Image.open("CM SVM.png"), caption="Confusion Matrix - SVM", use_container_width=True)
        st.image(Image.open("CM NB.png"), caption="Confusion Matrix - Naive Bayes", use_container_width=True)

    with col2:
        st.image(Image.open("CM KNN.png"), caption="Confusion Matrix - KNN", use_container_width=True)
        st.image(Image.open("CM RF.png"), caption="Confusion Matrix - Random Forest", use_container_width=True)
        st.image(Image.open("CM DT.png"), caption="Confusion Matrix - Decision Tree", use_container_width=True)

    st.subheader("3. Analisis Korelasi Antara Parameter Cuaca")
    st.image(Image.open("Korelasi Spearman.png"), caption="Heatmap Korelasi Spearman antar Parameter Cuaca", use_container_width=True)
    st.image(Image.open("Peringkat Korelasi.png"), caption="Peringkat Pengaruh Parameter terhadap Target", use_container_width=True)
