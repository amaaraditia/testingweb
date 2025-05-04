import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --- Header dengan Menu Akun di Kanan Atas ---
col_left, col_spacer, col_right = st.columns([4, 6, 2])
with col_right:
    akun = st.selectbox("👤 Akun", ["Admin", "Logout"], label_visibility="collapsed")
    if akun == "Logout":
        st.warning("Anda telah logout.")
        st.stop()

# --- Navigasi Menu di Sidebar ---
menu = st.sidebar.selectbox("Navigasi", ["Dashboard", "Prediksi", "Visualisasi"])
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Exit"):
    st.sidebar.warning("Aplikasi ditutup.")
    st.stop()

# --- Fungsi: Dashboard ---
def show_dashboard():
    st.title("Dashboard Prediksi Cuaca")
    st.markdown("""
    Selamat datang di sistem prediksi kelas cuaca berbasis model *Random Forest*.

    Silakan pilih menu di sidebar untuk:
    - Melakukan **Prediksi** berdasarkan input parameter iklim.
    - Melihat **Visualisasi** data atau hasil prediksi.
    """)

    try:
        df = pd.read_csv("Dataset Tren Cuaca.csv")
    except FileNotFoundError:
        st.error("Dataset 'Dataset Tren Cuaca.csv' tidak ditemukan.")
        return

    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['month'] = df['date'].dt.month

    st.subheader("Visualisasi Tren Cuaca")

    parameter_options = [
        "temp_average", "temp_max", "curah_hujan",
        "penyinaran_matahari", "tekanan_udara", "kelembaban_average",
        "kec_angin_average", "arah_angin_most",
        "arah_angin", "weather_encoded"
    ]

    parameter = st.selectbox("Pilih Parameter Cuaca", parameter_options)

    if 'weather' not in df.columns and 'weather_encoded' in df.columns:
        df['weather'] = df['weather_encoded'].map({
            0: 'CLOUDINESS', 1: 'RAIN', 2: 'THUNDERSTORM'
        })

    fig, ax = plt.subplots(figsize=(10, 6))

    if parameter == "weather_encoded":
        weather_monthly_dist = df.groupby(['month', 'weather_encoded']).size().unstack(fill_value=0)
        weather_monthly_dist.plot(kind='bar', stacked=True, ax=ax, color=['tab:blue', 'tab:orange', 'tab:green'])
        ax.set_title("Distribusi Weather Encoded per Bulan")
        ax.set_xlabel("Bulan")
        ax.set_ylabel("Frekuensi Kejadian")
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)

        st.info("""
        **Keterangan Weather Encoded:**
        - 0 = Cloudiness
        - 1 = Rain
        - 2 = Thunderstorm
        """)
    else:
        weather_classes = df['weather'].unique()
        colors = plt.cm.tab20.colors
        for i, weather in enumerate(sorted(weather_classes)):
            subset = df[df['weather'] == weather]
            ax.hist(subset[parameter], bins=20, alpha=0.8, label=weather,
                    stacked=True, color=colors[i % len(colors)])
        ax.set_title(f"Distribusi {parameter.replace('_', ' ').title()} berdasarkan Kelas Cuaca")
        ax.set_xlabel(parameter.replace('_', ' ').title())
        ax.set_ylabel("Frekuensi")
        ax.legend(title="Weather", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)

    st.pyplot(fig)
    st.subheader(f"Tabel Data Tren {parameter.replace('_', ' ').title()}")
    st.dataframe(df[['date', parameter]].sort_values(by='date').reset_index(drop=True))

# --- Fungsi: Prediksi ---
def show_prediction():
    st.title("Prediksi Kelas Cuaca")
    st.markdown("Masukkan data iklim untuk memprediksi kelas cuaca:")

    try:
        model = joblib.load("random_forest_model.pkl")
    except FileNotFoundError:
        st.error("Model 'random_forest_model.pkl' tidak ditemukan.")
        return

    # Form input
    temp_average = st.number_input("Suhu Rata-rata (°C)")
    temp_max = st.number_input("Suhu Maksimum (°C)")
    curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0.0)
    penyinaran_matahari = st.number_input("Penyinaran Matahari (%)")
    kelembaban_average = st.number_input("Kelembaban Rata-rata (%)")
    kec_angin_average = st.number_input("Kecepatan Angin Rata-rata (m/s)")
    arah_angin_most = st.number_input("Arah Angin Paling Sering (°)")
    arah_angin = st.number_input("Arah Angin Saat Ini (°)")

    if st.button("Prediksi Kelas Cuaca"):
        input_data = np.array([
            temp_average, temp_max, curah_hujan, penyinaran_matahari,
            kelembaban_average, kec_angin_average, arah_angin_most, arah_angin
        ]).reshape(1, -1)

        prediction = model.predict(input_data)[0]

        # Mapping kelas ke deskripsi
        kelas_dict = {
            0: "Cloudiness (Berawan)",
            1: "Rain (Hujan)",
            2: "Thunderstorm (Badai Petir)"
        }

        deskripsi_kelas = kelas_dict.get(prediction, "Tidak diketahui")
        st.success(f"Kelas cuaca yang diprediksi: **{prediction} - {deskripsi_kelas}**")

        st.caption("Keterangan:\n0 = Cloudiness (Berawan)\n1 = Rain (Hujan)\n2 = Thunderstorm (Badai Petir)")

# --- Fungsi: Visualisasi Evaluasi Model ---
def show_visualization():
    st.title("Visualisasi Evaluasi Model & Korelasi")

    st.subheader("1. Perbandingan Kinerja Model")
    try:
        st.image(Image.open("Perbandingan Model.png"),
                 caption="Perbandingan Akurasi, Presisi, Recall, dan F1-Score 6 Model",
                 use_container_width=True)
    except:
        st.warning("Gambar 'Perbandingan Model.png' tidak ditemukan.")

    st.subheader("2. Confusion Matrix")
    col1, col2 = st.columns(2)
    with col1:
        for name in ["CM NN.png", "CM SVM.png", "CM NB.png"]:
            try:
                st.image(Image.open(name), caption=name.split('.')[0], use_container_width=True)
            except:
                st.warning(f"Gambar '{name}' tidak ditemukan.")
    with col2:
        for name in ["CM KNN.png", "CM RF.png", "CM DT.png"]:
            try:
                st.image(Image.open(name), caption=name.split('.')[0], use_container_width=True)
            except:
                st.warning(f"Gambar '{name}' tidak ditemukan.")

    st.subheader("3. Korelasi Parameter Cuaca")
    for name in ["Korelasi Spearman.png", "Peringkat Korelasi.png"]:
        try:
            st.image(Image.open(name), caption=name.replace(".png", "").replace("_", " "), use_container_width=True)
        except:
            st.warning(f"Gambar '{name}' tidak ditemukan.")

# --- Pemanggilan Menu ---
if menu == "Dashboard":
    show_dashboard()
elif menu == "Prediksi":
    show_prediction()
elif menu == "Visualisasi":
    show_visualization()
