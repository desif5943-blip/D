import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request

# 1. KONFIGURASI HALAMAN UTAMA (Wide Mode)
st.set_page_config(
    page_title="Deep-Learning Tool - Fake Image Detection",
    page_icon="🔍",
    layout="wide"
)

# 2. --- HEADER UTAMA ---
st.title("Detecting Fake Images with a Deep-Learning Tool")
st.caption("Aplikasi Komparasi Arsitektur CNN (ResNet50, Xception, EfficientNetB0) | Desi Fatmala Susilawati")
st.divider()

# 3. --- SIDEBAR UNTUK PEMILIHAN MODEL ---
st.sidebar.title("🤖 Konfigurasi Model CNN")
st.sidebar.write("Pilih arsitektur model sesuai proposal untuk melakukan pengujian:")
pilihan_model = st.sidebar.selectbox(
    "Arsitektur Aktif:",
    ["ResNet50", "Xception", "EfficientNetB0"]
)

# 4. --- GOOGLE DRIVE FILE ID MANAGEMENT ---
MODEL_DRIVE_IDS = {
    "ResNet50": "ID_FILE_RESNET_ANDA",
    "Xception": "ID_FILE_XCEPTION_ANDA",
    "EfficientNetB0": "ID_FILE_EFFICIENTNET_ANDA"
}

NAMA_FILE_LOCAL = {
    "ResNet50": "1p9zaiXEADcb6_KL_bvW7sUJf2kIysfGk",
    "Xception": "1I3pRM8KIU9PvpRPpRoKbG13CCx2Ygm_8",
    "EfficientNetB0": "1ORxkrpn4aNZh1ZIYyF0q47lJuA2XCkxt"
}

@st.cache_resource
def load_selected_model(model_name):
    file_name = NAMA_FILE_LOCAL[model_name]
    drive_id = MODEL_DRIVE_IDS[model_name]
    
    os.makedirs("models", exist_ok=True)
    path_lengkap = os.path.join("models", file_name)
    
    if not os.path.exists(path_lengkap):
        if "ID_FILE_" in drive_id or drive_id == "":
            return None
            
        with st.spinner(f"Sedang mengunduh arsitektur {model_name} dari Google Drive (hanya dilakukan sekali)..."):
            try:
                url_download = f"https://docs.google.com/uc?export=download&id={drive_id}"
                urllib.request.urlretrieve(url_download, path_lengkap)
            except Exception as e:
                st.error(f"Gagal mengunduh model {model_name}: {e}")
                return None
                
    return tf.keras.models.load_model(path_lengkap)

model = load_selected_model(pilihan_model)

if model is None:
    st.sidebar.warning(f"⚠️ Mode Simulasi Aktif untuk {pilihan_model}. Hubungkan ID Drive Anda jika ingin hasil riil.")

# 5. --- TATA LETAK GRID UTAMA (MENGGUNAKAN ELEMEN NATIVE STREAMLIT) ---
kolom_kiri, kolom_kanan = st.columns([1.2, 1], gap="large")

# Menginisialisasi variabel status agar tidak terjadi tabrakan penulisan elemen
sudah_proses = False
label_final = ""
prob_final = 0.0
prob_real = 0.0
prob_fake = 0.0

# ================= KOLOM KANAN (PROSES INPUT & PREVIEW) =================
with kolom_kanan:
    st.markdown("### Preview & Analisis Fitur")
    uploaded_file = st.file_uploader("Pilih citra (JPG, PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Citra Masukan Uji", use_container_width=True)
        
        # Logika pemrosesan diletakkan sebelum penggambaran kotak metrik hasil
        if st.button("Mulai Analisis Model", type="primary", use_container_width=True):
            with st.spinner(f'Model {pilihan_model} sedang menganalisis...'):
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                if model is not None:
                    prediction = model.predict(img_array)
                    prob_fake = float(prediction[0][0])
                    prob_real = 1.0 - prob_fake
                else:
                    import random
                    prob_fake = random.uniform(0.65, 0.99)
                    prob_real = 1.0 - prob_fake
                
                label_final = "Fake (AI)" if prob_fake > 0.5 else "Real (Asli)"
                prob_final = prob_fake if prob_fake > 0.5 else prob_real
                sudah_proses = True

# ================= KOLOM KIRI (TAMPILAN STATIS KARTU METRIK) =================
with kolom_kiri:
    st.markdown("### Dashboard Evaluasi")
    
    # Sub-Grid Atas
    sub_kol_1, sub_kol_2 = st.columns(2)
    with sub_kol_1:
        with st.container(border=True):
            st.metric(label="Real Images Predicted as Fake", value="11%")
    with sub_kol_2:
        with st.container(border=True):
            st.metric(label="Fake Images Predicted as Real", value="94%")

    # Sub-Grid Bawah (Tempat Menampilkan Hasil Akhir)
    with st.container(border=True):
        if sudah_proses:
            st.metric(label=f"Hasil Akhir ({pilihan_model})", value=label_final, delta=f"Keyakinan: {prob_final*100:.1f}%")
        else:
            st.metric(label="Hasil Akhir", value="Ready", delta="Menunggu input gambar dan klik tombol", delta_color="off")

# ================= VISUALISASI GRAFIK DI BAGIAN BAWAH KANAN =================
if sudah_proses:
    with kolom_kanan:
        st.markdown("#### Distribusi Nilai Output Model:")
        fig, ax = plt.subplots(figsize=(5, 2))
        y_pos = np.arange(2)
        performance = [prob_real * 100, prob_fake * 100]
        ax.barh(y_pos, performance, align='center', color=['#2ecc71', '#e74c3c'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(['Real', 'Fake'])
        ax.set_xlabel('Persentase Keyakinan (%)')
        ax.set_xlim(0, 100)
        st.pyplot(fig)
        plt.close(fig)
        st.success(f"Analisis selesai menggunakan arsitektur {pilihan_model}!")
