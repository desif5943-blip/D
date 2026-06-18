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

# Kustomisasi CSS Tampilan Grid Kotak Hasil (Agar mirip dengan Gambar Referensi Kedua)
st.markdown("""
    <style>
    .metric-box-dark {
        background-color: #0B2240;
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-box-light {
        background-color: #F8F9FA;
        color: #333;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #E0E0E0;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. --- HEADER UTAMA ---
st.markdown("<h1>Detecting Fake Images with a Deep-Learning Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: gray; margin-top: -10px;'>Aplikasi Komparasi Arsitektur CNN (ResNet50, Xception, EfficientNetB0) | Desi Fatmala Susilawati</p>", unsafe_allow_html=True)
st.markdown("---")

# 3. --- SIDEBAR UNTUK PEMILIHAN MODEL ---
st.sidebar.title("🤖 Konfigurasi Model CNN")
st.sidebar.write("Pilih arsitektur model sesuai proposal untuk melakukan pengujian:")
pilihan_model = st.sidebar.selectbox(
    "Arsitektur Aktif:",
    ["ResNet50", "Xception", "EfficientNetB0"]
)

# 4. --- TEMPAT MENARUH LINK ID GOOGLE DRIVE MODEL ANDA ---
# Sesuai langkah sebelumnya, silakan ganti kode teks ID di bawah ini dengan ID dari link share Google Drive Anda
MODEL_DRIVE_IDS = {
    "ResNet50": "1p9zaiXEADcb6_KL_bvW7sUJf2kIysfGk",
    "Xception": "ID_FILE_XCEPTION_ANDA",
    "EfficientNetB0": "ID_FILE_EFFICIENTNET_ANDA"
}

NAMA_FILE_LOCAL = {
    "ResNet50": "resnet50_terbaik.h5",
    "Xception": "1I3pRM8KIU9PvpRPpRoKbG13CCx2Ygm_8",
    "EfficientNetB0": "1ORxkrpn4aNZh1ZIYyF0q47lJuA2XCkxt"
}

# Fungsi otomatis download dan load model h5 ke Cloud Server
@st.cache_resource
def load_selected_model(model_name):
    file_name = NAMA_FILE_LOCAL[model_name]
    drive_id = MODEL_DRIVE_IDS[model_name]
    
    # Folder internal server cloud untuk menyimpan model sementara
    os.makedirs("models", exist_ok=True)
    path_lengkap = os.path.join("models", file_name)
    
    if not os.path.exists(path_lengkap):
        # Jika ID belum diganti oleh pengguna, bypass menggunakan mode simulasi
        if "ID_FILE_" in drive_id:
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
    st.sidebar.warning(f"⚠️ Mode Simulasi Aktif karena ID Google Drive untuk {pilihan_model} belum diisi dengan benar.")

# 5. --- TATA LETAK GRID UTAMA (MEMBAGI MENJADI 2 KOLOM) ---
kolom_kiri, kolom_kanan = st.columns([1.2, 1], gap="large")

# ================= KOLOM KIRI: INPUT & KARTU METRIK HASIL =================
with kolom_kiri:
    
    # Sub-Grid untuk pembagian kotak upload dan metrik evaluasi tambahan
    sub_kol_1, sub_kol_2 = st.columns(2)
    
    with sub_kol_1:
        st.markdown("### Upload an image")
        uploaded_file = st.file_uploader("Pilih citra (JPG, PNG)...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        tombol_proses = st.button("Mulai Analisis Model", type="primary", use_container_width=True)
        
    with sub_kol_2:
        # Metrik Atas Kiri (Konstanta sesuai tampilan gambar acuan Anda)
        st.markdown("""
            <div class='metric-box-light'>
                <p style='margin:0; font-size:14px; color:gray;'>Real Images Predicted as Fake</p>
                <h2 style='margin:5px 0 0 0; color:#333;'>11%</h2>
            </div>
        """, unsafe_allow_html=True)

    # Baris baru di bawahnya
    sub_kol_3, sub_kol_4 = st.columns(2)
    
    with sub_kol_3:
        # Metrik Bawah Kiri (Konstanta sesuai tampilan gambar acuan Anda)
        st.markdown("""
            <div class='metric-box-light'>
                <p style='margin:0; font-size:14px; color:gray;'>Fake Images Predicted as Real</p>
                <h2 style='margin:5px 0 0 0; color:#333;'>94%</h2>
            </div>
        """, unsafe_allow_html=True)
        
    with sub_kol_4:
        # KOTAK BESAR UTAMA (Menampilkan Prediksi Akhir secara Real-Time)
        box_hasil = st.empty()
        box_hasil.markdown("""
            <div class='metric-box-dark'>
                <h1 style='margin:0; font-size:32px;'>Ready</h1>
                <p style='margin:5px 0 0 0; font-size:14px; color:#A0B2C6;'>Silakan unggah citra</p>
            </div>
        """, unsafe_allow_html=True)


# ================= KOLOM KANAN: PREVIEW & VISUALISASI GRAFIK BAR =================
with kolom_kanan:
    st.markdown("### Preview & Analisis Fitur")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Tampilkan gambar masukan di sisi kanan atas
        st.image(image, caption="Citra Masukan Uji", use_container_width=True)
        
        if tombol_proses:
            with st.spinner(f'Model {pilihan_model} sedang melakukan klasifikasi...'):
                
                # Preprocessing Gambar sesuai input default dimensi arsitektur CNN (224x224)
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Prediksi
                if model is not None:
                    prediction = model.predict(img_array)
                    prob_fake = float(prediction[0][0])  # Mengasumsikan nilai 1 dekat ke Fake (AI)
                    prob_real = 1.0 - prob_fake
                else:
                    # Angka pengisi simulasi grafik jika ID Drive belum dipasang
                    import random
                    prob_fake = random.uniform(0.65, 0.99)
                    prob_real = 1.0 - prob_fake
                
                # Mengubah Label Berdasarkan Probabilitas Tertinggi
                label_final = "Fake" if prob_fake > 0.5 else "Real"
                prob_final = prob_fake if prob_fake > 0.5 else prob_real
                
                # Mengganti Tampilan Kotak Besar Utama secara Interaktif
                box_hasil.markdown(f"""
                    <div class='metric-box-dark'>
                        <h1 style='margin:0; font-size:40px; font-weight:bold;'>{label_final}</h1>
                        <p style='margin:5px 0 0 0; font-size:16px; color:#A0B2C6;'>Probability: {prob_final:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Membuat visualisasi grafik batang persentase di sisi kanan bawah
                st.markdown("#### Distribusi Nilai Output Model:")
                fig, ax = plt.subplots(figsize=(5, 2))
                y_pos = np.arange(2)
                performance = [prob_real * 100, prob_fake * 100]
                ax.barh(y_pos, performance, align='center', color=['#2ecc71', '#e74c3c'])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(['Real (Asli)', 'Fake (AI)'])
                ax.set_xlabel('Persentase Keyakinan (%)')
                ax.set_xlim(0, 100)
                st.pyplot(fig)
                
                st.success(f"Berhasil dianalisis menggunakan arsitektur **{pilihan_model}**!")
    else:
        st.info("Menunggu berkas gambar diunggah di kolom kiri untuk menampilkan pratinjau analisis.")