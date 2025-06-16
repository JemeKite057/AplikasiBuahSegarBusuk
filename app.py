import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time 

st.set_page_config(page_title="Klasifikasi Buah Segar/Busuk", layout="centered")

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Peringatan: File CSS '{file_name}' tidak ditemukan. Menggunakan gaya default.")
    except Exception as e:
        st.warning(f"Peringatan: Gagal memuat file CSS. Error: {e}")

load_css('style.css')

@st.cache_resource
def load_fresh_rotten_model():
    model_path = 'fresh_rotten_classifier.h5'
    if not os.path.exists(model_path):
        st.error(f"Error: File model '{model_path}' tidak ditemukan. "
                 "Pastikan model 'fresh_rotten_classifier.h5' berada di direktori yang sama dengan skrip ini.")
        st.stop() 
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

model_fresh_rotten = load_fresh_rotten_model()

IMG_TARGET_SIZE = (100, 100) 
def load_image_for_prediction(img_file):
    img = image.load_img(img_file, target_size=IMG_TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch (misal: (1, 100, 100, 3))
    img_array = img_array / 255.0 # Normalisasi piksel ke rentang [0, 1]
    return img_array

def predict_fresh_or_rotten(model, img_file):
    img_array = load_image_for_prediction(img_file)
    prediction = model.predict(img_array)[0][0] # Ambil nilai skalar probabilitas untuk klasifikasi biner

    if prediction > 0.5:
        label = 'Busuk'
        confidence = prediction
    else:
        label = 'Segar' 
        confidence = 1 - prediction

    return label, confidence # Mengembalikan label dan tingkat keyakinan


st.title("Klasifikasi Buah: Segar atau Busuk?")
st.markdown("---") # Garis pemisah untuk estetika

st.write("Unggah gambar buah Apel, Pisang dan Jeruk untuk menentukan apakah buah tersebut segar atau busuk.")

uploaded_file = st.file_uploader("Pilih gambar buah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah oleh pengguna
    st.image(uploaded_file, caption='Gambar yang Diunggah', use_container_width=True) # Menggunakan use_container_width
    st.write("") # Baris kosong untuk spasi

    # Menampilkan indikator loading saat model memproses
    with st.spinner('Menganalisis gambar buah...'):
        time.sleep(1.5) # Simulasi waktu pemrosesan model (sesuaikan jika model Anda butuh waktu lebih lama)
        label, confidence = predict_fresh_or_rotten(model_fresh_rotten, uploaded_file)

    # Menampilkan hasil prediksi beserta tingkat keyakinan dalam persentase
    confidence_percent = confidence * 100 # Konversi probabilitas ke persentase

    if label == 'Segar':
        st.success(f"🎉 **Hasil Prediksi:** Buah ini adalah **{label}** dengan keyakinan **{confidence_percent:.2f}%**!")
    else:
        st.error(f"⚠️ **Hasil Prediksi:** Buah ini adalah **{label}** dengan keyakinan **{confidence_percent:.2f}%**.")

st.markdown("---") # Garis pemisah di bagian bawah
st.markdown("""
<div class="footer">
    Aplikasi Klasifikasi Buah oleh AkmalAditAlbarr | Dibuat dengan Streamlit dan TensorFlow
</div>
""", unsafe_allow_html=True)