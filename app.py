import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

# Judul aplikasi
st.title("Osteoarthritis Prediction from X-ray images")

# Deskripsi aplikasi
st.write("Upload Knee X-Ray images.")

# Load model YOLOv8 yang sudah dilatih
model_path = "models\best.pt"  # Ganti dengan path model kamu
model = YOLO(model_path)

# Upload gambar
uploaded_file = st.file_uploader("Uplad Images", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Tombol untuk memproses gambar
    if st.button("Predict"):
        st.write("Processing...")

        # Simpan sementara file yang diunggah
        image_path = "temp.jpg"
        image.save(image_path)

        # Lakukan prediksi menggunakan model
        results = model(image_path)

        # Ambil prediksi kelas dengan confidence tertinggi
        predicted_class = results[0].probs.top1  # Indeks kelas tertinggi
        confidence = results[0].probs.top1conf.item()  # Confidence score

        # Mapping indeks ke nama kelas (ganti sesuai label dataset kamu)
        class_labels = ["Normal", "Osteopenia", "Osteoporosis"]  # Sesuaikan dengan dataset
        predicted_label = class_labels[predicted_class]

        # Tampilkan hasil prediksi
        st.success(f"Prediction: **{predicted_label}**")
        st.write(f"Confidence Score: **{confidence:.2f}**")