import streamlit as st
import tempfile
import os
import shutil
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Tumor Otak Berbasis YOLOv8-CLS", layout='wide')
st.title("🧠 Deteksi Tumor Otak Menggunakan YOLOv8-CLS")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar MRI tumor otak", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "gambar.jpg")
    
    image = Image.open(uploaded_file)
    image.save(temp_file)
    
    # Tata letak halaman: 2 kolom utama
    col1, col2 = st.columns([1.2, 1.8])  # Gambar lebih kecil dari hasil deteksi
    
    with col1:
        st.image(image, caption="🖼 Gambar yang diunggah", use_container_width=True)
        confidence_threshold = st.slider("Threshold Keyakinan (%)", 0, 100, 50)
    
    with col2:
        if st.button("🔍 Deteksi Gambar"):
            with st.spinner("Model sedang memproses gambar..."):
                try:
                    model = YOLO('best.pt')
                    hasil = model(temp_file)
                    nama_objek = hasil[0].names
                    nilai_prediksi = hasil[0].probs.data.numpy().tolist()
                    objek_terdeteksi = nama_objek[np.argmax(nilai_prediksi)]
                    confidence = max(nilai_prediksi) * 100
                    
                    # Filter hasil berdasarkan threshold confidence
                    filtered_results = {k: v for k, v in zip(nama_objek.values(), nilai_prediksi) if v * 100 >= confidence_threshold}
                    
                    st.subheader(f"**Hasil Deteksi: {objek_terdeteksi} ({confidence:.2f}%)**")
                    
                    if filtered_results:
                        # Grafik lebih kecil agar tidak mendominasi tampilan
                        grafik = go.Figure([go.Bar(x=list(filtered_results.keys()), 
                                                   y=[v * 100 for v in filtered_results.values()], 
                                                   marker_color='lightblue')])
                        grafik.update_layout(title='📊 Tingkat Keyakinan Prediksi', 
                                             xaxis_title='Jenis Tumor', 
                                             yaxis_title='Keyakinan (%)', 
                                             height=280, 
                                             margin=dict(l=30, r=30, t=40, b=30))
                        st.plotly_chart(grafik, use_container_width=True)

                        # Tabel berada langsung di bawah grafik
                        st.markdown("### 📋 Hasil Klasifikasi")
                        df = pd.DataFrame({"Jenis Tumor": list(filtered_results.keys()), 
                                           "Keyakinan (%)": [v * 100 for v in filtered_results.values()]})
                        st.dataframe(df, height=180)

                        # Unduh hasil klasifikasi
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Unduh Hasil Klasifikasi", data=csv, file_name="hasil_klasifikasi.csv", mime='text/csv')
                    else:
                        st.warning("Tidak ada tumor yang melebihi threshold keyakinan.")
                
                except Exception as e:
                    st.error("Terjadi kesalahan saat memproses gambar.")
                    st.error(f"Error: {e}")
            
            # Hapus file sementara setelah digunakan
            shutil.rmtree(temp_dir, ignore_errors=True)

# Footer
st.markdown("<div style='text-align: center; color: gray; margin-top:20px;'>Program Aplikasi deteksi tumor otak @2025</div>", unsafe_allow_html=True)
