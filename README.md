# Klasifikasi Kode Resin Pada Kemasan Plastik

Aplikasi web untuk klasifikasi Resin Identification Code (RIC) pada kemasan plastik menggunakan Few-Shot Learning dengan EfficientNet-B2.

## Fitur 

- Klasifikasi 7 jenis plastik (PET, HDPE, PVC, LDPE, PP, PS, OTHER)
- Bilingual (Indonesia & English)
- Visualisasi confidence score dengan progress bar

## Tech Stack

- **Framework**: Flask
- **Model**: EfficientNet-B2 - Few Shot Learning
- **Frontend**: Bootstrap 5, Custom CSS
- **Deep Learning**: PyTorch, timm

## Command Instalasi di lokal

```bash
# Clone repo
git clone <repository-url>
cd skripsi

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
python app.py
```

Web app running di `http://localhost:5000`


## Cara Penggunaan

1. Upload gambar kode resin plastik
2. Klik tombol "Prediksi"
3. Lihat hasil klasifikasi dan confidence score
4. Lapor jika hasil tidak sesuai (opsional)

## License

© 2025 Dibuat oleh Alyani Septalia untuk Keperluan Skripsi