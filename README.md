# Klasifikasi Kode Resin Pada Kemasan Plastik

Aplikasi web untuk klasifikasi Resin Identification Code (RIC) pada kemasan plastik menggunakan Few-Shot Learning dengan EfficientNet-B2.

## Fitur Utama

- 🔍 Klasifikasi otomatis 7 jenis plastik (PET, HDPE, PVC, LDPE, PP, PS, OTHER)
- 🌐 Dukungan bilingual (Indonesia & English)
- 📊 Visualisasi confidence score dengan progress bar
- 📱 Responsive design untuk mobile dan desktop
- 🎨 Modern UI dengan animasi smooth

## Teknologi

- **Backend**: Flask (Python)
- **Model**: EfficientNet-B2 dengan Few-Shot Learning
- **Frontend**: Bootstrap 5, Custom CSS
- **Deep Learning**: PyTorch, timm

## Instalasi & Menjalankan Lokal

```bash
# Clone repository
git clone <repository-url>
cd skripsi

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
python app.py
```

Aplikasi akan berjalan di `http://localhost:5000`

## Deployment ke Render

1. Push repository ke GitHub
2. Login ke [render.com](https://render.com)
3. Buat Web Service baru
4. Connect repository ini
5. Render akan otomatis detect `render.yaml` dan deploy

File konfigurasi sudah tersedia di `render.yaml`.

## Struktur Project

```
app.py              # Main application
config.py           # Configuration settings
localization.py     # Translations (ID/EN)
model.py            # ML model handler
templates/          # HTML templates
static/             # CSS, JS, images
models/             # ML model files
```

## Cara Penggunaan

1. Upload gambar kode resin plastik
2. Klik tombol "Prediksi"
3. Lihat hasil klasifikasi dan confidence score
4. Lapor jika hasil tidak sesuai (opsional)

## License

© 2025 Dibuat oleh Alyani Septalia untuk Keperluan Skripsi
- **Error Handling**: Graceful handling of non-RIC images
- **Development Reports**: Report functionality for all cases
- **Debug Mode**: Configurable debug settings

## Usage

### Running the Application
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

### Configuration Options
Set environment variables to customize behavior:
```bash
# Set model variant (10shot, 5shot, or v3)
set MODEL_VARIANT=10shot  # Windows
export MODEL_VARIANT=10shot  # Linux/macOS
```

### Adding New Components
1. Create new component in `templates/components/`
2. Include in main template: `{% include 'components/your_component.html' %}`
3. Add any required CSS to `static/style.css`

### Adding New Languages
1. Add language data to `localization.py` in `LANGUAGES` and `PLASTIC_EDUCATION`
2. Add flag icon and dropdown item in `header.html`
3. Update language switching logic in `app.py` if needed

## File Descriptions

### `app.py`
Main Flask application with clean route definitions. Handles:
- Route definitions and request handling
- Session management
- File uploads and error handling
- Template rendering with context

### `config.py`
Centralized configuration including:
- Flask settings (secret key, debug mode)
- Model file mappings and variants
- Prediction thresholds
- File upload settings
- Directory paths

### `localization.py`
All text strings and translations:
- UI text in Indonesian and English
- Educational content about plastic types
- Helper functions for text retrieval

### `model.py`
ML model handling with:
- Model loading and initialization
- Prediction logic with confidence thresholds
- Chart generation for visualizations
- Class name formatting utilities

### Template Components
- **`header.html`**: Language switcher, notifications, hero section
- **`upload.html`**: File upload form with error handling
- **`classification_result.html`**: Prediction results with animated bars
- **`not_ric_result.html`**: Non-RIC handling with suggestions
- **`footer.html`**: Copyright footer with localized text

## Benefits of Modular Structure

1. **Maintainability**: Each component can be updated independently
2. **Reusability**: Components can be reused across different templates
3. **Scalability**: Easy to add new features or languages
4. **Testing**: Individual components can be tested separately
5. **Collaboration**: Multiple developers can work on different components
6. **Documentation**: Clear separation makes code self-documenting