# -*- coding: utf-8 -*-
"""
Localization for the RIC Classification application.
Contains all text strings in Indonesian and English.
"""

LANGUAGES = {
    'id': {
        'title': 'Klasifikasi Kode Resin Pada Kemasan Plastik',
        'subtitle': 'Few-Shot Learning dengan EfficientNet-B2',
        'upload_label': 'Unggah gambar kode resin',
        'choose_file': 'Pilih Gambar',
        'no_file_selected': 'Belum ada gambar terunggah',
        'predict_btn': 'Prediksi',
        'result_title': 'Hasil prediksi',
        'uploaded_image': 'Gambar yang diunggah',
        'confidence_percentage': 'Persentase kemiripan',
        'about_plastic': 'Tentang Plastik Ini',
        'report_btn': 'Hasil prediksi tidak sesuai? Lapor disini',
        'report_success': 'Terima kasih! Laporan Anda telah berhasil dikirim.',
        'no_file_error': 'Tidak ada file yang diunggah.',
        'not_ric_error': 'Bukan Kode Identifikasi Resin, tidak dapat diklasifikasi (confidence: {:.1f}%)',
        'not_ric_suggestion': 'Silakan upload gambar yang mengandung kode resin (angka 1-7 dalam segitiga)',
        'processing_error': 'Error memproses gambar: {}. Silakan coba lagi dengan gambar RIC yang jelas.',
        'try_another': 'Coba Gambar Lain',
        'not_ric_title': 'Bukan Kode Resin!',
        'processing': 'Memproses...',
        'debug_info': 'Info Debug: Confidence: {:.1f}%, Gap: {:.1f}%, Flags: {}',
        'footer': '© 2025 Dibuat oleh <a href="https://alyanins.tech" target="_blank" rel="noopener noreferrer">Alyani Septalia</a> untuk Keperluan Skripsi',
        'change_image': 'Ganti gambar',
        'or_try_sample': 'Atau coba salah satu gambar ini:',
        'sample_label': 'Contoh {}'
    },
    'en': {
        'title': 'Resin Identification Code Classification on Plastic Packaging',
        'subtitle': 'Few-Shot Learning with EfficientNet-B2',
        'upload_label': 'Upload resin code image',
        'choose_file': 'Choose File',
        'no_file_selected': 'No image uploaded yet',
        'predict_btn': 'Predict',
        'result_title': 'Prediction result',
        'uploaded_image': 'Uploaded image',
        'confidence_percentage': 'Confidence percentage',
        'about_plastic': 'About This Plastic',
        'report_btn': 'Prediction result incorrect? Report here',
        'report_success': 'Thank you! Your report has been successfully submitted.',
        'no_file_error': 'No file uploaded.',
        'not_ric_error': 'Not a Resin Identification Code, cannot be classified (confidence: {:.1f}%)',
        'not_ric_suggestion': 'Please upload an image containing resin code (numbers 1-7 in triangle)',
        'processing_error': 'Error processing image: {}. Please try again with a clear RIC image.',
        'try_another': 'Try Another Image',
        'not_ric_title': 'Not a Resin Code!',
        'processing': 'Processing...',
        'debug_info': 'Debug Info: Confidence: {:.1f}%, Gap: {:.1f}%, Flags: {}',
        'footer': '© 2025 Made by <a href="https://alyanins.tech" target="_blank" rel="noopener noreferrer">Alyani Septalia</a> for Undergraduate Thesis Purposes',
        'change_image': 'Change image',
        'or_try_sample': 'Or try one of these:',
        'sample_label': 'Sample {}'
    }
}



def get_language(session):
    """Get current language from session, default to Indonesian."""
    return session.get('language', 'id')

# Plastic type names with proper chemical names
PLASTIC_TYPES = {
    'id': {
        '1_PET': 'Polietilena Tereftalat (1/PET)',
        '2_HDPE': 'Polietilena Dens. Tinggi (2/HDPE)',
        '3_PVC': 'Polivinil Klorida (3/PVC)',
        '4_LDPE': 'Polietilena Dens. Rendah (4/LDPE)',
        '5_PP': 'Polipropilena (5/PP)',
        '6_PS': 'Polistirena (6/PS)',
        '7_OTHER': 'Lainnya (7/OTHER)',
        'PET': 'Polietilena Tereftalat (1/PET)',
        'HDPE': 'Polietilena Dens. Tinggi (2/HDPE)',
        'PVC': 'Polivinil Klorida (3/PVC)',
        'LDPE': 'Polietilena Dens. Rendah (4/LDPE)',
        'PP': 'Polipropilena (5/PP)',
        'PS': 'Polistirena (6/PS)',
        'OTHER': 'Lainnya (7/OTHER)'
    },
    'en': {
        '1_PET': 'Polyethylene Terephthalate (1/PET)',
        '2_HDPE': 'High-Density Polyethylene (2/HDPE)',
        '3_PVC': 'Polyvinyl Chloride (3/PVC)',
        '4_LDPE': 'Low-Density Polyethylene (4/LDPE)',
        '5_PP': 'Polypropylene (5/PP)',
        '6_PS': 'Polystyrene (6/PS)',
        '7_OTHER': 'Other (7/OTHER)',
        'PET': 'Polyethylene Terephthalate (1/PET)',
        'HDPE': 'High-Density Polyethylene (2/HDPE)',
        'PVC': 'Polyvinyl Chloride (3/PVC)',
        'LDPE': 'Low-Density Polyethylene (4/LDPE)',
        'PP': 'Polypropylene (5/PP)',
        'PS': 'Polystyrene (6/PS)',
        'OTHER': 'Other (7/OTHER)'
    }
}

def get_text(session, key, *args):
    """Get localized text for the given key and format with arguments."""
    lang = get_language(session)
    text = LANGUAGES[lang].get(key, LANGUAGES['id'][key])
    if args:
        return text.format(*args)
    return text

def get_plastic_type_name(session, class_index):
    """
    Get full plastic type name with chemical name for the given class label (e.g., '5_PP').
    """
    lang = get_language(session)
    # class_index bisa '5_PP' atau 'PP', kita pastikan konsisten
    key = str(class_index).strip()

    # Langsung cari key yang ada di kamus
    if key in PLASTIC_TYPES[lang]:
        return PLASTIC_TYPES[lang][key]

    # Kalau cuma PP / PET tanpa angka, coba cari yang cocok
    for full_key, full_name in PLASTIC_TYPES[lang].items():
        if full_key.endswith(key):
            return full_name

    # Kalau masih tidak ketemu
    return f"{key} - Unknown"