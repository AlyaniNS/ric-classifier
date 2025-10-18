# RIC Classification Application - Modular Structure

This Flask application classifies Resin Identification Codes (RIC) on plastic packaging using few-shot learning with EfficientNet-B2.

## Project Structure

```
app.py                          # Main Flask application (routes and app logic)
config.py                       # Configuration settings
localization.py                 # Localization data (Indonesian/English)
model.py                        # ML model handling and prediction logic
requirements.txt                # Python dependencies
static/
├── style.css                   # Main stylesheet with design tokens
├── uploads/                    # Uploaded images
└── history/                    # Upload history
templates/
├── index.html                  # Main template (now modular)
└── components/                 # Reusable template components
    ├── header.html            # Language switcher and hero section
    ├── upload.html            # File upload form
    ├── classification_result.html  # Prediction results display
    ├── not_ric_result.html    # Non-RIC error handling
    └── footer.html            # Footer with copyright
logs/
└── reports.txt                # User reports
```

## Key Features

### Modular Architecture
- **Separation of Concerns**: Each file has a specific responsibility
- **Reusable Components**: Template components can be easily maintained
- **Configuration Management**: Centralized settings in `config.py`
- **Localization**: Separate file for all translations

### Model Configuration
- **Multiple Model Variants**: Support for 10-shot, 5-shot, and v3 models
- **Environment Variables**: Configure model variant via `MODEL_VARIANT` env var
- **Automatic Fallback**: Falls back to available models if preferred not found
- **Cross-Platform**: Works on Windows, macOS, and Linux

### User Interface
- **Responsive Design**: Bootstrap 5 with custom design tokens
- **Animated Progress Bars**: Smooth animations for prediction results
- **Bilingual Support**: Indonesian and English languages
- **Modern Styling**: Clean, accessible design with eco-friendly theme

### Development Features
- **Component-Based Templates**: Easy to modify individual sections
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