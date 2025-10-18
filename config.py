"""
Configuration settings for the RIC Classification application.
"""
import os

# Flask Configuration
SECRET_KEY = 'your-secret-key-here'  # Change this in production
DEBUG = True
PORT = 5000

# Model Configuration
VARIANT_FILES = {
    "finetuned": ("models/fine_tuned.pt", "models/prototype_finetuned.pth"),  # Your new finetuned models
    "torchscript": ("models/model_effb2_full.pt", None),  # TorchScript model (single file)
    "best": ("models/best_val.pth", "models/prototype.pth"),  # Previous models
}

DEFAULT_MODEL_VARIANT = "finetuned"  # Use your new finetuned models as default

# Prediction Configuration
DEFAULT_TEMPERATURE = 0.05  # Lower temperature for sharper predictions (like your Colab)
DEFAULT_MIN_CONFIDENCE = 85.0  # Balanced threshold
CONFIDENCE_GAP_THRESHOLD = 30.0  # Larger gap requirement for better discrimination
MAX_CONFIDENCE_FOR_REJECTION = 99.5  # If confidence is too perfect, it might be wrong
MIN_SECOND_PLACE_CONFIDENCE = 5.0  # Second place should have some meaningful score

# File Upload Configuration
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Image Processing
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Logging
LOG_DIR = "logs"
REPORTS_FILE = "reports.txt"

def get_model_variant():
    """Get model variant from environment variable or default."""
    return os.getenv("MODEL_VARIANT", DEFAULT_MODEL_VARIANT).lower()

def get_base_dir():
    """Get the base directory of the application."""
    return os.path.dirname(os.path.abspath(__file__))