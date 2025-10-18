"""
RIC Classification Flask Application
Main application file with modular structure and clean separation of concerns.
"""
import os
from flask import Flask, render_template, request, send_file, redirect, url_for, session

# Import our modular components
from config import SECRET_KEY, DEBUG, PORT, UPLOAD_FOLDER, LOG_DIR, REPORTS_FILE
from localization import LANGUAGES, get_language, get_text, get_plastic_type_name
from model import RICClassifier

# --- Flask setup ---
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = SECRET_KEY

# --- Initialize model ---
classifier = RICClassifier()

# --- Helper functions ---
def ensure_upload_dir():
    """Ensure upload directory exists."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def ensure_log_dir():
    """Ensure log directory exists."""
    os.makedirs(LOG_DIR, exist_ok=True)

# --- Routes ---

@app.route("/<language>")
def set_language(language):
    """Handle language switching."""
    if language in LANGUAGES:
        session['language'] = language
    return redirect(url_for('index'))

@app.route("/", methods=["GET", "POST"])
def index():
    """Main application route."""
    prediction = None
    probs = None
    error = None
    image_url = None
    report_success = request.args.get("reported") == "true"
    plastic_info = None
    confidence = None
    is_not_ric = False

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = get_text(session, 'no_file_error')
        else:
            try:
                ensure_upload_dir()
                
                # Create unique filename to prevent caching issues
                import time
                timestamp = str(int(time.time()))
                file_extension = os.path.splitext(file.filename)[1]
                unique_filename = f"{timestamp}_{file.filename}"
                file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                
                file.save(file_path)
                image_url = f"uploads/{unique_filename}"

                # Get prediction from TorchScript model
                prediction, probs, confidence = classifier.predict(file_path)
                
                # Always show prediction (no exception handling)
                # Model will always return a prediction
                    
            except Exception as e:
                error = get_text(session, 'processing_error', str(e))

    # Get current language and texts
    current_lang = get_language(session)
    texts = LANGUAGES[current_lang]
    
    return render_template("index.html", prediction=prediction, probs=probs, error=error,
                           image_url=image_url, report_success=report_success,
                           confidence=confidence, is_not_ric=is_not_ric, texts=texts, current_lang=current_lang,
                           get_plastic_type_name=get_plastic_type_name)

@app.route("/chart.png")
def chart():
    """Serve generated chart image."""
    chart_image = classifier.get_chart()
    if chart_image:
        return send_file(chart_image, mimetype="image/png")
    return "No chart available", 404

@app.route("/report", methods=["POST"])
def report():
    """Handle report submissions."""
    prediction = request.form.get("prediction")
    image_url = request.form.get("image_url")
    
    ensure_log_dir()
    log_file = os.path.join(LOG_DIR, REPORTS_FILE)
    with open(log_file, "a") as f:
        f.write(f"Reported: {prediction} | Image: {image_url}\n")
    
    return redirect(url_for("index", reported="true"))

# --- Application entry point ---
if __name__ == "__main__":
    app.run(debug=DEBUG, port=PORT)
