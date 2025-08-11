import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from flask import Flask, render_template, request, send_file, redirect, url_for
from PIL import Image
from torchvision import transforms

# --- Fix matplotlib backend ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Flask setup ---
app = Flask(__name__, template_folder="templates", static_folder="static")

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Plastic education dictionary ---
plastic_education = {
    "1_PET": "PET digunakan untuk botol air dan minuman ringan. Mudah didaur ulang.",
    "2_HDPE": "HDPE biasanya digunakan untuk botol susu, galon, dan produk rumah tangga.",
    "3_PVC": "PVC digunakan untuk pipa dan kemasan. Sulit didaur ulang.",
    "4_LDPE": "LDPE dipakai pada kantong belanja dan plastik pembungkus.",
    "5_PP": "PP sering ditemukan di wadah makanan dan tutup botol.",
    "6_PS": "PS (styrofoam) digunakan untuk wadah makanan sekali pakai.",
    "7_OTHER": "Kategori lainnya termasuk plastik campuran dan bioplastik."
}

# --- Load backbone model ---
backbone = timm.create_model("efficientnet_b2", pretrained=False, num_classes=0)
# backbone.load_state_dict(torch.load("fsl_efficientnet_b2_backbone_v3.pth", map_location=device))
# backbone.load_state_dict(torch.load("/Users/alyani-dmg/skripsi/proto_effb2_5shot_backbone.pth", map_location=device)) #test with 5-shot
backbone.load_state_dict(torch.load("/Users/alyani-dmg/skripsi/proto_effb2_10shot_backbone.pth", map_location=device)) #test with 10-shot
backbone.eval()

# --- Load class prototypes ---
# proto_data = torch.load("class_prototypes_finetuned_v3.pt", map_location=device)
# proto_data = torch.load("/Users/alyani-dmg/skripsi/proto_effb2_5shot_prototypes.pt", map_location=device)  # 5-shot
proto_data = torch.load("/Users/alyani-dmg/skripsi/proto_effb2_10shot_prototypes.pt", map_location=device)  # 10-shot
prototypes = proto_data["prototypes"].to(device)
class_names = proto_data["class_names"]

# --- Image transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Chart image in memory ---
chart_image = None

# --- Prediction logic ---
def predict(image_path, temperature=0.1):
    global chart_image

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = F.normalize(backbone(img_tensor).flatten(1), dim=1)
        proto_norm = F.normalize(prototypes, dim=1)
        sims = F.cosine_similarity(query_emb, proto_norm)

    sims_softmax = torch.softmax(sims / temperature, dim=0).cpu().numpy() * 100
    sorted_indices = np.argsort(sims_softmax)[::-1]
    sims_sorted = sims_softmax[sorted_indices]
    class_sorted = [class_names[i] for i in sorted_indices]

    pred_class = class_sorted[0]

    # Plotting chart
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(class_sorted))
    bars = ax.barh(y, sims_sorted, color='skyblue')
    bars[0].set_color('orange')
    for i, v in enumerate(sims_sorted):
        ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(class_sorted)
    ax.set_xlabel("Similarity (%)")
    ax.set_title(f"Prediction: {pred_class} ({sims_sorted[0]:.2f}%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()

    # Top 3 box
    top3_text = "\n".join([f"{class_sorted[i]}: {sims_sorted[i]:.2f}%" for i in range(min(3, len(sims_sorted)))])
    fig.text(0.02, 0.01, f"Top 3:\n{top3_text}", ha='left', va='bottom', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', bbox_inches='tight')
    plt.close(fig)
    img_io.seek(0)
    chart_image = img_io

    return pred_class, {cls: float(prob) for cls, prob in zip(class_sorted, sims_sorted)}

# --- Index route ---
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probs = None
    error = None
    image_url = None
    report_success = request.args.get("reported") == "true"
    plastic_info = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "Tidak ada file yang diunggah."
        else:
            upload_dir = os.path.join("static", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)
            image_url = f"uploads/{file.filename}"

            prediction, probs = predict(file_path)
            plastic_info = plastic_education.get(prediction)

    return render_template("index.html", prediction=prediction, probs=probs, error=error,
                           image_url=image_url, report_success=report_success, plastic_info=plastic_info)

# --- Serve chart image ---
@app.route("/chart.png")
def chart():
    global chart_image
    if chart_image:
        return send_file(chart_image, mimetype="image/png")
    return "No chart available", 404

# --- Report logic ---
@app.route("/report", methods=["POST"])
def report():
    prediction = request.form.get("prediction")
    image_url = request.form.get("image_url")
    os.makedirs("logs", exist_ok=True)
    with open("logs/reports.txt", "a") as f:
        f.write(f"Reported: {prediction} | Image: {image_url}\n")
    return redirect(url_for("index", reported="true"))

# --- Run app ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)
