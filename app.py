import os
import io
import base64
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
import matplotlib.pyplot as plt

# --- Flask setup ---
app = Flask(__name__, template_folder="templates", static_folder="static")

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model backbone ---
backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
state_dict = torch.load(r"D:\Skripsi\App\fsl_efficientnet_b0_backbone_finetuned_v2.pth", map_location=device)
backbone.load_state_dict(state_dict, strict=True)
backbone = backbone.to(device).eval()

# --- Load prototypes ---
proto_data = torch.load(r"D:\Skripsi\App\class_prototypes_finetuned_v2.pt", map_location=device)
prototypes = proto_data["prototypes"].to(device)
class_names = proto_data["class_names"]

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Prediction function ---
def predict(image_path, visualize=False):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        query_emb = backbone(img_tensor).flatten(1)
        sims = F.cosine_similarity(query_emb, prototypes)
        probs = torch.softmax(sims, dim=0).cpu().numpy() * 100

    pred_idx = torch.argmax(sims).item()
    pred_class = class_names[pred_idx]

    # Chart file path
    chart_path = "static/history/prediction_chart.png"

    if visualize:
        plt.figure(figsize=(6, 4))
        bars = plt.barh(class_names, probs, color='skyblue')
        bars[pred_idx].set_color('orange')
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{probs[i]:.1f}%", va='center')
        plt.title(f"Predicted Class: {pred_class} ({probs[pred_idx]:.2f}%)", fontsize=14)
        plt.xlabel("Similarity (%)")
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()

    return pred_class, dict(zip(class_names, probs)), chart_path

# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    chart = None
    probs = None
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")
        file_path = os.path.join("static", file.filename)
        file.save(file_path)
        prediction, probs, chart = predict(file_path)
    return render_template("index.html", prediction=prediction, probs=probs, chart=chart)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
