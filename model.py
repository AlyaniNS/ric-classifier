"""
RIC Classification Model handler supporting multiple model types.
Supports both backbone+prototype and TorchScript approaches.
"""
import os
import io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    VARIANT_FILES, get_model_variant, get_base_dir,
    DEFAULT_TEMPERATURE, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
)

class RICClassifier:
    """RIC Classification model supporting multiple model types."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = None
        self.prototypes = None
        self.model = None  # For TorchScript
        self.class_names = ['1_PET', '2_HDPE', '3_PVC', '4_LDPE', '5_PP', '6_PS', '7_OTHER']
        self.chart_image = None
        self.transform = self._create_transform()
        self.model_type = None
        self._load_model()
    
    def _create_transform(self):
        """Create image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])
    
    def _preprocess_image(self, image_path):
        """Load and preprocess image."""
        img = Image.open(image_path).convert("RGB")
        
        # Optional: Apply slight enhancements
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)  # Slight contrast boost
        
        return img
    
    def _load_model(self):
        """Load model based on variant configuration."""
        base_dir = get_base_dir()
        requested_variant = get_model_variant()
        
        if requested_variant not in VARIANT_FILES:
            requested_variant = "finetuned"  # Default to finetuned
        
        backbone_path, prototype_path = VARIANT_FILES[requested_variant]
        backbone_full_path = os.path.join(base_dir, backbone_path)
        
        # Check if this is a TorchScript model (single file, no prototypes)
        if prototype_path is None:
            if os.path.exists(backbone_full_path):
                print(f"[Model] Loading TorchScript model: {backbone_full_path}")
                self.model = torch.jit.load(backbone_full_path, map_location=self.device)
                self.model.eval()
                self.model_type = "torchscript"
                print(f"[Model] TorchScript model loaded successfully")
                return
            else:
                raise RuntimeError(f"TorchScript model not found at {backbone_full_path}")
        
        # Load backbone + prototype models
        prototype_full_path = os.path.join(base_dir, prototype_path)
        
        if not os.path.exists(backbone_full_path):
            raise RuntimeError(f"Backbone model not found at {backbone_full_path}")
        if not os.path.exists(prototype_full_path):
            raise RuntimeError(f"Prototype model not found at {prototype_full_path}")
        
        print(f"[Model] Loading backbone: {backbone_full_path}")
        print(f"[Model] Loading prototypes: {prototype_full_path}")
        
        # Load backbone (handle both TorchScript and regular PyTorch models)
        try:
            # Try loading as TorchScript first
            self.backbone = torch.jit.load(backbone_full_path, map_location=self.device)
            print(f"[Model] Backbone loaded as TorchScript")
        except Exception:
            # Fallback to regular PyTorch model
            self.backbone = torch.load(backbone_full_path, map_location=self.device, weights_only=False)
            print(f"[Model] Backbone loaded as regular PyTorch model")
        
        if hasattr(self.backbone, 'eval'):
            self.backbone.eval()
        
        # Load prototypes
        try:
            self.prototypes = torch.jit.load(prototype_full_path, map_location=self.device)
            print(f"[Model] Prototypes loaded as TorchScript")
        except Exception:
            self.prototypes = torch.load(prototype_full_path, map_location=self.device, weights_only=False)
            print(f"[Model] Prototypes loaded as regular PyTorch model")
            if isinstance(self.prototypes, dict) and 'prototypes' in self.prototypes:
                self.prototypes = self.prototypes['prototypes']
        
        self.model_type = "backbone_prototype"
        print(f"[Model] Backbone + prototype models loaded successfully")
        print(f"[Model] Classes: {self.class_names}")
    
    def format_class_name(self, class_name):
        """Convert class names like '1_PET' to 'PET' for display."""
        if '_' in class_name:
            return class_name.split('_', 1)[1]
        return class_name
    
    @torch.no_grad()
    def predict(self, image_path, temperature=DEFAULT_TEMPERATURE):
        """
        Predict RIC class using loaded model.
        
        Returns:
            tuple: (prediction, probabilities, confidence)
        """
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load and preprocess image
        img = self._preprocess_image(image_path)
        x = self.transform(img).unsqueeze(0).to(self.device)
        
        # Inference based on model type
        if self.model_type == "torchscript":
            sims = self.model(x)
        elif self.model_type == "backbone_prototype":
            # Extract features using backbone
            features = self.backbone(x)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)  # Flatten if needed
            
            # Ensure prototypes are in the right format
            if isinstance(self.prototypes, torch.Tensor):
                prototypes = self.prototypes
            elif hasattr(self.prototypes, 'data'):
                prototypes = self.prototypes.data
            else:
                prototypes = self.prototypes
            
            # Move prototypes to same device as features
            prototypes = prototypes.to(self.device)
            
            # Compute similarities with prototypes
            features = nn.functional.normalize(features, p=2, dim=1)
            prototypes_normalized = nn.functional.normalize(prototypes, p=2, dim=1)
            sims = torch.mm(features, prototypes_normalized.t()) / temperature
        else:
            raise RuntimeError(f"Unknown model type: {self.model_type}")
        
        # Convert to probabilities
        probs = torch.softmax(sims, dim=1)[0].cpu().numpy() * 100
        
        # Get prediction results
        pred_idx = sims.argmax(1).item()
        pred_class = self.class_names[pred_idx]
        confidence = probs[pred_idx]
        
        # Sort results for display
        sorted_indices = np.argsort(probs)[::-1]
        sims_sorted = probs[sorted_indices]
        class_sorted = [self.class_names[i] for i in sorted_indices]
        
        # Generate chart
        self._create_chart(class_sorted, sims_sorted, pred_class)
        
        # Format results for display
        formatted_probs = {self.format_class_name(cls): float(prob) for cls, prob in zip(class_sorted, sims_sorted)}
        return self.format_class_name(pred_class), formatted_probs, confidence
    
    def _create_chart(self, class_sorted, sims_sorted, pred_class):
        """Create visualization chart for predictions."""
        # Convert to pure Python types to avoid matplotlib strictness/numpy mismatch
        class_sorted = [str(c) for c in class_sorted]
        sims_sorted = [float(s) for s in sims_sorted]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        y = range(len(class_sorted))
        bars = ax.barh(list(y), sims_sorted, color='skyblue')
        bars[0].set_color('orange')
        
        for i, v in enumerate(sims_sorted):
            ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=10)
        
        ax.set_yticks(list(y))
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
        self.chart_image = img_io
    
    def get_chart(self):
        """Get the generated chart image."""
        return self.chart_image

