"""
Brain Tumor MRI Classifier — Inference Module
Model: google/vit-base-patch16-224 (fine-tuned)
Classes: glioma, meningioma, notumor, pituitary
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from transformers import ViTForImageClassification
import torchvision.transforms as transforms


# ─────────────────────────── Constants ───────────────────────────

CLASS_NAMES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_INFO: dict[str, dict] = {
    "glioma": {
        "full_name": "Glioma Tumor",
        "description": "A tumor that originates in the glial cells of the brain or spine.",
        "severity": "High",
        "color": "#e74c3c",
    },
    "meningioma": {
        "full_name": "Meningioma Tumor",
        "description": "A tumor that arises from the meninges surrounding the brain and spinal cord.",
        "severity": "Moderate",
        "color": "#e67e22",
    },
    "notumor": {
        "full_name": "No Tumor Detected",
        "description": "No signs of tumor found in the MRI scan.",
        "severity": "None",
        "color": "#27ae60",
    },
    "pituitary": {
        "full_name": "Pituitary Tumor",
        "description": "A tumor that forms in the pituitary gland at the base of the brain.",
        "severity": "Moderate",
        "color": "#8e44ad",
    },
}

MODEL_PATH = Path(__file__).parent.parent / "model" / "vit_brain_tumor.pth"
MODEL_NAME  = "google/vit-base-patch16-224"
IMAGE_SIZE  = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────── Data classes ────────────────────────

@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    top_k: list[dict]          # [{"class": ..., "confidence": ...}, ...]
    gradcam_image: np.ndarray  # BGR uint8, same size as input


# ─────────────────────────── Transforms ──────────────────────────

def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ─────────────────────────── Model loader ────────────────────────

_model_cache: ViTForImageClassification | None = None


def load_model() -> ViTForImageClassification:
    """
    Singleton loader — loads weights once and caches in memory.
    Subsequent calls return the cached model instantly.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}.\n"
            "Run the training notebook first to generate vit_brain_tumor.pth"
        )

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    _model_cache = model
    print(f"[ModelLoader] Loaded weights from {MODEL_PATH} on {DEVICE}")
    return _model_cache


# ─────────────────────────── Grad-CAM ────────────────────────────

class GradCAM:
    """
    Grad-CAM for ViT — attaches hooks to the last attention layer's output.
    ViT doesn't have conv layers, so we use the last transformer block's
    layer norm output as the target feature map.
    """

    def __init__(self, model: ViTForImageClassification):
        self.model    = model
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        # Target: last ViT encoder block
        target_layer = self.model.vit.encoder.layer[-1].layernorm_before

        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        original_image: np.ndarray,
    ) -> np.ndarray:
        """
        Returns a Grad-CAM heatmap overlaid on the original image.

        Args:
            input_tensor   : (1, 3, 224, 224) preprocessed tensor
            class_idx      : predicted class index to explain
            original_image : HxWx3 BGR uint8 numpy array

        Returns:
            HxWx3 BGR uint8 numpy array with heatmap overlay
        """
        self.model.zero_grad()
        input_tensor = input_tensor.to(DEVICE).requires_grad_(True)

        output = self.model(pixel_values=input_tensor).logits
        score  = output[0, class_idx]
        score.backward()

        # gradients & activations: (1, num_patches+1, hidden_dim)
        grads = self.gradients[0]        # (num_patches+1, hidden_dim)
        acts  = self.activations[0]      # (num_patches+1, hidden_dim)

        # Drop the [CLS] token (index 0), keep patch tokens
        grads = grads[1:]   # (196, hidden_dim)
        acts  = acts[1:]    # (196, hidden_dim)

        # Weight activations by mean gradient
        weights = grads.mean(dim=-1)             # (196,)
        cam     = (weights.unsqueeze(-1) * acts).sum(dim=-1)  # (196,)
        cam     = F.relu(cam)

        # Reshape to spatial grid: 196 = 14×14 patches
        cam = cam.reshape(14, 14).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize heatmap to original image dimensions
        h, w  = original_image.shape[:2]
        cam   = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        return overlay


# ─────────────────────────── Inference ───────────────────────────

def preprocess(image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """
    Converts a PIL image to model input tensor + keeps BGR numpy for Grad-CAM.

    Returns:
        tensor       : (1, 3, 224, 224) float32
        original_bgr : (224, 224, 3) uint8 BGR numpy array
    """
    image_rgb = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    original_bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)

    tensor = get_transform()(image_rgb).unsqueeze(0)  # add batch dim
    return tensor, original_bgr


def predict(image: Image.Image, top_k: int = 4) -> PredictionResult:
    """
    Full inference pipeline: preprocess → forward pass → Grad-CAM.

    Args:
        image : PIL Image (any size, any mode)
        top_k : number of top predictions to return

    Returns:
        PredictionResult dataclass
    """
    model  = load_model()
    gradcam = GradCAM(model)

    tensor, original_bgr = preprocess(image)

    # Forward pass
    with torch.no_grad():
        logits = model(pixel_values=tensor.to(DEVICE)).logits
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # Top-k predictions
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_k_results = [
        {
            "class":      CLASS_NAMES[i],
            "full_name":  CLASS_INFO[CLASS_NAMES[i]]["full_name"],
            "confidence": float(probs[i]),
        }
        for i in top_indices
    ]

    predicted_class = CLASS_NAMES[top_indices[0]]
    confidence      = float(probs[top_indices[0]])

    # Grad-CAM (needs gradients — run outside no_grad)
    gradcam_image = gradcam.generate(tensor, int(top_indices[0]), original_bgr)

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        top_k=top_k_results,
        gradcam_image=gradcam_image,
    )