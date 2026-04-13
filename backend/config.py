from pathlib import Path

APP_TITLE = "EyeNet - Retinal Disease Classification Suite"
APP_VERSION = "2.0.0"
API_PREFIX = "/api/v1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "eyenet_ensemble_optimized.pth"
DEFAULT_BACKEND_HOST = "0.0.0.0"
DEFAULT_BACKEND_PORT = 8000

# ── Model Architecture ────────────────────────────────────────────────────────
MODEL_NAME = "EyeNet Elite Ensemble (ResNet50 + EffNetB0 + DenseNet121)"
IMAGE_SIZE = 256
NUM_CLASSES = 4

# STRICT Class Mapping — MUST MATCH TRAINING
# 0: DR, 1: Glaucoma, 2: Cataract, 3: Normal
CLASS_NAMES = [
    "Diabetic Retinopathy",
    "Glaucoma",
    "Cataract",
    "Normal",
]

CLASS_SLUGS = [
    "diabetic_retinopathy",
    "glaucoma",
    "cataract",
    "normal",
]

# ── Preprocessing & Augmentation ──────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PREPROCESSING_STEPS = [
    "Circular Fundus Cropping",
    "LAB Space CLAHE Enhancement",
    "Unsharp Mask Sharpening",
    "Bicubic Resizing to 256x256",
    "ImageNet Normalization",
]

# ── Stability & Anomaly Detection ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.65
DOMINANCE_THRESHOLD = 0.90
CONSECUTIVE_PREDICTION_LIMIT = 5

# Screening: require this minimum softmax mass on "Normal" before accepting a healthy label.
# Below this, the API prefers the strongest disease class to reduce false negatives.
NORMAL_CLASS_INDEX = 3
NORMAL_MIN_CONFIDENCE_FOR_HEALTHY = 0.35

# Training loss: extra multiplier on inverse-frequency weights for these indices (DR, Glaucoma).
CLASS_WEIGHT_DR_BOOST = 1.35
CLASS_WEIGHT_GLAUCOMA_BOOST = 1.35

ANTI_OVERFITTING_FEATURES = [
    "OneCycleLR Scheduling",
    "Label Smoothing (0.1)",
    "Weighted Random Sampling",
    "Dropout (0.3-0.5)",
    "LayerNorm Fusion",
    "Weight decay with AdamW",
    "Early stopping on validation macro F1",
    "Gradient clipping",
]

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
}
