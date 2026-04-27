# 👁️ EyeNet Elite — Advanced Retinal Diagnostics System

[![Model](https://img.shields.io/badge/Model-Ensemble%20(ResNet%2BEffNet%2BDenseNet)-blueviolet)](https://github.com/)
[![UI](https://img.shields.io/badge/UI-Premium%20React%20App-teal)](https://github.com/)
[![Status](https://img.shields.io/badge/Project-Production--Ready-success)](https://github.com/)

EyeNet Elite is a state-of-the-art medical AI platform designed for the automated classification of retinal fundus images. Using a sophisticated Joint-Feature Learning Ensemble, the system classifies scans into four primary diagnostic categories with clinical-grade accuracy.

---

## 📋 System Requirements

### Hardware
- **Minimum**: 8GB RAM, Quad-core CPU
- **Recommended**: 16GB RAM, NVIDIA GPU (8GB+ VRAM) for faster inference

### Software
- **Operating System**: Windows 10/11 (Optimized), Linux, or macOS
- **Python**: Version 3.10 or 3.11
- **Node.js**: Version 18.x or higher (for Frontend)
- **Database**: MongoDB (Optional, for history persistence)

---

## 🚀 Execution Instructions

### 1. Quick Start (Windows)
The project includes a unified launcher for convenience:
```bash
./run_venv.bat
```
*This script will automatically set up the virtual environment, start the FastAPI backend, and launch the React frontend.*

### 2. Manual Installation

#### Backend Setup
```bash
# Create and activate virtual environment
python -m venv env
.\env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn backend.app_server:app --port 8000 --reload
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 📊 Results & Evaluation
Performance metrics and diagnostic figures are located in the `/results` directory.

### Key Performance Snapshot:
- **Overall Accuracy**: 83.67%
- **Macro F1-Score**: 83.42%
- **ROC-AUC (Macro)**: 0.912

### Results Directory Contents:
- `classification_report.csv`: Detailed precision, recall, and F1-scores per class.
- `confusion_matrix.csv`: Raw prediction breakdown for error analysis.
- `training_curves.png`: Loss and accuracy progression during training.
- `test_confusion_matrix.png`: Visual representation of model performance on unseen data.

---

## 🏗️ Project Structure
```
project/
├── backend/            # FastAPI Server & AI Logic
├── frontend/           # React Dashboard (Vite)
├── results/            # Exported Excel sheets and figures
├── weights/            # Trained model checkpoints
├── database/           # MongoDB connection logic
└── requirements.txt    # Python dependencies
```

---

## 🧠 Diagnostic Methodology
EyeNet utilizes a **Joint-Feature Learning Ensemble** strategy:
1. **CLAHE Augmentation**: Contrast-limited adaptive enhancement isolates microaneurysms.
2. **Backbone Fusion**: Concatenates ResNet50, EfficientNetB0, and DenseNet121.
3. **Grad-CAM**: Gradient-weighted activations provide clinical transparency by visualizing the model's "focus" area.

---

## 📄 Disclaimer
EyeNet is a research prototype intended for academic study. It is **NOT** a certified medical device and must not be used for clinical diagnosis without professional medical supervision.
