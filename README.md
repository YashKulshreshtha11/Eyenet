# 👁️ EyeNet Elite — Advanced Retinal Diagnostics System

[![Ensemble Model](https://img.shields.io/badge/Model-Ensemble%20(ResNet%2BEffNet%2BDenseNet)-blueviolet)](https://github.com/)
[![UI](https://img.shields.io/badge/UI-Premium%20React%20App-teal)](https://github.com/)
[![Status](https://img.shields.io/badge/Final%20Year%20Project-Ready-success)](https://github.com/)

**EyeNet Elite** is a premium, state-of-the-art medical AI platform designed for the automated classification of retinal fundus images into four primary diagnostic categories: **Diabetic Retinopathy, Glaucoma, Cataract, and Normal.**

## 🌟 Key Features
- **Ultra-Performance Ensemble**: Combines ResNet50, EfficientNet-B0, and DenseNet121 with a learned Feature Fusion head for clinically reliable accuracy.
- **Explainable AI (XAI)**: High-fidelity Grad-CAM heatmaps highlight pathological markers (exudates, optic disc cupping) directly on the fundus scan.
- **Fabulous UI/UX**: A bespoke, glassmorphic React dashboard featuring **fluid responsiveness**, infinite scroll history, and **Syncopate/Sora** professional typography.
- **Medical-Grade Preprocessing**: Circular fundus cropping, CLAHE enhancement, and high-pass sharpening optimized for retinal vascular detail.
- **Production-Ready Backend**: High-concurrency FastAPI server with MongoDB persistence for secure diagnostic record-keeping.

## 🛠 Technology Stack
- **Deep Learning**: PyTorch, Torchvision, Albumentations
- **Backend Service**: FastAPI, Uvicorn, Python 3.11
- **UI Architecture**: Vite, React 18, Framer Motion (Animations), Lucide (Icons)
- **Typography & Brand**: Syncopate (Logo), Sora (Headings), Plus Jakarta Sans (Body)
- **Database Layer**: MongoDB (Local or Atlas)

---

## 🏗️ Project Architecture

```
project/
├── backend/
│   ├── app_server.py            ← FastAPI main entry + history routing
│   ├── training_pipeline.py     ← Core EyeNet Architecture + TTA logic
│   └── models/
│       └── schema.py            ← Pydantic response models
├── frontend/
│   ├── src/
│   │   ├── App.jsx              ← Main UI Hub (Analysis, History tabs)
│   │   └── index.css            ← Premium Glassmorphism Design System
│   └── public/                  ← High-res assets & icons
├── database/
│   └── db.py                    ← MongoDB Atlas/Local connection logic
├── weights/                     ← eyenet_ensemble.pth (Optimized weights)
├── requirements.txt             ← Full dependency locked-file
├── run_venv.bat                 ← Automated Windows launcher
└── PROJECT_GUIDE.md             ← Developer implementation notes
```

---

## ⚡ Quick Start

### 1 — Environment Setup
```bash
# Install backend dependencies
pip install -r requirements.txt

# Start the full stack (FastAPI + Vite proxy)
run_venv.bat
```

### 2 — Manual Launch
**Backend:**
```bash
python -m uvicorn backend.app_server:app --port 8001 --reload
```
**Frontend:**
```bash
cd project/frontend
npm install
npm run dev
```

---

## 🌐 AI Diagnostics API

Base URL: `http://localhost:8000/api/v1`  
Swagger Documentation: `http://localhost:8000/docs`

### `POST /predict`
Upload a fundus image for real-time neural analysis.

**Response Schema:**
```json
{
  "predicted_class":   "Glaucoma",
  "probabilities": {
    "Diabetic Retinopathy": 0.04,
    "Glaucoma":             0.92,
    "Cataract":             0.02,
    "Normal":               0.02
  },
  "heatmap_base64":    "<Spatial activation PNG>",
  "inference_time_ms": 342.0
}
```

---

## 🧠 Diagnostic Methodology

EyeNet utilizes a **Joint-Feature Learning Ensemble** strategy:
1. **CLAHE Augmentation**: Contrast-limited adaptive enhancement isolates microaneurysms.
2. **Backbone Fusion**: Concatenates ResNet50 (residual features), EfficientNetB0 (efficient scaling), and DenseNet121 (dense connections).
3. **Learned Head**: A shallow MLP network performs final inference on the combined feature-mesh.
4. **Grad-CAM**: Gradient-weighted activations provide clinical transparency by visualizing the model's "focus" area.

---

## 🗄️ MongoDB Configuration

Edit `.env`:

```env
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE_NAME=eyenet_db
MONGODB_LOGS_COLLECTION=system_logs
```

The system **functions without MongoDB** — history/logs are simply disabled when the DB is unreachable.

---

## ⚠️ Disclaimer

EyeNet is a **research prototype** intended for academic study only.  
It is **NOT** a certified medical device and must **not** be used for clinical diagnosis.

---

## 📄 License

MIT — free to use, modify, and distribute with attribution.
