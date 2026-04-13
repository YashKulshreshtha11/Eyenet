# EyeNet Project Developer Guide

EyeNet is an elite retinal disease classification platform designed for high-impact project demonstrations and academic evaluation. The system features a synchronized **FastAPI + React** stack with an explainable AI core.

## 🛠️ Core Infrastructure
- **Engine**: `ResNet50 + EfficientNetB0 + DenseNet121` Feature-Fusion Ensemble.
- **Explainability**: Integrated Grad-CAM spatial activation mapping.
- **Frontend**: Vite-powered React 18 dashboard with Framer Motion animations.
- **Backend**: FastAPI with async MongoDB persistence.

## 📁 Key File Map
- **UI Logic**: `project/frontend/src/App.jsx`
- **Design System**: `project/frontend/src/index.css`
- **App Server**: `backend/app_server.py`
- **Model Graph**: `backend/training_pipeline.py`
- **DB Helper**: `database/db.py`

## 🚀 Launch Procedures

### 1. Unified Launcher (Recommended)
```bash
# Windows
run_venv.bat
```

### 2. Individual Services
**Start Backend (Port 8001):**
```bash
python -m uvicorn backend.app_server:app --port 8000 --reload
```

**Start Frontend (Vite Dev Server):**
```bash
cd project/frontend
npm run dev
```

## 🏋️ Model Training & Evaluation
To retrain the ensemble or evaluate against a new test set:

```bash
python train.py \
  --data_dir path/to/dataset \
  --epochs 25 \
  --batch_size 16 \
  --device cuda \
  --output weights/eyenet_ensemble.pth
```

Outputs are automatically logged to the `reports/` directory with Confusion Matrices and F1-Score summaries.

---
**💡 Academic Note:** This project is built for transparency. Ensure that during the viva/presentation, the **Grad-CAM heatmaps** are used to explain the AI's decision-making process to the panel.
