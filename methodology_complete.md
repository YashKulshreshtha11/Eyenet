# 📔 EyeNet: Complete System Methodology

This document summarizes the exact scientific and technical methodologies used in the finalized version of the **EyeNet Elite Retinal Diagnostics System**.

---

## 1. Dataset Strategy
### 1.1 Data Acquisition & Taxonomy
The system is optimized for **4-class classification**:
1.  **Diabetic Retinopathy**
2.  **Glaucoma**
3.  **Cataract**
4.  **Normal (Healthy)**

### 1.2 Dataset Segmentation
The pipeline utilizes a **Stratified 70:15:15 Split**. This guarantees that each disease class is equally represented across the **Training (70%)**, **Validation (15%)**, and **Testing (15%)** sets, ensuring statistically valid performance metrics (Precision, Recall, F1) without class-frequency bias.

---

## 2. Advanced Preprocessing Pipeline
To maximize high-level feature extraction, all fundus scans undergo the following automated sequence:
1.  **Circular Masking:** Isolates the retinal area from the black periorbital background.
2.  **Spatial Standardization:** Input images are resized to a high-resolution **`256x256`** grid.
3.  **Channel-Wise CLAHE:** Contrast Limited Adaptive Histogram Equalization is applied to the L-channel (LAB space) to accentuate subtle microaneurysms and vascular abnormalities.
4.  **Unsharp Masking:** A subtle sharpening filter is applied to resolve fine edges in the retinal nerve fiber layer (RNFL).

---

## 3. Deep Learning Architecture 
EyeNet employs a **Joint-Feature Learning Ensemble** strategy:
-   **Multi-Backbone Integration:** Combines **ResNet50**, **EfficientNetB0**, and **DenseNet121** architectures simultaneously.
-   **Feature Concatenation:** High-dimensional feature maps from all three backbones are combined into a single `4352`-dimensional vector.
-   **Learned Fusion Head:** A multi-layered dense network (**1024 -> 512 -> 256 -> 4**) processes the fused features using **Batch Normalization** and **Dropout** for robust generalization.

---

## 4. Explainable AI (XAI) Module
To ensure clinical transparency, EyeNet implements a **Grad-CAM (Gradient-weighted Class Activation Mapping)** engine:
-   **Layer Mapping:** Attachments are made to the final convolutional blocks of the ResNet50 backbone.
-   **Visual Transparency:** A custom alpha-blending engine overlays a "Turbo" colormap heatmap on the original scan. This physically highlights the anatomical regions (e.g., hemorrhages, optic disc) that influenced the AI's diagnosis, building critical medical trust.

---

## 5. Technical Stack
-   **Deep Learning Framework:** PyTorch & Torchvision
-   **Back-End Engine:** FastAPI (Parallelized Uvicorn server)
-   **Database Layer:** MongoDB Local/Cloud (Record persistence & clinical logging)
-   **Front-End UI:** Premium React 18 Dashboard (Vite-optimized).
-   **Animation & UX:** Framer Motion & Lucide Icons.
-   **Typography:** Syncopate (Branding), Sora (Headings), Plus Jakarta Sans (Body).

---
*Developed for Advanced Retinal Research and Academic Evaluation.*