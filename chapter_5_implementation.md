# 5. IMPLEMENTATION

## 5.1 Dataset Acquisition

The EyeNet system is developed using labeled retinal fundus images collected from publicly available datasets. The finalized robust dataset contains four distinct categories representing conditions such as diabetic retinopathy, glaucoma, cataract, and normal retina. 

To ensure precision during model processing and evaluation, the dataset is organized into structured, class-wise folders where each directory maps strictly to its specific disease category. *(Note: Age-related macular degeneration (AMD) was intentionally removed from the final taxonomy to improve boundary-class specificity and precision metrics.)*

## 5.2 Dataset Splitting & K-Fold Strategy

The dataset is divided using a highly structured Stratified K-Fold Cross-Validation approach to ensure unbiased data distribution and statistically significant generalization. 

Rather than a static split, the data is dynamically segmented. During each fold, the data mimics a 70:15:15 equivalent representation across training, validation, and testing logic. The splitting process is strictly stratified in a class-wise manner using fixed random seeds (`42`) to guarantee equal representation of minority disease categories and absolute reproducibility. 

## 5.3 Image Preprocessing and Enhancement

Before feeding images into the deep learning pipeline, several domain-specific preprocessing steps are applied to standardise inputs and isolate pathological markers.

All images undergo a circular fundus crop padding process before being resized to a high-resolution `256x256` dimension to ensure optimal vascular detail retention. Pixel channels are then normalized using ImageNet statistical means to accelerate stable gradient convergence.

Crucially, Contrast Limited Adaptive Histogram Equalization (CLAHE) is conditionally applied exclusively to the Lightness (L) channel within the LAB color space. A subsequent unsharp masking algorithm is utilized to further boost the structure of retinal features such as microaneurysms, blood vessels, and exudates without destructively clipping the original anatomical structures. 

## 5.4 Advanced Data Augmentation

To heavily prevent over-regularization while maintaining dataset diversity, targeted Albumentations augmentations are applied during internal training batches.

This includes rotation limits restricted to realistic clinical boundaries ($\pm 15^\circ$), along with shift, scale, and horizontal flipping. Furthermore, `CoarseDropout` (Random Erasing) and structural `GaussNoise` operations are sparsely introduced to artificially compel the network to learn invariant, occlusion-resistant features across varied lighting conditions. 

## 5.5 Model Development (EyeNet Architecture Integration)

The EyeNet model utilizes an advanced Convolutional Neural Network (CNN) Ensemble strategy, unifying ResNet50, EfficientNetB0, and DenseNet121 architectures into a single inference graph. 

Rather than simple output averaging, EyeNet mathematically concatenates the high-level feature vectors extracted from the global average pooling dimensions of each backbone network resulting in a dense `4352`-dimensional feature map.

This map is passed into a **Learned Fusion Head**. The fusion network passes the signal through a deep `1024 -> 512 -> 256` pipeline, heavily constrained by `nn.BatchNorm1d` layers and tuned `nn.Dropout()` gates (`0.3, 0.2, 0.1` respectively). 

## 5.6 Training and Validation Mechanisms

The model is driven by the state-of-the-art **AdamW optimizer**, explicitly decoupling weight-decay (`1e-2`) from gradient updates to vastly improve structural generalization on complex medical datasets. 

The primary cost framework utilizes a fully Weighted `CrossEntropyLoss` metric to naturally counteract severe class imbalances, avoiding the disruptive underfitting bounds typical of Focal Loss on small datasets. Additionally, `ReduceLROnPlateau` scheduling maps the optimization pace, rapidly dropping the `1e-4` baseline learning rate by half when validation stagnation is detected over three consecutive epochs.

## 5.7 Explainability & Real-Time Diagnostics

The fully trained system is deployed as a high-concurrency FastAPI service with a synchronized **React 18** frontend interface, completely isolating complex background PyTorch operations from the clinical user.

Upon fundus scan upload, the system returns a Softmax-calibrated probability array alongside real-time **Gradient-weighted Class Activation Mapping (Grad-CAM)**. Grad-CAM programmatically traces prediction weights backward through the `layer4` conv-blocks of the ResNet backbone, extracting exact spatial derivatives.

These derivatives are projected as a high-fidelity "Turbo" heatmap over the original image, highlighting the anatomical regions (e.g., optic disc, lesions) that influenced the network's classification. This implementation guarantees AI transparency and builds critical medical trust for clinical presentations.
