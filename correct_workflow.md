# Corrected Workflow Documentation - EyeNet Project

## 4.2 Collecting and Preparing Data

In the next step, data collection and preparation was done, which plays a major role in model performance. Two main datasets were used, one from Kaggle eye diseases classification dataset and another ODIR5K dataset. Both datasets were combined to create a larger and more useful dataset, although merging them was not completely straightforward. Some balancing techniques were applied so that all classes get almost equal representation. The dataset was then divided into training, validation and testing sets.

After that, preprocessing was performed where all images were resized to **256x256 dimensions** (not 224x224), and advanced enhancement techniques were applied including **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for brightness and contrast adjustment, and **unsharp masking** for noise reduction and sharpening. Circular fundus cropping was also applied to focus on the retinal region. **Data cleaning for corrupted images and duplicate removal was not implemented** in the current codebase.

## 4.3 Exploring the Data

Before training the model, data exploration was done to better understand the dataset. Visual analysis helped in identifying how diseases appear in images and which regions are important, like optic disc, macula and blood vessels. Some disease specific patterns like hemorrhages and exudates were also observed.

Statistical analysis was also performed to check class distribution and image quality variations. Since the dataset was not perfectly balanced, data augmentation techniques were applied like rotation, flipping, brightness changes and scaling. These methods helped in improving the dataset size and diversity, but still it was not completely perfect.

## 4.4 Feature Learning, Model Building

After data ready we start make model. CNN thing is used to find patterns in fundus images. Try few different architechtures also, see which one catch disease more better. Sometimes one work good sometime not.

## 4.5 Training and Validation

Model train with all dataset together then test on new images, it never seen before, we use kfold or some other ways to make sure model really learn something not just guessing.

## 4.6 Testing and Interpretation

When the model starts producing consistent outputs, it is tested more thoroughly. The results are analyzed to make sure the system identifies diseases reliably and not by chance.

## 4.7 Final Integration & Documentation

Last step put every thing together simple way, write report, draw some flowchart, make system ready to show, nothing too perfect.

---

## **Actual Implementation Details**

### **Real Preprocessing Pipeline (Runtime)**

**File**: `backend/services/fundus_ops.py`
```python
def preprocess_fundus_bgr(image_bgr: np.ndarray) -> np.ndarray:
    image_bgr = crop_fundus_circle(image_bgr)      # Circular cropping
    image_bgr = apply_clahe(image_bgr)             # CLAHE enhancement  
    image_bgr = apply_unsharp_mask(image_bgr)     # Sharpening
    return image_bgr
```

**File**: `backend/services/vision.py`
```python
def get_base_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),            # 256x256 (not 224x224)
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
```

### **Actual Preprocessing Steps**
1. **Circular Fundus Cropping**: Extract circular retinal region
2. **CLAHE Enhancement**: Adaptive histogram equalization in LAB space
3. **Unsharp Mask**: Gaussian blur + weighted addition for sharpening
4. **Resize to 256x256**: Runtime resizing (not 224x224)
5. **ImageNet Normalization**: Standard mean/std normalization

### **Quality Assessment (Implemented)**
**File**: `backend/services/vision.py` - `evaluate_image_quality()`
- Brightness, contrast, sharpness analysis
- Fundus coverage and circularity detection
- Quality verdict: "Excellent", "Good", "Needs Review"

### **NOT Implemented**
- Corrupted image removal
- Duplicate image detection
- Label verification
- 224x224 resizing (uses 256x256)
- Pre-training data preprocessing (all runtime)

### **Model Architecture (Actually Implemented)**
- **Ensemble CNN**: ResNet-18 + EfficientNet-B0 + DenseNet-121 + Custom CNN
- **Input Size**: 256x256 (config.py: IMAGE_SIZE = 256)
- **Classes**: 4 (Diabetic Retinopathy, Glaucoma, Cataract, Normal)
- **Explainability**: Grad-CAM with ensemble averaging

### **Training Features (Actually Implemented)**
- **Augmentation**: Rotation, flipping, brightness, scaling
- **Loss Function**: Binary cross-entropy for multi-label
- **Optimization**: Adam optimizer with learning rate scheduling
- **Validation**: K-fold cross-validation

This corrected workflow reflects the actual implementation in your EyeNet codebase rather than theoretical descriptions.
