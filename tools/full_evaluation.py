import torch
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from torch.utils.data import DataLoader
from backend.models.model import build_model
from backend.services.data_pipeline import RetinalDataset, collect_samples
from backend.config import CLASS_NAMES, IMAGE_SIZE

def run_evaluation():
    device = torch.device("cpu")
    weights_path = "weights/eyenet_ensemble_optimized.pth"
    data_dir = "../use_this/archive/dataset_split_with_odir/test"
    
    # 1. Load Model
    model = build_model(pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state['state_dict'] if 'state_dict' in state else state)
    model.to(device)
    model.eval()
    
    # 2. Load Data
    samples = collect_samples(data_dir)
    dataset = RetinalDataset(samples, training=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    # 3. Inference
    glaucoma_idx = CLASS_NAMES.index("Glaucoma") if "Glaucoma" in CLASS_NAMES else 1
    SAFETY_BIAS = 1.5 # Clinical sensitivity boost for Glaucoma
    
    print(f"Running evaluation with Clinical Bias (Glaucoma Bias: {SAFETY_BIAS})...")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            
            # Apply Sensitivity Bias to Glaucoma
            biased_probs = probs.clone()
            biased_probs[:, glaucoma_idx] *= SAFETY_BIAS
            
            # Choose best class based on biased probabilities
            preds = torch.argmax(biased_probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()) # Store original probs for AUC-ROC
            
    # 4. Metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, output_dict=True)
    
    # Calculate Macro AUC-ROC
    try:
        auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc_roc = 0.0

    # Format for output
    results = {
        "confusion_matrix": cm.tolist(),
        "accuracy": report['accuracy'],
        "macro_avg": report['macro avg'],
        "auc_roc": auc_roc,
        "per_class": {name: report[name] for name in CLASS_NAMES}
    }
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_evaluation()
