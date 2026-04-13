import torch
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services import prediction_service
from backend.config import CLASS_NAMES, PROJECT_ROOT

def run_system_audit():
    print("Starting Full AI Pipeline Audit...")
    print("-" * 50)

    # 1. Environment Check
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # 2. Weights Parity Audit
    weights_path = PROJECT_ROOT / "weights" / "eyenet_ensemble.pth"
    print(f"Checking weights at: {weights_path}")
    
    if not weights_path.exists():
        print("CRITICAL ERROR: Weights file not found!")
        return

    print("Attempting STRICT Model Load...")
    model = prediction_service.load_model(str(weights_path))
    
    if model is None:
        print("CRITICAL ERROR: Model failed to load (Architecture mismatch).")
        return
    print("SUCCESS: Model loaded successfully (100% layer parity confirmed).")

    # 3. Dynamic Inference Check
    print("\nRunning Dynamic Range Analysis...")
    # Generate 5 random "images" and check if it predicts the same thing every time
    results = []
    for i in range(5):
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)
            pred = output.argmax(1).item()
            results.append(pred)
    
    unique_preds = len(set(results))
    if unique_preds == 1:
        print(f"Warning: Model returned class {results[0]} for 5 random inputs. (Potential Bias)")
    else:
        print(f"SUCCESS: Model shows variability across random noise ({unique_preds} different classes).")

    # 4. Class Mapping Verification
    print(f"\nVerifying Class Mapping:")
    print(f"Configuration: {CLASS_NAMES}")
    expected_order = ["Diabetic Retinopathy", "Glaucoma", "Cataract", "Normal"]
    if CLASS_NAMES == expected_order:
        print("SUCCESS: Mapping matches verified Training order.")
    else:
        print(f"ERROR: Mapping MISMATCH! Expected {expected_order}")

    print("-" * 50)
    print("Pipeline audit finished successfully.")

if __name__ == "__main__":
    run_system_audit()
