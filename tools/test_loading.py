import torch
import sys
import os

# Add backend to path
sys.path.insert(0, r"d:\Drive_Folder\final_year_project\project")

from backend.models.model import build_model

def test_loading():
    device = torch.device("cpu")
    model = build_model(pretrained=False)
    weights_path = r"d:\Drive_Folder\final_year_project\project\weights\eyenet_ensemble.pth"
    
    if os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        print(f"MISSING KEYS ({len(missing)}):")
        print(missing[:10])
        print("\nUNEXPECTED KEYS ({len(unexpected)}):")
        print(unexpected[:10])
    else:
        print("Weights not found")

if __name__ == "__main__":
    test_loading()
