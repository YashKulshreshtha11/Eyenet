import torch
import os

weights_path = r"d:\Drive_Folder\final_year_project\project\weights\eyenet_ensemble.pth"
if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    print(f"Total keys: {len(state_dict)}")
    for key in sorted(state_dict.keys()):
        if "weight" in key:
            print(f"{key}: {state_dict[key].shape}")
else:
    print("File not found")
