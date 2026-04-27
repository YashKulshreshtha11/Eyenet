import torch
import os

weights_path = r"d:\Drive_Folder\final_year_project\project\weights\eyenet_ensemble.pth"
if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    keys = sorted(state_dict.keys())
    print("ATTENTION KEYS:")
    for k in keys:
        if "attention" in k:
            print(k)
    print("\nFC KEYS:")
    for k in keys:
        if "fc" in k and "." in k:
            print(k)
    print("\nFUSION KEYS:")
    for k in keys:
        if "fusion" in k:
            print(k)
else:
    print("File not found")
