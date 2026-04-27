import torch
import json

def check_progress():
    path = "d:/Drive_Folder/final_year_project/project/weights/eyenet_ensemble_optimized_last.pth"
    try:
        data = torch.load(path, map_location='cpu')
        history = data.get('history', [])
        if not history:
            print("No history found in checkpoint.")
            return
        
        last_epoch = history[-1]
        print(json.dumps(last_epoch, indent=2))
        
        best_f1 = data.get('best_macro_f1', 0.0)
        print(f"Best Validation Macro F1 so far: {best_f1}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    check_progress()
