import os
import subprocess
import sys

def main():
    # Detect the best Python interpreter
    # Prefer local venv if it exists
    python_exe = sys.executable
    venv_paths = [
        os.path.join(".venv311", "Scripts", "python.exe"),
        os.path.join("env", "Scripts", "python.exe"),
        os.path.join(".venv", "Scripts", "python.exe"),
    ]
    for venv in venv_paths:
        if os.path.isfile(venv):
            python_exe = os.path.abspath(venv)
            break

    # Potential dataset locations
    candidates = [
        "../use_this/archive/dataset_split_with_odir",
        "../use_this/archive/dataset",
        "./prepared_dataset",
        "./prepared_dataset_odir"
    ]
    
    data_dir = None
    for cand in candidates:
        if os.path.isdir(cand):
            data_dir = cand
            break
    
    if not data_dir:
        print(f"Error: Could not find dataset directory. Tried: {', '.join(candidates)}")
        print("You can run 'python preprocess_dataset.py' first.")
        return

    # Optimized command for >80% accuracy
    is_cuda = subprocess.run(["nvidia-smi"], capture_output=True, shell=True).returncode == 0
    if not is_cuda:
        print("Warning: No GPU detected. Training on CPU will be slow.")
        print("Reduced default epochs to 20 to save time.")
    
    cmd = [
        python_exe,
        "training_pipeline.py",
        "--data_dir", data_dir,
        "--epochs", "20",
        "--batch_size", "8" if not is_cuda else "16",
        "--device", "cuda" if is_cuda else "cpu",
        "--robust_aug",  
        "--mixup_alpha", "0.2",
        "--label_smoothing", "0.1",
        "--focal_gamma", "2.0",  
        "--output", "weights/eyenet_ensemble_optimized.pth",
        "--patience", "6",
        "--head_lr", "1e-3",
        "--backbone_lr", "2e-4"
    ]

    print("Starting optimized EyeNet Elite training...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
    except KeyboardInterrupt:
        print("Training interrupted by user.")

if __name__ == "__main__":
    main()
