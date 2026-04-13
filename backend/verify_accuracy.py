import torch_bootstrap
torch_bootstrap.prepare_torch_environment()
import torch
import os
from models.model import EyeNetEnsemble
from torch.utils.data import DataLoader
from backend.config import CLASS_SLUGS, NUM_CLASSES
from backend.services.data_pipeline import RetinalDataset, collect_samples

def run_local_test():
    # 1. Load your trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EyeNetEnsemble(pretrained=False, num_classes=NUM_CLASSES).to(device)
    
    # LOAD YOUR NEW WEIGHTS HERE
    weights_path = "./weights/eyenet_ensemble.pth"
    state = torch.load(weights_path, map_location=device)
    payload = state.get("state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(payload, strict=True)
    model.eval()

    # 2. Setup Test Data
    # Updated path based on discovery
    dataset_path = os.path.abspath("../../use_this/archive/dataset")
    if not os.path.exists(dataset_path):
        # Alternative attempt
        dataset_path = os.path.abspath("../dataset")
        
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return

    print(f"📂 Loading dataset from: {dataset_path}")
    test_samples = collect_samples(dataset_path)
    if not test_samples:
        print(f"❌ Error: No test images found under class folders {CLASS_SLUGS}")
        return
    test_dataset = RetinalDataset(test_samples, training=False)
    
    # Use a subset if dataset is too large, or just test full
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    # 3. Predict & Compare
    correct = 0
    total = min(100, len(test_dataset)) # Test on 100 images for speed
    
    print(f"🔬 Starting Accuracy Test on {total} samples...")
    
    model.eval()
    with torch.no_grad():
        count = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            count += images.size(0)
            if count >= total:
                break

    accuracy = (correct / count) * 100
    print(f"\n✅ FINAL TEST ACCURACY: {accuracy:.2f}%")
    print(f"Correct: {correct}/{count}")

if __name__ == "__main__":
    run_local_test()
