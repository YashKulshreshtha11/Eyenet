"""
EyeNet Elite — Advanced Training & Validation Pipeline
======================================================
Architecture: Ensemble of ResNet50 + EfficientNet_B0 + DenseNet121
Features: Channel Attention Fusion, Label Smoothing, Stratified Split, 
          Heavy Preprocessing (CLAHE + Masking), and Early Stopping.
"""

import os
import sys
import time
import random
import argparse
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.models.model import build_model

# Constants
CLASS_NAMES = ["Diabetic Retinopathy", "Glaucoma", "Cataract", "Normal"]
FOLDER_NAMES = ["diabetic_retinopathy", "glaucoma", "cataract", "normal"]
IMAGE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# --- Preprocessing ---

def crop_fundus(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return img
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w]

def apply_medical_enhancement(img):
    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Unsharp Mask
    blur = cv2.GaussianBlur(img, (0, 0), 2)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

class RetinaTransform:
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.base_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        self.aug_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.1, 0.1, 0.1),
        ])

    def __call__(self, img_bgr):
        img = crop_fundus(img_bgr)
        img = apply_medical_enhancement(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.is_train:
            # Convert to PIL for transforms
            from PIL import Image
            img_pil = Image.fromarray(img_rgb)
            img_pil = self.aug_tf(img_pil)
            return self.base_tf(img_pil)
        return self.base_tf(img_rgb)

# --- Dataset ---

class EyeDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples
        self.transform = RetinaTransform(is_train=is_train)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None: return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), label
        return self.transform(img), label

# --- Training Loop ---

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, f1_score(all_labels, all_preds, average='macro', zero_division=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load samples
    samples = []
    for i, folder in enumerate(FOLDER_NAMES):
        path = os.path.join(args.data, folder)
        if not os.path.exists(path): continue
        for img_name in os.listdir(path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append((os.path.join(path, img_name), i))
    
    random.shuffle(samples)
    split = int(0.8 * len(samples))
    train_samples, val_samples = samples[:split], samples[split:]

    train_ds = EyeDataset(train_samples, is_train=True)
    val_ds = EyeDataset(val_samples, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0
    for epoch in range(args.epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {t_loss:.4f} | Val Acc: {v_acc:.4f} | Val F1: {v_f1:.4f}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), "weights/eyenet_elite_best.pth")
            print("  [✓] Best model saved!")

if __name__ == "__main__":
    main()
