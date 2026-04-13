from __future__ import annotations

import hashlib
import io
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from backend.torch_bootstrap import prepare_torch_environment

prepare_torch_environment()

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from backend.config import CLASS_SLUGS, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from backend.services.fundus_ops import preprocess_fundus_bgr, read_image_bgr

Sample = Tuple[str, int]


class RandomJPEGCompression:
    """
    Simulate web-style JPEG artifacts deterministically per-sample randomness.

    Note: this is TRAINING-ONLY and is off by default via robust_aug=False.
    """

    def __init__(self, p: float = 0.25, quality_range: Tuple[int, int] = (45, 95)):
        self.p = float(p)
        qmin, qmax = quality_range
        self.quality_range = (int(min(qmin, qmax)), int(max(qmin, qmax)))

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        # torchvision pipelines pass PIL Image at this stage
        try:
            from PIL import Image
        except Exception:
            return img

        if not hasattr(img, "save"):
            return img

        q = random.randint(self.quality_range[0], self.quality_range[1])
        buf = io.BytesIO()
        try:
            img.save(buf, format="JPEG", quality=q, optimize=True)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        except Exception:
            return img


class RandomGamma:
    """Small gamma perturbation to mimic camera pipeline differences."""

    def __init__(self, p: float = 0.20, gamma_range: Tuple[float, float] = (0.85, 1.20)):
        self.p = float(p)
        gmin, gmax = gamma_range
        self.gamma_range = (float(min(gmin, gmax)), float(max(gmin, gmax)))

    def __call__(self, img):
        if random.random() >= self.p:
            return img
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        try:
            from torchvision.transforms import functional as TF
        except Exception:
            return img
        try:
            return TF.adjust_gamma(img, gamma=gamma, gain=1.0)
        except Exception:
            return img


def get_train_transform(*, robust_aug: bool = False) -> transforms.Compose:
    """
    Optimized training transform for EyeNet Elite.
    Uses RandAugment for state-of-the-art generalization while preserving 
    the fundus-specific preprocessing requirements.
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            # Base resize to match config
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            
            # RandAugment: learns the best augmentation policy automatically
            transforms.RandAugment(num_ops=2, magnitude=9),
            
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            
            # Subtle fundus-safe jitter
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value="random"),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class RetinalDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        training: bool = False,
        robust_aug: bool = False,
        cache_dir: str | Path | None = None,
    ):
        self.samples = list(samples)
        self.transform = get_train_transform(robust_aug=robust_aug) if training else get_eval_transform()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, image_path: str) -> Path:
        digest = hashlib.md5(image_path.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.png"

    def _load_preprocessed_rgb(self, image_path: str) -> np.ndarray:
        cache_path = self._cache_path(image_path) if self.cache_dir else None
        if cache_path and cache_path.exists():
            cached = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
            if cached is not None:
                return cv2.cvtColor(cached, cv2.COLOR_BGR2RGB)

        image_bgr = read_image_bgr(image_path)
        processed_bgr = preprocess_fundus_bgr(image_bgr)
        if cache_path:
            cv2.imwrite(str(cache_path), processed_bgr)
        return cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image_rgb = self._load_preprocessed_rgb(image_path)
        return self.transform(image_rgb).float(), label


def _iter_class_dirs(root_dir: Path) -> Iterable[Tuple[Path, int]]:
    for class_index, class_slug in enumerate(CLASS_SLUGS):
        direct = root_dir / class_slug
        if direct.is_dir():
            yield direct, class_index
            continue
        fallback = next(
            (candidate for candidate in root_dir.iterdir() if candidate.is_dir() and candidate.name.lower() == class_slug.lower()),
            None,
        )
        if fallback:
            yield fallback, class_index


def collect_samples(root_dir: str | Path) -> List[Sample]:
    root_path = Path(root_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    samples: List[Sample] = []
    if not root_path.exists():
        return samples

    for class_dir, class_index in _iter_class_dirs(root_path):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in valid_exts:
                samples.append((str(image_path), class_index))
    return samples


def detect_pre_split_dataset(root_dir: str | Path) -> bool:
    root_path = Path(root_dir)
    return (root_path / "train").is_dir() and (root_path / "val").is_dir()


def stratified_split(
    samples: Sequence[Sample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    by_class: Dict[int, List[Sample]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    test_samples: List[Sample] = []

    for items in by_class.values():
        rng.shuffle(items)
        total = len(items)
        train_end = max(1, int(total * train_ratio))
        val_end = train_end + max(1, int(total * val_ratio))
        train_samples.extend(items[:train_end])
        val_samples.extend(items[train_end:val_end])
        test_samples.extend(items[val_end:])

    return train_samples, val_samples, test_samples


def build_weighted_sampler(samples: Sequence[Sample]) -> WeightedRandomSampler:
    labels = [label for _, label in samples]
    class_counts = np.bincount(labels)
    sample_weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def class_distribution(samples: Sequence[Sample]) -> Dict[int, int]:
    distribution: Dict[int, int] = {}
    for _, label in samples:
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def drop_filename_label_conflicts(samples: Sequence[Sample]) -> List[Sample]:
    """
    Drop ambiguous samples where the same filename appears with different labels.
    This protects training from obvious cross-class duplicate leakage.
    """
    by_name: Dict[str, set[int]] = {}
    for image_path, label in samples:
        by_name.setdefault(Path(image_path).name.lower(), set()).add(label)
    conflict_names = {name for name, labels in by_name.items() if len(labels) > 1}
    if not conflict_names:
        return list(samples)
    return [
        sample
        for sample in samples
        if Path(sample[0]).name.lower() not in conflict_names
    ]


def export_preprocessed_dataset(source_dir: str | Path, output_dir: str | Path) -> int:
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    processed_count = 0

    for image_path in source_path.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in valid_exts:
            continue

        relative_path = image_path.relative_to(source_path).with_suffix(".png")
        destination = output_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        image_bgr = read_image_bgr(image_path)
        processed_bgr = preprocess_fundus_bgr(image_bgr)
        cv2.imwrite(str(destination), processed_bgr)
        processed_count += 1

    return processed_count
