from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from facenet_pytorch import MTCNN  # type: ignore
except Exception:
    MTCNN = None


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


mtcnn = MTCNN(keep_all=False, device=_default_device()) if MTCNN is not None else None

class SoftFaceMaskTransform:
    def __init__(self, mtcnn, blur=65, expand=1.5, bg_color=0):
        self.mtcnn = mtcnn
        self.blur = blur
        self.expand = expand
        self.bg_color = bg_color

    def __call__(self, img):
        # If optional deps aren't available, fall back to identity.
        if self.mtcnn is None or cv2 is None:
            return img

        w, h = img.size
        boxes, _ = self.mtcnn.detect(img)

        if boxes is None:
            return img

        x1, y1, x2, y2 = boxes[0]

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * self.expand, (y2 - y1) * self.expand

        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        axes = (int((x2 - x1) / 2), int((y2 - y1) / 2))

        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        mask = cv2.GaussianBlur(mask, (self.blur, self.blur), 0)
        mask = mask.astype(np.float32) / 255.0

        img_np = np.array(img).astype(np.float32)

        bg = np.ones_like(img_np) * self.bg_color
        masked = img_np * mask[..., None] + bg * (1 - mask[..., None])

        return Image.fromarray(masked.astype(np.uint8))


transform = transforms.Compose(
    [
        SoftFaceMaskTransform(mtcnn=mtcnn),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(280),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=1),
        transforms.RandomAffine(5, (0.02, 0.02), (0.95, 1.05)),
        transforms.RandomAutocontrast(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.225]),
    ]
)



class GrayscaleImageDataset(Dataset):
    def __init__(self, root_dir, transform=transform):
        self.root_dir = str(root_dir)
        self.transform = transform

        root = Path(self.root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images: list[str] = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                images.append(str(p))
        self.images = sorted(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB') 
        if self.transform:
            image = self.transform(image)    
        return image 
    
