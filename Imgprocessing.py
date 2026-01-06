from __future__ import annotations


from pathlib import Path
from typing import Optional

from PIL import Image
import cv2
from facenet_pytorch import MTCNN
import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _mtcnn_detect_face_is_broken() -> bool:
    if MTCNN is None:
        return False
    try:
        import inspect
        from facenet_pytorch.models.utils.detect_face import detect_face as _detect_face 
        return "self.mtcnn" in inspect.getsource(_detect_face)
    except Exception:
        return False


if _mtcnn_detect_face_is_broken():
    print(
        "Warning: facenet_pytorch detect_face() looks corrupted (references `self`). "
        "Disabling face masking. Fix by reinstalling: `pip install --force-reinstall --no-cache-dir facenet-pytorch==2.6.0`"
    )
    mtcnn = None
else:
    mtcnn = MTCNN(keep_all=False, device=_default_device()) if MTCNN is not None else None


class SoftFaceMaskTransform:
    def __init__(self, mtcnn: Optional[object], blur: int = 65, expand: float = 1.5, bg_color: int = 0):
        self.mtcnn = mtcnn
        self.blur = int(blur)
        self.expand = float(expand)
        self.bg_color = int(bg_color)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.mtcnn is None or cv2 is None:
            return img

        w, h = img.size
        boxes, _ = self.mtcnn.detect(img)

        if boxes is None or len(boxes) == 0:
            return img

        x1, y1, x2, y2 = boxes[0]

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = (x2 - x1) * self.expand, (y2 - y1) * self.expand

        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return img

        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        axes = (max(1, int((x2 - x1) / 2)), max(1, int((y2 - y1) / 2)))

        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        k = max(1, self.blur)
        if k % 2 == 0:
            k += 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
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
    ],
)


class GrayscaleImageDataset(Dataset):
    def __init__(self, root_dir, transform=transform):
        self.root_dir = str(root_dir)
        self.transform = transform

        root = Path(self.root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                images.append(str(p))
        self.images = sorted(images)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        max_tries = 25
        for offset in range(max_tries):
            j = (idx + offset) % len(self.images)
            img_path = self.images[j]
            try:
                with Image.open(img_path) as img:
                    image = img.convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image
            except Exception:
                continue
        raise OSError(f"Failed to load a valid image after {max_tries} tries (starting at index {idx}).")