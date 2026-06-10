"""
Unpacked shard 디렉토리에서 unlabeled 이미지를 로딩.
GT가 없으므로 (image, path)만 반환 — pseudo-label은 학습 루프에서 Teacher가 생성.
"""
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(image_dir: str) -> List[str]:
    return sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in IMG_EXT
    )


class ShardImageDataset(Dataset):
    """
    YOLO-World 입력 규약:
      - 0~1 정규화된 RGB tensor (B,3,H,W)
      - imgsz × imgsz 정사각 리사이즈 (letterbox 없이 단순 resize — KD 학습엔 충분)
    """

    def __init__(self, image_dir: str, imgsz: int = 640):
        self.image_dir = image_dir
        self.imgsz = imgsz
        self.files = list_images(image_dir)
        if not self.files:
            raise RuntimeError(f"이미지 없음: {image_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        ow, oh = img.size
        img = img.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        x = torch.from_numpy(_pil_to_chw01(img))
        return {
            "image": x,
            "path": path,
            "orig_hw": torch.tensor([oh, ow], dtype=torch.int32),
        }


def _pil_to_chw01(img: Image.Image):
    import numpy as np
    arr = np.asarray(img, dtype="float32") / 255.0  # HWC
    return arr.transpose(2, 0, 1).copy()  # CHW


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], 0),
        "path": [b["path"] for b in batch],
        "orig_hw": torch.stack([b["orig_hw"] for b in batch], 0),
    }
