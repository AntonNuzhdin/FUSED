import json
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A

IMG_SIZE = 512


def _edge_from_mask(mask_np: np.ndarray) -> np.ndarray:
    """Canny 1px boundary of a binary mask (matches train_mvss_protocol)."""
    if mask_np.sum() == 0:
        return np.zeros_like(mask_np, dtype=np.float32)
    m8 = (mask_np * 255).astype(np.uint8)
    return (cv2.Canny(m8, 50, 150) > 0).astype(np.float32)


def _build_train_aug() -> A.Compose:
    """Training augmentation recipe (albumentations)."""
    return A.Compose([
        A.RandomScale(scale_limit=0.2, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                   contrast_limit=0.1, p=1),
        A.ImageCompression(quality_range=(70, 100), p=0.2),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ])


# Image normalization: ImageNet statistics.
_img_transform = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _classify_subset(subset: str) -> tuple[int, str]:
    """Return (cls_label, mask_kind) for a subset string.

    cls_label: 0 for real, 1 for fake.
    mask_kind: 'zeros' | 'load' | 'ones'
    """
    if subset.endswith("_real"):
        return 0, "zeros"
    # fake
    if "_partial_" in subset:
        return 1, "load"
    if "_entire_" in subset or "_flat_" in subset:
        return 1, "ones"
    raise ValueError(f"unrecognized OpenSDI subset: {subset!r}")


class OpenSDIDataset(Dataset):
    def __init__(self, manifest_path: str, img_size: int = IMG_SIZE,
                 augment: bool = False):
        with open(manifest_path) as f:
            self.entries = json.load(f)
        self.img_size = img_size
        self.augment = bool(augment)
        # Build once; albumentations Compose is thread-safe under DataLoader workers.
        self._aug = _build_train_aug() if self.augment else None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, mask_path, subset = self.entries[idx]
        cls_label, mask_kind = _classify_subset(subset)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Step to next index on broken file
            return self.__getitem__((idx + 1) % len(self))

        # Load the mask BEFORE augmentation so geometric transforms co-apply.
        # For 'ones'/'zeros' subsets we still synthesize a full-image mask at
        # source resolution, so scale/rotate leave it a valid constant mask.
        if mask_kind == "load" and mask_path is not None:
            try:
                m = Image.open(mask_path).convert("L")
                mask_np = (np.array(m, dtype=np.uint8) > 127).astype(np.uint8)
            except Exception:
                mask_np = np.zeros((img.height, img.width), dtype=np.uint8)
                mask_kind = "zeros"  # degrade to real-mask semantics on load failure
        elif mask_kind == "ones":
            mask_np = np.ones((img.height, img.width), dtype=np.uint8)
        else:  # zeros
            mask_np = np.zeros((img.height, img.width), dtype=np.uint8)

        # Apply the training augmentations jointly to image and mask.
        if self._aug is not None:
            img_np = np.array(img, dtype=np.uint8)                  # HWC uint8
            if img_np.shape[:2] != mask_np.shape[:2]:
                mask_np = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            out = self._aug(image=img_np, mask=mask_np)
            img = Image.fromarray(out["image"])                     # RGB PIL
            mask_np = out["mask"].astype(np.uint8)                  # HW uint8

        img_t = _img_transform(img)

        # Mask -> float tensor at IMG_SIZE (NEAREST to preserve binary values).
        mask_resized = cv2.resize(mask_np, (self.img_size, self.img_size),
                                  interpolation=cv2.INTER_NEAREST)
        mask_t = torch.from_numpy(
            (mask_resized > 0).astype(np.float32)).unsqueeze(0)

        edge_t = torch.from_numpy(_edge_from_mask(mask_t[0].numpy())).unsqueeze(0)

        return {
            "image": img_t,
            "seg_labels": mask_t,
            "edge": edge_t,
            "member": torch.tensor(cls_label, dtype=torch.long),  # 0=real, 1=fake
            "cls_labels": torch.tensor(cls_label, dtype=torch.long),
            "pair_id": torch.tensor(idx, dtype=torch.long),
            "subset": subset,
        }


def collate(batch):
    out = {
        "image":      torch.stack([b["image"] for b in batch], dim=0),
        "seg_labels": torch.stack([b["seg_labels"] for b in batch], dim=0),
        "edge":       torch.stack([b["edge"] for b in batch], dim=0),
        "member":     torch.stack([b["member"] for b in batch], dim=0),
        "cls_labels": torch.stack([b["cls_labels"] for b in batch], dim=0),
        "pair_id":    torch.stack([b["pair_id"] for b in batch], dim=0),
        "subset":     [b["subset"] for b in batch],
    }
    return out
