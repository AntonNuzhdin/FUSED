import json
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.opensdi import build_model

DATASETS = ("CelebAHQ", "CityScapes", "OpenImages", "SUN_RGBD")
IMG_SIZE = 512

_img_t = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_mask_t = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE],
                      interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


def build_entries(data_root: Path):
    """List[(img_path, mask_path | None, member)]. member: 0=real, 1=inp, 2=exch."""
    entries = []
    for ds in DATASETS:
        rd = data_root / "test-data" / "data" / "originals" / ds
        if rd.exists():
            for p in sorted(rd.iterdir()):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    entries.append((str(p), None, 0))
        for sub, m in (("standard_inpainting", 1), ("inpainting_exchange", 2)):
            d = data_root / "test-data" / "data" / sub / ds
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                stem = p.stem.removesuffix("_simple")
                tag = f"_{ds}_"
                idx = stem.rfind(tag)
                if idx == -1:
                    continue
                mask_stem = stem[:idx]
                mp = None
                masks_dir = data_root / "test-data" / "masks" / f"{ds}_masks"
                for ext in (".jpg", ".png"):
                    cand = masks_dir / f"{mask_stem}{ext}"
                    if cand.exists():
                        mp = str(cand)
                        break
                entries.append((str(p), mp, m))
    return entries


class FlatINPX(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, mask_path, member = self.entries[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.entries))
        img_t = _img_t(img)
        if mask_path is None:
            mask_t = torch.zeros(1, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
        else:
            m = Image.open(mask_path).convert("L")
            mt = _mask_t(m).squeeze(0)
            mask_t = (mt > 0.5).float().unsqueeze(0)
        return img_t, mask_t, member


def collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([b[1] for b in batch], dim=0)
    member = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, masks, member


@torch.no_grad()
def evaluate(checkpoint, data_root, out_path, batch_size, device,
             from_checkpoint=True):
    model = build_model(checkpoint, device, from_checkpoint)
    model.eval()

    entries = build_entries(Path(data_root))
    counts = defaultdict(int)
    for _, _, m in entries:
        counts[m] += 1
    print(f"Test entries: real={counts[0]}  inp={counts[1]}  exch={counts[2]}", flush=True)
    ds = FlatINPX(entries)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=8, pin_memory=True, collate_fn=collate)

    per_class_scores  = {0: [], 1: [], 2: []}
    per_class_correct = {0: [], 1: [], 2: []}
    per_class_iou_img = {0: [], 1: [], 2: []}
    per_class_f1_img  = {0: [], 1: [], 2: []}

    for imgs, masks, members in tqdm(loader, desc="fused-eval", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast("cuda"):
            res = model(imgs)
        logits = res["logits"]
        cls_logits = res.get("cls_logits")
        probs = logits.sigmoid().squeeze(1).float().cpu().numpy()
        gt = masks.squeeze(1).cpu().numpy()
        if cls_logits is not None:
            cls_scores = cls_logits.softmax(dim=1)[:, 1].float().cpu().numpy()
        else:
            cls_scores = probs.reshape(probs.shape[0], -1).mean(axis=1)
        for i in range(imgs.size(0)):
            m = int(members[i].item())
            score = float(cls_scores[i])
            per_class_scores[m].append(score)
            true_label = 0 if m == 0 else 1
            per_class_correct[m].append(int((score >= 0.5) == true_label))

            pred = probs[i]
            g = gt[i]
            pred_bin = (pred >= 0.5).astype(np.float32)
            gt_empty = (g.sum() == 0)
            pred_empty = (pred_bin.sum() == 0)
            if gt_empty and pred_empty:
                iou_i, f1_i = 1.0, 1.0
            elif gt_empty or pred_empty:
                iou_i, f1_i = 0.0, 0.0
            else:
                tp = float((pred_bin * g).sum())
                fp = float((pred_bin * (1 - g)).sum())
                fn = float(((1 - pred_bin) * g).sum())
                iou_i = tp / (tp + fp + fn + 1e-8)
                f1_i = 2 * tp / (2 * tp + fp + fn + 1e-8)
            per_class_iou_img[m].append(iou_i)
            per_class_f1_img[m].append(f1_i)

    real_acc = float(np.mean(per_class_correct[0])) if per_class_correct[0] else float("nan")
    inp_acc  = float(np.mean(per_class_correct[1])) if per_class_correct[1] else float("nan")
    exc_acc  = float(np.mean(per_class_correct[2])) if per_class_correct[2] else float("nan")

    iou_img_inp = float(np.mean(per_class_iou_img[1]))
    iou_img_exc = float(np.mean(per_class_iou_img[2]))
    f1_img_inp  = float(np.mean(per_class_f1_img[1]))
    f1_img_exc  = float(np.mean(per_class_f1_img[2]))

    # The two conditions are scored as separate real-vs-fake problems against the
    # shared real set, which is what the paired columns of Table 4 report.
    def detection(fake_scores):
        scores = per_class_scores[0] + fake_scores
        labels = [0] * len(per_class_scores[0]) + [1] * len(fake_scores)
        return (float(roc_auc_score(labels, scores)),
                float(f1_score(labels, [int(s >= 0.5) for s in scores])))

    auc_inp, f1_inp = detection(per_class_scores[1])
    auc_exc, f1_exc = detection(per_class_scores[2])

    inpx_mean_acc = (real_acc + exc_acc) / 2
    inp_mean_acc  = (real_acc + inp_acc) / 2
    gap = abs(inp_mean_acc - inpx_mean_acc)

    res = {
        "model": "FUSED on INP-X (directory-walk eval)",
        "checkpoint": checkpoint,
        "n": dict(counts),
        "per_class_acc": {"real": real_acc, "inpaint": inp_acc, "exchange": exc_acc},
        "per_image": {
            "iou_per_image": {"inpaint": iou_img_inp, "exchange": iou_img_exc,
                              "tampered_mean": (iou_img_inp + iou_img_exc) / 2},
            "f1_per_image":  {"inpaint": f1_img_inp, "exchange": f1_img_exc,
                              "tampered_mean": (f1_img_inp + f1_img_exc) / 2},
        },
        "detection": {
            "inp":   {"f1": f1_inp, "auc": auc_inp, "acc": inp_mean_acc},
            "inp_x": {"f1": f1_exc, "auc": auc_exc, "acc": inpx_mean_acc},
            "gap": gap,
        },
    }
    print(f"\n{'='*60}")
    print(f"{'':22s}{'INP':>10s}{'INP-X':>10s}")
    print(f"{'Pix. F1 (per-image)':22s}{f1_img_inp * 100:10.2f}{f1_img_exc * 100:10.2f}")
    print(f"{'IoU (per-image)':22s}{iou_img_inp * 100:10.2f}{iou_img_exc * 100:10.2f}")
    print(f"{'F1 (detection)':22s}{f1_inp * 100:10.2f}{f1_exc * 100:10.2f}")
    print(f"{'AUC':22s}{auc_inp * 100:10.2f}{auc_exc * 100:10.2f}")
    print(f"{'Acc.':22s}{inp_mean_acc * 100:10.2f}{inpx_mean_acc * 100:10.2f}")
    print(f"\nAccuracy cost of removing the artifact = {gap * 100:.2f} points")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved -> {out_path}")


@hydra.main(version_base=None, config_path="../configs/eval", config_name="inpx_test")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(cfg.checkpoint, cfg.data_root, cfg.out, cfg.batch_size, device,
             cfg.model_from_checkpoint)


if __name__ == "__main__":
    main()
