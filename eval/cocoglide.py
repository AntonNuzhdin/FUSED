import csv
import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval._metrics import _Acc
from eval.opensdi import build_model

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


class CocoGlide(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        with open(self.root / "table.csv") as f:
            rows = list(csv.DictReader(f))
        self.entries = []
        for r in rows:
            self.entries.append({"img": r["real"], "mask": None, "member": 0})
            self.entries.append({"img": r["fake"], "mask": r["mask"], "member": 1})

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(self.root / e["img"]).convert("RGB")
        img_t = _img_t(img)
        if e["mask"] is None:
            mask_t = torch.zeros(1, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
        else:
            m = Image.open(self.root / e["mask"]).convert("L")
            arr = (np.array(_mask_t(m).squeeze(0)) > 0.5).astype(np.float32)
            mask_t = torch.from_numpy(arr).unsqueeze(0)
        return img_t, mask_t, e["member"]


def collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([b[1] for b in batch], dim=0)
    member = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return imgs, masks, member


@torch.no_grad()
@hydra.main(version_base=None, config_path="../configs/eval", config_name="cocoglide")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.checkpoint, device, cfg.model_from_checkpoint)
    model.eval()

    ds = CocoGlide(cfg.data_root)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate)
    print(f"CocoGlide entries (real+fake interleaved): {len(ds)}", flush=True)

    acc = {0: _Acc(), 1: _Acc()}
    for imgs, masks, member in tqdm(loader, desc="eval", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        member = member.to(device)
        with torch.autocast("cuda"):
            res = model(imgs)
        logits = res["logits"]
        cls_logits = res.get("cls_logits")
        probs = logits.sigmoid().squeeze(1)
        gt = masks.squeeze(1)
        for i in range(imgs.size(0)):
            m = int(member[i].item())
            acc[m].update(probs[i:i + 1], gt[i:i + 1], is_tampered=(m == 1))
            if cls_logits is not None:
                acc[m].img_scores[-1] = cls_logits[i].softmax(dim=0)[1].item()

    real, fake = acc[0], acc[1]
    overall = {"real": real.summary(), "fake": fake.summary()}
    all_scores = real.img_scores + fake.img_scores
    all_labels = [0] * len(real.img_scores) + [1] * len(fake.img_scores)
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except Exception:
        auc = float("nan")
    f1_bin = f1_score(all_labels, [int(s >= 0.5) for s in all_scores])

    overall["binary"] = {
        "auc": auc, "f1": f1_bin,
        "acc": float(np.mean([(s >= 0.5) == bool(l)
                              for s, l in zip(all_scores, all_labels)])),
        "mean_acc": (real.image_acc() + fake.image_acc()) / 2,
        "cls_acc_real": real.image_acc(), "cls_acc_fake": fake.image_acc(),
        "iou_per_image_fake": fake.summary().get("iou_per_image", float("nan")),
        "f1_per_image_fake": fake.summary().get("f1_per_image", float("nan")),
        "iou_per_image_real": real.summary().get("iou_per_image", float("nan")),
        "f1_per_image_real": real.summary().get("f1_per_image", float("nan")),
    }
    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"overall": overall}, f, indent=2,
                  default=lambda o: float("nan") if isinstance(o, float) else o)
    b = overall["binary"]
    print()
    print(f"Checkpoint: {cfg.checkpoint}")
    print(f"CocoGlide ({len(ds)//2} pairs)")
    print(f"  cls_acc real={b['cls_acc_real']*100:.2f}%  fake={b['cls_acc_fake']*100:.2f}%  "
          f"acc={b['acc']*100:.2f}  binAUC={b['auc']*100:.2f}  binF1={b['f1']*100:.2f}")
    print(f"  Pix.F1 (fake, per-image) = {b['f1_per_image_fake']*100:.2f}")
    print(f"  IoU    (fake, per-image) = {b['iou_per_image_fake']*100:.2f}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
