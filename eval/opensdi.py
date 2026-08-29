import inspect
import json
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval._metrics import per_image_iou_f1
from fused.checkpoint import load_fused
from fused.data.opensdi import OpenSDIDataset, collate
from fused.model.model import FUSED

GENERATORS = ["sd15", "sd2", "sdxl", "sd3", "flux"]
GENERATOR_LABELS = {"sd15": "SD1.5", "sd2": "SD2.1", "sdxl": "SDXL",
                    "sd3": "SD3", "flux": "Flux.1"}


def build_model(checkpoint, device, from_checkpoint=True):
    """Build the model for a checkpoint and load its weights.

    Training writes the resolved model config into every checkpoint, so an
    ablation arm rebuilds its own architecture from ``checkpoint`` alone. Pass
    ``from_checkpoint=False`` to build ``configs/model/fused.yaml`` instead.
    """
    model_cfg = OmegaConf.load(_REPO_ROOT / "configs" / "model" / "fused.yaml")
    if from_checkpoint:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved = raw.get("cfg", {}).get("model") if isinstance(raw, dict) else None
        del raw
        if saved is None:
            print("No model config in the checkpoint; using configs/model/fused.yaml")
        else:
            saved = OmegaConf.to_container(OmegaConf.create(saved), resolve=True)
            accepted = set(inspect.signature(FUSED.__init__).parameters)
            for key, value in saved.items():
                if key in accepted:
                    model_cfg[key] = value
            unused = sorted(set(saved) - accepted)
            if unused:
                print("Checkpoint config keys not used by this model: "
                      + ", ".join(unused))
    return load_fused(instantiate(model_cfg), checkpoint, device).eval()


def _binary_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)


def _new_bucket():
    return {"loc_n": 0, "iou_sum": 0.0, "f1_sum": 0.0,
            "det_n": 0, "det_correct": 0, "tp": 0, "fp": 0, "fn": 0}


@torch.no_grad()
@hydra.main(version_base=None, config_path="../configs/eval", config_name="opensdi")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.checkpoint, device, cfg.model_from_checkpoint)

    dataset = OpenSDIDataset(cfg.test_json)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)
    print(f"Test set: {len(dataset)} images")

    buckets = defaultdict(_new_bucket)
    for batch in tqdm(loader, desc="OpenSDI test"):
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["seg_labels"].to(device, non_blocking=True)
        cls_labels = batch["cls_labels"].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            result = model(imgs)
        pred_mask = (result["logits"].sigmoid() >= 0.5).float()
        pred_cls = result["cls_logits"].argmax(dim=1)

        for i, subset in enumerate(batch["subset"]):
            bucket = buckets[subset.split("_", 1)[0]]
            if subset.endswith("_partial_fake"):
                iou, f1 = per_image_iou_f1(pred_mask[i, 0], masks[i, 0])
                bucket["loc_n"] += 1
                bucket["iou_sum"] += iou
                bucket["f1_sum"] += f1
            gt = int(cls_labels[i].item())
            pred = int(pred_cls[i].item())
            bucket["det_n"] += 1
            bucket["det_correct"] += int(pred == gt)
            if gt == 1 and pred == 1:
                bucket["tp"] += 1
            elif gt == 0 and pred == 1:
                bucket["fp"] += 1
            elif gt == 1 and pred == 0:
                bucket["fn"] += 1

    per_generator = {}
    for gen in GENERATORS:
        b = buckets[gen]
        per_generator[gen] = {
            "label": GENERATOR_LABELS[gen],
            "loc_iou": b["iou_sum"] / max(1, b["loc_n"]),
            "loc_f1": b["f1_sum"] / max(1, b["loc_n"]),
            "loc_n": b["loc_n"],
            "det_f1": _binary_f1(b["tp"], b["fp"], b["fn"]),
            "det_acc": b["det_correct"] / max(1, b["det_n"]),
            "det_n": b["det_n"],
        }

    average = {
        key: sum(per_generator[g][key] for g in GENERATORS) / len(GENERATORS)
        for key in ("loc_iou", "loc_f1", "det_f1", "det_acc")
    }
    out = {"checkpoint": cfg.checkpoint, "test_json": cfg.test_json,
           "per_generator": per_generator,
           "average": {**average, "note": "unweighted mean over the five generators"}}

    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cfg.out, "w"), indent=2)

    header = f"{'Generator':10s} {'IoU':>8s} {'Pix.F1':>8s} {'Det.F1':>8s} {'Det.Acc':>8s}"
    print("\n" + header)
    print("-" * len(header))
    for gen in GENERATORS:
        r = per_generator[gen]
        print(f"{r['label']:10s} {r['loc_iou'] * 100:8.1f} {r['loc_f1'] * 100:8.1f} "
              f"{r['det_f1'] * 100:8.1f} {r['det_acc'] * 100:8.1f}")
    print("-" * len(header))
    print(f"{'Average':10s} {average['loc_iou'] * 100:8.1f} {average['loc_f1'] * 100:8.1f} "
          f"{average['det_f1'] * 100:8.1f} {average['det_acc'] * 100:8.1f}")
    print(f"\nWrote {cfg.out}")


if __name__ == "__main__":
    main()
