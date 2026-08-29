"""Localization and detection stratified by edit size (Appendix D).

    uv run python eval/area_stratified.py checkpoint=<latest.pth>
"""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BIN_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.01]
BIN_LABELS = ["(0,2%]", "(2,5%]", "(5,10%]", "(10,20%]",
              "(20,30%]", "(30,50%]", ">50%"]


def bin_index(area):
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] < area <= BIN_EDGES[i + 1]:
            return i
    return 0


def evaluate(cfg):
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from eval._metrics import per_image_iou_f1
    from eval.opensdi import build_model
    from fused.data.opensdi import OpenSDIDataset, collate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.checkpoint, device, cfg.model_from_checkpoint)

    dataset = OpenSDIDataset(cfg.test_json)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)

    bins = [{"n": 0, "iou_sum": 0.0, "f1_sum": 0.0, "det_correct": 0}
            for _ in BIN_LABELS]
    with torch.no_grad():
        for batch in tqdm(loader, desc="area stratified"):
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["seg_labels"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                result = model(imgs)
            pred_mask = (result["logits"].sigmoid() >= 0.5).float()
            pred_cls = result["cls_logits"].argmax(dim=1)

            for i, subset in enumerate(batch["subset"]):
                if not subset.endswith("_partial_fake"):
                    continue
                area = float(masks[i, 0].mean().item())
                if area <= 0:
                    continue
                slot = bins[bin_index(area)]
                iou, f1 = per_image_iou_f1(pred_mask[i, 0], masks[i, 0])
                slot["n"] += 1
                slot["iou_sum"] += iou
                slot["f1_sum"] += f1
                slot["det_correct"] += int(int(pred_cls[i].item()) == 1)

    rows = []
    for label, slot in zip(BIN_LABELS, bins):
        n = max(1, slot["n"])
        rows.append({"bin": label, "n": slot["n"],
                     "iou": slot["iou_sum"] / n,
                     "pixel_f1": slot["f1_sum"] / n,
                     "det_acc": slot["det_correct"] / n})

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "area_stratified.json"
    json.dump({"checkpoint": cfg.checkpoint, "bins": rows}, open(path, "w"), indent=2)

    header = f"{'Area bin':12s} {'n':>8s} {'IoU':>8s} {'Pix.F1':>8s} {'Det.Acc':>8s}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['bin']:12s} {row['n']:8d} {row['iou'] * 100:8.1f} "
              f"{row['pixel_f1'] * 100:8.1f} {row['det_acc'] * 100:8.1f}")
    print(f"\nWrote {path}")
    return path


def plot(json_path, out_path, baseline_path=None, baseline_label="baseline"):
    import matplotlib.pyplot as plt

    rows = json.load(open(json_path))["bins"]
    baseline = json.load(open(baseline_path))["bins"] if baseline_path else None
    labels = [r["bin"] for r in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, title in ((axes[0], "pixel_f1", "Localization"),
                           (axes[1], "det_acc", "Detection")):
        if baseline is not None:
            ax.plot(x, [r[key] for r in baseline], color="#d1495b", marker="o",
                    linewidth=2, label=baseline_label)
        ax.plot(x, [r[key] for r in rows], color="#2f6f9f", marker="s", linestyle="--",
                linewidth=2, label="FUSED")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("Ground-truth edit size", fontsize=11)
        ax.set_ylabel("Pixel F1" if key == "pixel_f1" else "Accuracy", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=10, frameon=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


@hydra.main(version_base=None, config_path="../configs/eval", config_name="area")
def main(cfg: DictConfig):
    out_dir = Path(cfg.out_dir)
    json_path = out_dir / "area_stratified.json"
    if not cfg.plot_only:
        json_path = evaluate(cfg)
    if cfg.plot:
        plot(json_path, out_dir / "area_stratified.png",
             cfg.baseline_json, cfg.baseline_label)


if __name__ == "__main__":
    main()
