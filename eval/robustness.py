import io
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

GENERATOR_LABELS = {"sd15": "SD1.5", "sd2": "SD2.1", "sdxl": "SDXL",
                    "sd3": "SD3", "flux": "Flux.1"}
BLUR_LEVELS = [3, 7, 11, 15, 19, 23]
JPEG_LEVELS = [100, 90, 80, 70, 60, 50]
LEVELS = {"blur": BLUR_LEVELS, "jpeg": JPEG_LEVELS}


def corrupt(image, kind, level):
    """Apply one corruption to a PIL image."""
    from PIL import Image, ImageFilter

    if kind == "blur":
        sigma = 0.15 * level + 0.35
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(level))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class CorruptedPartialFakes(torch.utils.data.Dataset):
    """Partial-fake images of one generator, corrupted before normalization."""

    def __init__(self, manifest_path, generator, kind, level, img_size=512):
        import cv2
        from PIL import Image
        from torchvision import transforms

        self.cv2, self.Image = cv2, Image
        with open(manifest_path) as handle:
            entries = json.load(handle)
        self.entries = [e for e in entries
                        if e[2] == f"{generator}_partial_fake" and e[1] is not None]
        self.kind, self.level, self.img_size = kind, level, img_size
        self.transform = transforms.Compose([
            transforms.Resize([img_size, img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, mask_path, _ = self.entries[idx]
        image = self.Image.open(img_path).convert("RGB")
        if self.kind is not None:
            image = corrupt(image, self.kind, self.level)
        mask = np.array(self.Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = self.cv2.resize((mask > 127).astype(np.uint8),
                               (self.img_size, self.img_size),
                               interpolation=self.cv2.INTER_NEAREST)
        return self.transform(image), torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)


def evaluate(cfg):
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from eval._metrics import per_image_iou_f1
    from eval.opensdi import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.checkpoint, device, cfg.model_from_checkpoint)
    generators = list(cfg.generators)

    results = {gen: {kind: [] for kind in LEVELS} for gen in generators}
    with torch.no_grad():
        for gen in generators:
            for kind, levels in LEVELS.items():
                for level in levels:
                    dataset = CorruptedPartialFakes(cfg.test_json, gen, kind, level)
                    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                                        num_workers=cfg.num_workers, pin_memory=True)
                    f1_sum = 0.0
                    count = 0
                    for imgs, masks in tqdm(loader, desc=f"{gen} {kind}={level}", leave=False):
                        imgs = imgs.to(device, non_blocking=True)
                        masks = masks.to(device, non_blocking=True)
                        with torch.autocast("cuda", enabled=device.type == "cuda"):
                            out = model(imgs)
                        pred = (out["logits"].sigmoid() >= 0.5).float()
                        for i in range(imgs.shape[0]):
                            _, f1 = per_image_iou_f1(pred[i, 0], masks[i, 0])
                            f1_sum += f1
                            count += 1
                    mean_f1 = f1_sum / max(1, count)
                    results[gen][kind].append({"level": level, "pixel_f1": mean_f1, "n": count})
                    print(f"{gen} {kind}={level}: pixel F1 {mean_f1 * 100:.1f} (n={count})",
                          flush=True)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "robustness.json"
    json.dump({"checkpoint": cfg.checkpoint, "results": results}, open(path, "w"), indent=2)
    print(f"Wrote {path}")
    return path


def plot(json_path, out_path, baseline_path=None, baseline_label="baseline"):
    import matplotlib.pyplot as plt

    data = json.load(open(json_path))["results"]
    baseline = json.load(open(baseline_path))["results"] if baseline_path else None
    generators = list(data)

    fig, axes = plt.subplots(len(generators), 2, figsize=(9.5, 4.0 * len(generators)),
                             squeeze=False)
    for row, gen in enumerate(generators):
        for col, kind in enumerate(("blur", "jpeg")):
            ax = axes[row][col]
            levels = [p["level"] for p in data[gen][kind]]
            positions = np.arange(len(levels))
            if baseline is not None and gen in baseline:
                ax.plot(positions, [p["pixel_f1"] for p in baseline[gen][kind]],
                        color="#d1495b", marker="o", linewidth=2, label=baseline_label)
            ax.plot(positions, [p["pixel_f1"] for p in data[gen][kind]],
                    color="#2f6f9f", marker="s", linestyle="--", linewidth=2, label="FUSED")
            ax.set_xticks(positions)
            ax.set_xticklabels([str(l) for l in levels])
            ax.set_xlabel("Gaussian blur kernel" if kind == "blur" else "JPEG quality",
                          fontsize=11)
            ax.set_title(f"{'Gaussian Blur' if kind == 'blur' else 'JPEG Compression'} "
                         f"({GENERATOR_LABELS.get(gen, gen)})", fontsize=12)
            ax.grid(alpha=0.25)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[row][0].set_ylabel("Pixel F1", fontsize=11)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


@hydra.main(version_base=None, config_path="../configs/eval", config_name="robustness")
def main(cfg: DictConfig):
    out_dir = Path(cfg.out_dir)
    json_path = out_dir / "robustness.json"
    if not cfg.plot_only:
        json_path = evaluate(cfg)
    if cfg.plot:
        plot(json_path, out_dir / "robustness.png",
             cfg.baseline_json, cfg.baseline_label)


if __name__ == "__main__":
    main()
