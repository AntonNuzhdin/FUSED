import json
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

GENERATORS = ["sd15", "sd2", "sdxl", "sd3", "flux"]
GENERATOR_LABELS = {"sd15": "SD1.5", "sd2": "SD2.1", "sdxl": "SDXL",
                    "sd3": "SD3", "flux": "Flux.1"}
BRANCHES = ["svit", "cnx"]
BRANCH_TITLES = {"svit": "SparseViT tokens", "cnx": "ConvNeXt tokens"}


def collect(cfg):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from fused.checkpoint import load_fused
    from fused.data.opensdi import OpenSDIDataset, collate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    saved = raw.get("cfg", {}).get("model") if isinstance(raw, dict) else None
    del raw
    model_cfg = (OmegaConf.create(saved) if saved is not None
                 else OmegaConf.load(_REPO_ROOT / "configs" / "model" / "fused.yaml"))
    model = load_fused(instantiate(model_cfg), cfg.checkpoint, device).eval()
    num_experts = int(model_cfg.moe_num_experts)
    top_k = int(model_cfg.moe_top_k)

    dataset = OpenSDIDataset(cfg.test_json)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate, pin_memory=True)

    counts = defaultdict(lambda: {b: np.zeros(num_experts) for b in BRANCHES})
    slots = defaultdict(lambda: {b: 0 for b in BRANCHES})

    with torch.no_grad():
        for batch in tqdm(loader, desc="routing utilization"):
            imgs = batch["image"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                result = model(imgs)
            layout = result["token_layout"]
            n_svit, n_cnx = layout["n_svit"], layout["n_cnx"]
            probs = result["router_logits"].float().softmax(-1)
            topk = probs.topk(top_k, dim=-1).indices.cpu().numpy()

            spans = {"svit": slice(0, n_svit), "cnx": slice(n_svit, n_svit + n_cnx)}
            for i, subset in enumerate(batch["subset"]):
                gen = subset.split("_", 1)[0]
                for branch, span in spans.items():
                    selected = topk[i, span].reshape(-1)
                    if selected.size == 0:
                        continue
                    counts[gen][branch] += np.bincount(selected, minlength=num_experts)
                    slots[gen][branch] += selected.size

    out = {
        "checkpoint": cfg.checkpoint,
        "num_experts": num_experts,
        "top_k": top_k,
        "uniform": 1.0 / num_experts,
        "per_generator": {
            gen: {
                branch: (counts[gen][branch] / max(1, slots[gen][branch])).tolist()
                for branch in BRANCHES
            }
            for gen in GENERATORS
        },
    }
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "routing_utilization.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"Wrote {path}")
    return path


def plot(json_path, out_path):
    import matplotlib.pyplot as plt

    data = json.load(open(json_path))
    num_experts = data["num_experts"]
    uniform = data["uniform"]
    x = np.arange(num_experts)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(GENERATORS)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    width = 0.8 / len(GENERATORS)
    for ax, branch in zip(axes, BRANCHES):
        for j, gen in enumerate(GENERATORS):
            values = data["per_generator"][gen][branch]
            ax.bar(x + (j - (len(GENERATORS) - 1) / 2) * width, values, width,
                   label=GENERATOR_LABELS[gen], color=colors[j], zorder=3)
        ax.axhline(uniform, linestyle="--", color="0.35", linewidth=1.2, zorder=4,
                   label="uniform (1/E)" if branch == BRANCHES[0] else None)
        ax.set_title(BRANCH_TITLES[branch], fontsize=13)
        ax.set_xlabel("Expert index", fontsize=12)
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Utilization", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(GENERATORS) + 1,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


@hydra.main(version_base=None, config_path="../configs/eval", config_name="routing")
def main(cfg: DictConfig):
    out_dir = Path(cfg.out_dir)
    json_path = out_dir / "routing_utilization.json"
    if not cfg.plot_only:
        json_path = collect(cfg)
    if cfg.plot:
        plot(json_path, out_dir / "routing_utilization.png")


if __name__ == "__main__":
    main()
