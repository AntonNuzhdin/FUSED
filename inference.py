"""Single-image (or folder) inference with FUSED.

Loads a checkpoint, runs the model on one image or every image in a folder, and
for each writes:
  - <stem>_prob.png     predicted probability mask (grayscale)
  - <stem>_mask.png     binary mask (threshold 0.5)
  - <stem>_overlay.png  input with the predicted region tinted red
and prints P(fake).

    uv run python inference.py checkpoint=<ckpt> input=image.jpg
    uv run python inference.py checkpoint=<ckpt> input=folder/

Preprocessing is the same as at training time: Resize(512) + ImageNet
normalization.
"""
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from eval.opensdi import build_model

IMG_SIZE = 512
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_img_t = transforms.Compose([
    transforms.Resize([IMG_SIZE, IMG_SIZE]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def run_one(model, img_path: Path, out_dir: Path, device, threshold: float = 0.5):
    pil = Image.open(img_path).convert("RGB")
    W, H = pil.size
    x = _img_t(pil).unsqueeze(0).to(device)
    with torch.autocast("cuda", enabled=device.type == "cuda"):
        res = model(x)
    prob = res["logits"].sigmoid()[0, 0].float().cpu().numpy()        # (512, 512)
    p_fake = float(res["cls_logits"].softmax(-1)[0, 1].item())

    # Resize the probability map back to the original resolution.
    prob_full = np.array(
        Image.fromarray((prob * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    binmask = (prob_full >= threshold).astype(np.uint8)

    stem = img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((prob_full * 255).astype(np.uint8)).save(out_dir / f"{stem}_prob.png")
    Image.fromarray((binmask * 255).astype(np.uint8)).save(out_dir / f"{stem}_mask.png")

    # Red overlay.
    base = np.array(pil).astype(np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    alpha = (binmask[..., None].astype(np.float32)) * 0.5
    overlay = (base * (1 - alpha) + red * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_dir / f"{stem}_overlay.png")

    print(f"{img_path.name}: P(fake)={p_fake:.4f}  fake_pixels={int(binmask.sum())}/{W*H}")
    return p_fake


@hydra.main(version_base=None, config_path="configs", config_name="inference")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.checkpoint, device, cfg.model_from_checkpoint)

    in_path = Path(cfg.input)
    if in_path.is_dir():
        imgs = sorted(p for p in in_path.iterdir() if p.suffix.lower() in _IMG_EXTS)
    else:
        imgs = [in_path]
    if not imgs:
        print(f"No images found at {in_path}")
        return

    out_dir = Path(cfg.out_dir)
    for p in imgs:
        run_one(model, p, out_dir, device, threshold=cfg.threshold)
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
