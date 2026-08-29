"""Download the FUSED release checkpoints from the Hugging Face Hub into ``weights/``.

    uv run python scripts/download_weight.py                    # all checkpoints
    uv run python scripts/download_weight.py fused_opensdi.pth  # just one
    FORCE=1 uv run python scripts/download_weight.py            # re-download

Files already present are skipped. The checkpoints do not carry the frozen
ConvNeXt weights, which ``fused.checkpoint.load_fused`` restores from timm.

    fused_opensdi.pth         FUSED trained on OpenSDID SD1.5  -> eval/infer checkpoint=
    fused_sofake.pth          FUSED trained on So-Fake-Set     -> eval/infer checkpoint=
    sparsevit_pretrained.pth  pretrained SparseViT backbone    -> trainer.checkpoint=
"""
import os
import sys
from pathlib import Path

REPO_ID = "aonuzhdin/FUSED"

# Repo root is the parent of this scripts/ directory.
WEIGHTS = Path(__file__).resolve().parent.parent / "weights"

FILES = (
    "fused_opensdi.pth",
    "fused_sofake.pth",
    "sparsevit_pretrained.pth",
)


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is required: `uv sync` (it is a project "
                 "dependency) or `uv pip install huggingface_hub`.")

    requested = sys.argv[1:] or list(FILES)
    unknown = [name for name in requested if name not in FILES]
    if unknown:
        sys.exit(f"Unknown checkpoint(s): {', '.join(unknown)}\n"
                 f"Available: {', '.join(FILES)}")

    force = os.environ.get("FORCE", "").lower() in ("1", "true", "yes")
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    for name in requested:
        out = WEIGHTS / name
        if out.exists() and not force:
            print(f"[skip] {name} already present ({out.stat().st_size / 1e6:.0f} MB) "
                  f"— set FORCE=1 to re-download")
            continue
        print(f"[get ] {name}")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            local_dir=str(WEIGHTS),
            force_download=force,
        )

    print(f"\nDone -> {WEIGHTS}")


if __name__ == "__main__":
    main()
