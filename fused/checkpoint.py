"""Checkpoint loading for the FUSED model.

``load_fused`` accepts either an already-instantiated model or a Hydra model
config node, and handles both checkpoint formats used here:

  (a) the SparseViT backbone initialization, whose keys are prefixed
      ``encoder_net.`` and are remapped to ``encoder.``, keeping ``lmu.*`` and
      leaving the task heads randomly initialized;
  (b) a full FUSED training checkpoint, which is loaded as is.

A full training checkpoint may omit the frozen ConvNeXt weights: the constructor
already loads them from timm, so those keys are optional. Any other missing
parameter still raises when ``strict=True``. Extra tensors are reported and
ignored.
"""

import torch
import torch.nn as nn


def load_fused(cfg_or_model, ckpt_path: str, device, strict: bool = True):
    """Build (if needed) the FUSED model and load weights from ``ckpt_path``.

    Args:
        cfg_or_model: an ``nn.Module``, or a Hydra model config node with a
            ``_target_`` to instantiate.
        ckpt_path: a backbone-init or a full training checkpoint.
        device: device to move the model onto.
        strict: raise if the checkpoint is missing model parameters.

    Returns:
        The model on ``device``.
    """
    if isinstance(cfg_or_model, nn.Module):
        model = cfg_or_model
    else:
        from hydra.utils import instantiate

        model = instantiate(cfg_or_model)

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = raw.get("state_dict", raw.get("model", raw))
    src = {k[len("module."):] if k.startswith("module.") else k: v
           for k, v in src.items()}

    if any(k.startswith("encoder_net.") for k in src):
        remap = {}
        for key, value in src.items():
            if key.startswith("encoder_net."):
                remap["encoder." + key[len("encoder_net."):]] = value
            elif key.startswith("lmu."):
                remap[key] = value
        result = model.load_state_dict(remap, strict=False)
        matched = len(remap) - len(result.unexpected_keys)
        print(f"Backbone initialized from {ckpt_path}: {matched} keys matched, "
              f"{len(result.missing_keys)} left random (task heads)")
    else:
        result = model.load_state_dict(src, strict=False)
        matched = len(src) - len(result.unexpected_keys)
        print(f"Loaded {ckpt_path}: {matched} keys matched, "
              f"{len(result.missing_keys)} missing, "
              f"{len(result.unexpected_keys)} unused in this configuration")
        missing = result.missing_keys
        required_missing = [
            k for k in missing
            if not k.startswith("convnext.") and not k.endswith("num_batches_tracked")
        ]
        if missing:
            message = ("checkpoint is missing model parameters: "
                       + ", ".join(missing[:10]))
            if required_missing and strict:
                raise RuntimeError(message)
            if required_missing:
                print("Warning:", message)
            elif any(k.startswith("convnext.") for k in missing):
                print(f"  ({sum(k.startswith('convnext.') for k in missing)} frozen "
                      f"convnext.* keys filled from timm)")

    return model.to(device)
