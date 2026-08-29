"""FUSED: forensic-semantic expert fusion for detecting and localizing AI inpainting.

    from hydra.utils import instantiate
    from fused.checkpoint import load_fused
    from fused.data import OpenSDIDataset
    from fused.losses import fused_loss

    model = load_fused(instantiate(cfg.model), checkpoint, device)
"""

__version__ = "1.0.0"
