"""FUSED model package.

The model is built by Hydra from ``configs/model/fused.yaml``::

    from hydra.utils import instantiate
    model = instantiate(cfg.model)
"""

from fused.model.model import FUSED

__all__ = ["FUSED"]
