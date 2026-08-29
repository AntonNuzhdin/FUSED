"""FUSED: forensic-semantic Mixture-of-Experts for AI inpainting detection.
"""

from functools import partial

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from fused.model.blocks import AttentionPooling, SelfAttention, SwiGLU
from fused.model.decoder import MultiScaleFusion
from fused.model.moe import ExpertFFN, SparseMoE
from fused.model.sparse_vit import SparseViT


class FUSED(nn.Module):
    def __init__(
        self,
        num_classes=2,
        dim=512,
        num_latents=8,
        num_heads=8,
        moe_num_experts=8,
        moe_top_k=2,
        moe_expansion=4,
        ffn_type='moe',
        seg_bridge=True,
        enable_sparsevit_branch=True,
        enable_convnext_branch=True,
        convnext_model='convnext_xxlarge.clip_laion2b_soup',
        depth=(5, 8, 20, 7),
        embed_dim=(64, 128, 320, 512),
        head_dim=64,
        img_size=512,
        s_blocks3=(8, 4, 2, 1),
        s_blocks4=(2, 1),
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.2,
        pretrained_path=None,
    ):
        super().__init__()

        if not (enable_sparsevit_branch or enable_convnext_branch):
            raise ValueError("At least one branch must be enabled")
        if ffn_type not in ('moe', 'dense'):
            raise ValueError(f"Unknown ffn_type '{ffn_type}'. Choose 'moe' | 'dense'.")

        self.enable_sparsevit_branch = enable_sparsevit_branch
        self.enable_convnext_branch = enable_convnext_branch
        self.ffn_type = ffn_type
        self.img_size = img_size
        # The bridge needs both streams; with a single branch the mask is decoded
        # from whichever stream is present.
        self.seg_bridge = bool(seg_bridge) and enable_sparsevit_branch and enable_convnext_branch

        # Forensic branch: trainable SparseViT.
        if enable_sparsevit_branch:
            if dim != embed_dim[-1]:
                raise ValueError(f"dim ({dim}) must equal embed_dim[-1] ({embed_dim[-1]})")
            self.encoder = SparseViT(
                layers=list(depth),
                embed_dim=list(embed_dim),
                img_size=img_size,
                s_blocks3=list(s_blocks3),
                s_blocks4=list(s_blocks4),
                head_dim=head_dim,
                drop_path_rate=drop_path_rate,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                pretrained_path=pretrained_path,
            )
            # Multi-scale fusion of the forensic stage features (Eq. 1).
            self.lmu = MultiScaleFusion(embed_dim=dim)
            self.svit_pooling = AttentionPooling(dim=dim, num_queries=num_latents,
                                                 num_heads=num_heads)

        # Semantic branch: frozen CLIP-ConvNeXt. num_classes=0 already removes the
        # classifier; the pooling is dropped as well since only forward_features
        # is used.
        if enable_convnext_branch:
            print(f"Loading frozen semantic backbone: {convnext_model}")
            self.convnext = timm.create_model(convnext_model, pretrained=True, num_classes=0)
            self.convnext.global_pool = nn.Identity()
            self.convnext.eval()
            for param in self.convnext.parameters():
                param.requires_grad = False

            # Inputs arrive normalized with ImageNet statistics; re-normalize them
            # to the statistics the frozen backbone was trained with. The input
            # statistics are constants, so they are not part of the state dict.
            cfg = self.convnext.default_cfg
            self.register_buffer('clip_mean', torch.tensor(cfg['mean']).view(1, 3, 1, 1).float())
            self.register_buffer('clip_std', torch.tensor(cfg['std']).view(1, 3, 1, 1).float())
            self.register_buffer('in_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                                 persistent=False)
            self.register_buffer('in_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                                 persistent=False)

            self.convnext_proj = nn.Conv2d(self.convnext.num_features, dim, 1)

        # MoE fusion applied before cross-stream self-attention.
        self.concat_norm2 = nn.LayerNorm(dim)
        self.post_moe_norm = nn.LayerNorm(dim)
        self.pre_pool_norm = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        if ffn_type == 'moe':
            self.moe_ffn = SparseMoE(dim=dim, num_experts=moe_num_experts,
                                     top_k=moe_top_k, expansion=moe_expansion)
        else:
            # Dense-FFN control: one SwiGLU whose hidden width matches the total
            # width of the top_k experts that the MoE activates per token.
            self.moe_ffn = ExpertFFN(dim=dim, expansion=moe_expansion * moe_top_k)

        self.final_attn_pool = AttentionPooling(dim=dim, num_queries=1, num_heads=num_heads)
        self.head = nn.Sequential(
            SwiGLU(dim, hidden_dim=dim * 2, out_dim=dim, dropout=0.1),
            nn.Linear(dim, num_classes),
        )

        # Segmentation decoders. The bridge concatenates the forensic map with the
        # upsampled post-fusion semantic tokens; the single-stream variants decode
        # from whichever map is available.
        if self.seg_bridge:
            self.joint_seg_decoder = self._conv_decoder(2 * dim, dim)
        elif enable_convnext_branch and not enable_sparsevit_branch:
            self.cnx_seg_decoder = self._conv_decoder(dim, dim)

    @staticmethod
    def _conv_decoder(in_channels, dim):
        """Two 3x3 Conv-GroupNorm-GELU blocks followed by a 1x1 projection."""
        return nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, dim), nn.GELU(),
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(32, dim // 2), nn.GELU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),
        )

    def _semantic_tokens(self, x):
        with torch.no_grad():
            x = x * (self.in_std / self.clip_std) + (self.in_mean - self.clip_mean) / self.clip_std
            feat = self.convnext.forward_features(x)
        feat = self.convnext_proj(feat)
        return feat.flatten(2).transpose(1, 2)

    def _upsample(self, logits):
        return F.interpolate(logits, size=(self.img_size, self.img_size),
                             mode="bilinear", align_corners=False)

    def forward(self, x):
        tokens = []
        forensic_map = None
        diversity_loss = x.new_tensor(0.0)

        if self.enable_sparsevit_branch:
            features = self.encoder(x)
            forensic_map = self.lmu.fuse_features(list(features.values()))
            svit_tokens, _ = self.svit_pooling(forensic_map.flatten(2).transpose(1, 2))
            diversity_loss = self.svit_pooling.diversity_loss(svit_tokens)
            tokens.append(svit_tokens)

        if self.enable_convnext_branch:
            cnx_tokens = self._semantic_tokens(x)
            tokens.append(cnx_tokens)

        n_svit = tokens[0].shape[1] if self.enable_sparsevit_branch else 0
        n_cnx = tokens[-1].shape[1] if self.enable_convnext_branch else 0

        # Z' = LN(Z + MoE(LN(Z))), then Z'' = LN(Z' + MSA(Z')).
        z = torch.cat(tokens, dim=1)
        aux_loss = None
        router_logits = None
        if self.ffn_type == 'moe':
            ffn_out, aux_loss, router_logits = self.moe_ffn(self.concat_norm2(z))
        else:
            ffn_out = self.moe_ffn(self.concat_norm2(z))
        z = self.post_moe_norm(z + ffn_out)
        z = self.pre_pool_norm(self.self_attn(z))

        # Localization.
        if self.seg_bridge:
            grid = int(round(n_cnx ** 0.5))
            cnx_fused = z[:, n_svit:n_svit + n_cnx, :]
            cnx_spatial = cnx_fused.transpose(1, 2).reshape(cnx_fused.shape[0], -1, grid, grid)
            cnx_up = F.interpolate(cnx_spatial, size=forensic_map.shape[-2:],
                                   mode="bilinear", align_corners=False)
            seg_logits = self._upsample(
                self.joint_seg_decoder(torch.cat([forensic_map, cnx_up], dim=1))
            )
        elif self.enable_sparsevit_branch:
            seg_logits = self._upsample(self.lmu.conv_last(forensic_map))
        else:
            grid = int(round(n_cnx ** 0.5))
            cnx_fused = z[:, :n_cnx, :]
            cnx_spatial = cnx_fused.transpose(1, 2).reshape(cnx_fused.shape[0], -1, grid, grid)
            seg_logits = self._upsample(self.cnx_seg_decoder(cnx_spatial))

        # Detection.
        pooled, _ = self.final_attn_pool(z)
        cls_logits = self.head(pooled.squeeze(1))

        result = {
            "logits": seg_logits,
            "cls_logits": cls_logits,
            "diversity_loss": diversity_loss,
        }
        if aux_loss is not None:
            result["moe_aux_loss"] = aux_loss
        if router_logits is not None:
            result["router_logits"] = router_logits
            result["token_layout"] = {"n_svit": int(n_svit), "n_cnx": int(n_cnx)}
        return result

    def train(self, mode=True):
        super().train(mode)
        if self.enable_convnext_branch:
            self.convnext.eval()
        return self

    def __str__(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (super().__str__()
                + f"\nAll params: {total / 1e6:.2f}M"
                + f"\nTrainable: {trainable / 1e6:.2f}M")
