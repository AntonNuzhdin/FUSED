"""SparseViT backbone encoder.

Derived from the official implementation released by the SparseViT authors:

    Su, L.; Ma, X.; Zhu, X.; Niu, C.; Lei, Z.; and Zhou, J.-Z. 2025. Can We Get
    Rid of Handcrafted Feature Extractors? SparseViT: Nonsemantics-Centered,
    Parameter-Efficient Image Manipulation Localization Through Spare-Coding
    Transformer. In Proceedings of the AAAI Conference on Artificial
    Intelligence, volume 39, 7024-7032.

Copyright remains with the original authors and this file is used under the
terms of that project's license.
"""

from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

layer_scale = True
init_value = 1e-6


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def block(x, block_size):
    B, H, W, C = x.shape
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.reshape(B, Hp // block_size, block_size, Wp // block_size, block_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x, H, Hp, C


def unblock(x, Ho):
    B, H, W, win_H, win_W, C = x.shape
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, H * win_H, W * win_W, C)
    Wp = Hp = H * win_H
    Wo = Ho
    if Hp > Ho or Wp > Wo:
        x = x[:, :Ho, :Wo, :].contiguous()
    return x


def alter_sparse(x, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    assert x.shape[1] % sparse_size == 0 & x.shape[2] % sparse_size == 0, \
        'image size should be divisible by block_size'
    grid_size = x.shape[1] // sparse_size
    out, H, Hp, C = block(x, grid_size)
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = out.reshape(-1, sparse_size, sparse_size, C)
    out = out.permute(0, 3, 1, 2)
    return out, H, Hp, C


def alter_unsparse(x, H, Hp, C, sparse_size=8):
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, Hp // sparse_size, Hp // sparse_size, sparse_size, sparse_size, C)
    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    out = unblock(x, H)
    out = out.permute(0, 3, 1, 2)
    return out


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, sparse_size=0, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls = layer_scale
        self.sparse_size = sparse_size
        if self.ls:
            self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x_befor = x.flatten(2).transpose(1, 2)
        B, N, H, W = x.shape
        if self.ls:
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            x = x_befor + self.drop_path(self.gamma_1 * x)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))
        else:
            x, Ho, Hp, C = alter_sparse(x, self.sparse_size)
            Bf, Nf, Hf, Wf = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.attn(self.norm1(x))
            x = x.transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
            x = alter_unsparse(x, Ho, Hp, C, self.sparse_size)
            x = x.flatten(2).transpose(1, 2)
            x = x_befor + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class SparseViT(nn.Module):
    def __init__(self, layers=[5, 8, 20, 7], img_size=224, in_chans=3,
                 s_blocks3=[8, 4, 2, 1], s_blocks4=[2, 1],
                 embed_dim=[64, 128, 320, 512], head_dim=64, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=None, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        num_heads = [dim // head_dim for dim in embed_dim]

        self.blocks1 = nn.ModuleList([
            CBlock(dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio,
                   qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                   attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(layers[0])])
        self.norm1 = norm_layer(embed_dim[0])

        self.blocks2 = nn.ModuleList([
            CBlock(dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio,
                   qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                   attn_drop=attn_drop_rate, drop_path=dpr[i + layers[0]], norm_layer=norm_layer)
            for i in range(layers[1])])
        self.norm2 = norm_layer(embed_dim[1])

        self.blocks3 = nn.ModuleList()
        for i in range(layers[2]):
            blk = SABlock(
                dim=embed_dim[2], num_heads=num_heads[2],
                sparse_size=32 // s_blocks3[i // 5], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i + layers[0] + layers[1]],
                norm_layer=norm_layer)
            self.blocks3.append(blk)
        self.norm3 = norm_layer(embed_dim[2])

        self.blocks4 = nn.ModuleList()
        for i in range(layers[3]):
            blk = SABlock(
                dim=embed_dim[3], num_heads=num_heads[3],
                sparse_size=16 // s_blocks4[i // 4], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i + layers[0] + layers[1] + layers[2]],
                norm_layer=norm_layer)
            self.blocks4.append(blk)
        self.norm4 = norm_layer(embed_dim[3])

        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self._uniformer_init_weights()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _uniformer_init_weights(self):
        if self.pretrained_path is not None:
            state_dict = torch.load(self.pretrained_path, map_location='cpu', weights_only=False)
            new_state_dict = {}
            for k, v in state_dict['model'].items():
                if k.startswith('backbone.'):
                    new_key = k[len('backbone.'):]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            self.load_state_dict(new_state_dict, strict=False)
            print('load pretrained weights from \'{}\'.'.format(self.pretrained_path))

    def forward_features(self, x):
        outputs = {}
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for index, blk in enumerate(self.blocks3):
            x = blk(x)
            if (index + 1) % 5 == 0 and (index + 1) // 5 != 4:
                outputs.update({"third" + str((index + 1) // 5): x})
        x_out = self.norm3(x.permute(0, 2, 3, 1))
        outputs.update({"third": x_out.permute(0, 3, 1, 2).contiguous()})
        x = self.patch_embed4(x)
        for index, blk in enumerate(self.blocks4):
            x = blk(x)
            if (index + 1) % 4 == 0:
                outputs.update({"last" + str((index + 1) // 4): x})
        x_out = self.norm4(x.permute(0, 2, 3, 1))
        outputs.update({"last": x_out.permute(0, 3, 1, 2).contiguous()})
        return outputs

    def forward(self, x):
        outputs = self.forward_features(x)
        return outputs
