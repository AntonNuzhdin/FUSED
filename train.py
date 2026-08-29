"""Train FUSED on the OpenSDID SD1.5 split (Hydra entrypoint, DDP via torchrun).

Hyperparameters are defined in ``configs/trainer/default.yaml``.

    torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
        trainer.out_dir=<dir>

Ablation arms are selected with model overrides, for example
``model.ffn_type=dense`` or ``model.moe_top_k=1``.
"""

import json
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from fused.checkpoint import load_fused
from fused.data.opensdi import collate
from fused.losses import fused_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_wandb(cfg, out_dir):
    """Start a W&B run on rank 0, or return None when logging is disabled."""
    wb = cfg.get("wandb")
    if wb is None or str(wb.get("mode", "disabled")) == "disabled":
        return None
    import wandb

    run = wandb.init(
        project=wb.get("project", "fused"),
        entity=wb.get("entity") or None,
        name=wb.get("name") or out_dir.name,
        group=wb.get("group") or None,
        tags=list(wb.get("tags") or []),
        notes=wb.get("notes") or None,
        mode=str(wb.get("mode")),
        dir=str(out_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"W&B: {run.url or '[offline]'} (mode={wb.get('mode')})", flush=True)
    return run


@torch.no_grad()
def validate(model, loader, device, world_size):
    """Per-image localization F1 on manipulated images plus detection metrics."""
    model.eval()
    f1_sum = 0.0
    n_fake = 0
    n_correct = n_total = 0
    scores, labels = [], []

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["seg_labels"].to(device, non_blocking=True)
        member = batch["member"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(imgs)

        seg_logits = result["logits"]
        if seg_logits.shape[-2:] != masks.shape[-2:]:
            seg_logits = F.interpolate(seg_logits, size=masks.shape[-2:],
                                       mode="bilinear", align_corners=False)
        preds = (seg_logits.sigmoid().float() >= 0.5).float()
        probs = result["cls_logits"].float().softmax(dim=1)[:, 1].cpu().numpy()

        for i in range(imgs.shape[0]):
            label = int(member[i].item() > 0)
            n_correct += int(int(probs[i] >= 0.5) == label)
            n_total += 1
            scores.append(float(probs[i]))
            labels.append(label)
            if label == 1:
                pred, gt = preds[i, 0], masks[i, 0]
                gt_empty = gt.sum().item() == 0
                pred_empty = pred.sum().item() == 0
                if gt_empty and pred_empty:
                    f1_sum += 1.0
                elif not (gt_empty or pred_empty):
                    tp = float((pred * gt).sum())
                    fp = float((pred * (1 - gt)).sum())
                    fn = float(((1 - pred) * gt).sum())
                    f1_sum += 2 * tp / (2 * tp + fp + fn + 1e-8)
                n_fake += 1
    model.train()

    if world_size > 1:
        totals = torch.tensor([f1_sum, float(n_fake), float(n_correct), float(n_total)],
                              dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        f1_sum, n_fake, n_correct, n_total = totals.tolist()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, (scores, labels))
        scores = [s for part in gathered for s in part[0]]
        labels = [l for part in gathered for l in part[1]]

    try:
        auc = float(roc_auc_score(labels, scores))
        det_f1 = float(f1_score(labels, [int(s >= 0.5) for s in scores]))
    except ValueError:
        auc = det_f1 = float("nan")
    return {
        "loc_f1": f1_sum / max(1, n_fake),
        "det_acc": n_correct / max(1, n_total),
        "det_f1": det_f1,
        "det_auc": auc,
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    tr = cfg.trainer
    out_dir = Path(tr.out_dir)

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    set_seed(int(tr.seed) + rank)

    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, out_dir / "config.yaml")
    wandb_run = init_wandb(cfg, out_dir) if is_main else None

    model = instantiate(cfg.model)
    model = load_fused(model, tr.checkpoint, device)

    start_epoch = 0
    global_step = 0
    if tr.resume:
        state = torch.load(tr.resume, map_location=device, weights_only=False)
        model.load_state_dict(state.get("state_dict", state), strict=False)
        start_epoch = int(state.get("epoch", -1)) + 1
        global_step = int(state.get("step", 0))
        if is_main:
            print(f"Resumed {tr.resume} at epoch {start_epoch}", flush=True)

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    unwrap = lambda m: m.module if isinstance(m, DDP) else m

    if wandb_run is not None and bool(cfg.wandb.get("watch_model", False)):
        import wandb

        wandb.watch(unwrap(model), log="all",
                    log_freq=int(cfg.wandb.get("watch_log_freq", 200)))

    train_ds = instantiate(cfg.dataset.train)
    val_ds = instantiate(cfg.dataset.val)
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=int(tr.seed)) if ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False) if ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=int(tr.batch_size), shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=int(tr.num_workers), pin_memory=True,
        drop_last=True, collate_fn=collate, persistent_workers=int(tr.num_workers) > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(tr.batch_size), shuffle=False, sampler=val_sampler,
        num_workers=int(tr.num_workers), pin_memory=True, collate_fn=collate,
        persistent_workers=int(tr.num_workers) > 0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tr.lr), weight_decay=float(tr.weight_decay),
    )

    accum = int(tr.accum)
    num_epochs = int(tr.epochs)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"train={len(train_ds)} val={len(val_ds)} trainable={n_params / 1e6:.2f}M "
              f"eff_batch={int(tr.batch_size) * world_size * accum} lr={tr.lr}", flush=True)

    history = []
    for epoch in range(start_epoch, num_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        started = time.time()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            imgs = batch["image"].to(device, non_blocking=True)
            targets = {
                "seg_labels": batch["seg_labels"].to(device, non_blocking=True),
                "edge": batch["edge"].to(device, non_blocking=True),
                "member": batch["member"].to(device, non_blocking=True),
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(imgs)
                loss, parts = fused_loss(
                    result, targets,
                    lambda_seg=float(tr.lambda_seg),
                    lambda_edge=float(tr.lambda_edge),
                    lambda_cls=float(tr.lambda_cls),
                    lambda_moe=float(tr.lambda_moe),
                    lambda_div=float(tr.lambda_div),
                )
            (loss / accum).backward()
            if (step + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            running += loss.item()

            if is_main and (step + 1) % int(tr.log_every) == 0:
                print(f"ep{epoch:02d} | {step + 1:>5d}/{len(train_loader)} | "
                      f"loss={loss.item():.4f} seg={parts['l_seg'].item():.4f} "
                      f"edge={parts['l_edge'].item():.4f} cls={parts['l_cls'].item():.4f} "
                      f"moe={parts['l_moe'].item():.6f} div={parts['l_div'].item():.6f} | "
                      f"avg={running / (step + 1):.4f}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": loss.item(),
                        "train/loss_seg": parts["l_seg"].item(),
                        "train/loss_edge": parts["l_edge"].item(),
                        "train/loss_cls": parts["l_cls"].item(),
                        "train/loss_moe": parts["l_moe"].item(),
                        "train/loss_div": parts["l_div"].item(),
                        "train/running_avg": running / (step + 1),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "global_step": global_step,
                    })

        metrics = validate(model, val_loader, device, world_size)
        if is_main:
            epoch_loss = running / max(1, len(train_loader))
            history.append({"epoch": epoch, "train_loss": epoch_loss, **metrics})
            print(f"=> ep{epoch:02d} | train_loss={epoch_loss:.4f} | "
                  f"val_loc_f1={metrics['loc_f1']:.4f} val_det_f1={metrics['det_f1']:.4f} "
                  f"val_det_acc={metrics['det_acc']:.4f} val_det_auc={metrics['det_auc']:.4f} | "
                  f"{time.time() - started:.0f}s", flush=True)
            ckpt = {"state_dict": unwrap(model).state_dict(),
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch, "step": global_step, "metrics": metrics}
            torch.save(ckpt, out_dir / f"epoch_{epoch:02d}.pth")
            torch.save(ckpt, out_dir / "latest.pth")
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))
            if wandb_run is not None:
                wandb_run.log({
                    "train/epoch_loss": epoch_loss,
                    "val/loc_f1": metrics["loc_f1"],
                    "val/det_f1": metrics["det_f1"],
                    "val/det_acc": metrics["det_acc"],
                    "val/det_auc": metrics["det_auc"],
                    "epoch": epoch,
                    "global_step": global_step,
                    "epoch_time_s": time.time() - started,
                })

    if is_main:
        print(f"Training complete. Evaluation checkpoint: {out_dir / 'latest.pth'}", flush=True)
        if wandb_run is not None:
            wandb_run.finish()
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
