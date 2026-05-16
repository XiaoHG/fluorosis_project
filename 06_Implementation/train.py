"""
Training script for fluorosis diagnosis with 5 loss modes.

Modes:
  ce          — standard cross-entropy (one-hot targets)
  cumulative  — Coral/CORN K-1 binary CE (ordinal baseline)
  sord        — SORD soft-encoded CE (ordinal, no regularization)
  edl         — evidential deep learning (one-hot targets + KL)
  edl_orcu    — full EDL + SORD + hinge ordinal regularization
"""
import os
import sys
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from data import create_dataloaders
from models import build_model
from losses import CombinedLoss
from eval import evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Train fluorosis diagnosis model")
    p.add_argument("--task", type=str, default="df", choices=["df", "sf"])
    p.add_argument("--data_root", type=str, required=True, help="Path to project root (contains data/)")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr_backbone", type=float, default=None)
    p.add_argument("--lr_head", type=float, default=None)
    p.add_argument("--lambda_orcu", type=float, default=0.5)
    p.add_argument("--lambda_kl", type=float, default=0.1)
    p.add_argument("--orcu_t", type=float, default=3.0)
    p.add_argument("--orcu_lambda_reg", type=float, default=0.01,
                   help="Weight for ORCU hinge ordinal regularization")
    p.add_argument("--stage_1_epochs", type=int, default=None)
    p.add_argument("--stage_2_epochs", type=int, default=None)
    p.add_argument("--mode", type=str, default="edl_orcu",
                   choices=["edl_orcu", "ce", "cumulative", "sord", "edl", "orcu"],
                   help="ce|cumulative|sord|edl|orcu|edl_orcu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_defaults(task):
    if task == "df":
        return {
            "batch_size": 32, "epochs": 100, "lr_backbone": 1e-4, "lr_head": 1e-3,
            "weight_decay": 0.05, "stage_1_epochs": 5, "stage_2_epochs": 30, "warmup_epochs": 5,
        }
    else:
        return {
            "batch_size": 16, "epochs": 150, "lr_backbone": 1e-4, "lr_head": 1e-3,
            "weight_decay": 0.05, "stage_1_epochs": 10, "stage_2_epochs": 45, "warmup_epochs": 10,
        }


def build_optimizer(model, defaults, args):
    lr_bb = args.lr_backbone or defaults["lr_backbone"]
    lr_hd = args.lr_head or defaults["lr_head"]
    wd = defaults["weight_decay"]
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("backbone") else head_params).append(p)
    return optim.AdamW([
        {"params": backbone_params, "lr": lr_bb},
        {"params": head_params, "lr": lr_hd},
    ], weight_decay=wd)


def build_scheduler(optimizer, defaults, args):
    epochs = args.epochs or defaults["epochs"]
    warmup = defaults["warmup_epochs"]
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup)
    return SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])


def main():
    args = parse_args()
    defaults = get_defaults(args.task)

    batch_size = args.batch_size or defaults["batch_size"]
    epochs = args.epochs or defaults["epochs"]
    stage_1 = args.stage_1_epochs or defaults["stage_1_epochs"]
    stage_2 = args.stage_2_epochs or defaults["stage_2_epochs"]

    torch.manual_seed(args.seed)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "checkpoints",
        f"{args.task}_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Task: {args.task} | Mode: {args.mode} | Device: {device} | Output: {output_dir}")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_root, task=args.task, batch_size=batch_size, num_workers=args.num_workers)
    print(f"[INFO] Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    # Model
    model = build_model(task=args.task, mode=args.mode)
    model.to(device)
    print(f"[INFO] Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Criterion — mode-dependent
    mode = args.mode
    if mode == "ce":
        class CEWrapper(nn.Module):
            def forward(self, alpha, z, targets, epoch=None):
                loss = F.nll_loss(F.log_softmax(z, dim=-1), targets)
                return loss, {"stage": 0, "L_ce": loss.item()}
        criterion = CEWrapper()

    elif mode == "cumulative":
        from losses.cumulative_loss import CumulativeLoss
        cum_loss_fn = CumulativeLoss(num_classes=4)
        class CumWrapper(nn.Module):
            def forward(self, alpha, z, targets, epoch=None):
                if hasattr(model, 'ordinal_logits') and model.ordinal_logits is not None:
                    ol = model.ordinal_logits
                else:
                    ol = z
                loss = cum_loss_fn(ol, targets)
                return loss, {"stage": 0, "L_cum": loss.item()}
        criterion = CumWrapper()

    elif mode == "sord":
        from losses.orcu_loss import ORCULoss
        sord_fn = ORCULoss(num_classes=4, t=args.orcu_t, lambda_reg=0.0)
        class SORDWrapper(nn.Module):
            def forward(self, alpha, z, targets, epoch=None):
                loss = sord_fn(z, targets)
                return loss, {"stage": 0, "L_sord": loss.item()}
        criterion = SORDWrapper()

    elif mode == "orcu":
        from losses.orcu_loss import ORCULoss
        orcu_fn = ORCULoss(num_classes=4, t=args.orcu_t, lambda_reg=args.orcu_lambda_reg)
        class ORCUWrapper(nn.Module):
            def forward(self, alpha, z, targets, epoch=None):
                loss = orcu_fn(z, targets)
                return loss, {"stage": 0, "L_orcu": loss.item()}
        criterion = ORCUWrapper()

    elif mode == "edl":
        from losses.edl_loss import EDLLoss
        edl_fn = EDLLoss(num_classes=4, kl_lambda=args.lambda_kl)
        class EDLWrapper(nn.Module):
            def forward(self, alpha, z, targets, epoch=None):
                loss = edl_fn(alpha, targets, epoch, epochs)
                return loss, {"stage": 0, "L_edl": loss.item()}
        criterion = EDLWrapper()

    else:  # edl_orcu
        criterion = CombinedLoss(
            num_classes=4, lambda_orcu=args.lambda_orcu, lambda_kl=args.lambda_kl,
            orcu_t=args.orcu_t, orcu_lambda_reg=args.orcu_lambda_reg,
            stage_1_epochs=stage_1, stage_2_epochs=stage_2,
            total_epochs=epochs)

    # Optimizer & scheduler
    optimizer = build_optimizer(model, defaults, args)
    scheduler = build_scheduler(optimizer, defaults, args)

    # Training loop
    best_val_acc = 0.0
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = {"L_ce": 0.0, "L_edl": 0.0, "L_orcu": 0.0, "L_cum": 0.0,
                        "L_sord": 0.0, "L_total": 0.0}
        epoch_stage, n_batches = 0, 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            alpha, z = model(images)
            loss, loss_info = criterion(alpha, z, targets, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_stage = loss_info.get("stage", 0)
            for k in epoch_losses:
                if k in loss_info:
                    epoch_losses[k] += loss_info[k]
            n_batches += 1

        scheduler.step()
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        val_metrics = evaluate(model, val_loader, device)
        is_best = val_metrics["acc"] > best_val_acc
        if is_best:
            best_val_acc = val_metrics["acc"]

        history.append({
            "epoch": epoch, "stage": epoch_stage,
            "train_loss": epoch_losses,
            "val_metrics": {k: v for k, v in val_metrics.items()
                            if not k.startswith("evidence_class_")},
        })

        loss_key = next((k for k in ["L_total", "L_ce", "L_cum", "L_sord", "L_edl", "L_orcu"]
                         if epoch_losses.get(k, 0) != 0), "L_total")
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1:3d}/{epochs}] S={epoch_stage} | "
                  f"{loss_key}={epoch_losses.get(loss_key, 0):.4f} | "
                  f"Val Acc={val_metrics['acc']:.4f} F1={val_metrics['macro_f1']:.4f} "
                  f"QWK={val_metrics['qwk']:.4f} ECE={val_metrics['ece']:.4f} "
                  f"Unim={val_metrics['pct_unimodal']:.2%} U={val_metrics['mean_uncertainty']:.3f}")

        ckpt = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "best_val_acc": best_val_acc,
            "args": vars(args), "history": history,
        }
        torch.save(ckpt, os.path.join(output_dir, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(output_dir, "best.pt"))

    # Final test evaluation
    best_ckpt = torch.load(os.path.join(output_dir, "best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    print(f"\n[Test Results]")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in test_metrics.items() if isinstance(v, (float, int))}, f, indent=2)
    print(f"[INFO] Metrics saved to {metrics_path}")

    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
