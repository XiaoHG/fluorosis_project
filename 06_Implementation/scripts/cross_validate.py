"""
5-Fold Cross-Validation wrapper for EDL+ORCU fluorosis diagnosis.

Runs train_one_fold() K times with stratified splits,
aggregates mean ± std, and performs paired t-tests.
"""
import os
import sys
import json
import copy
import argparse
import numpy as np
from itertools import combinations
from scipy import stats

import torch
from torch.utils.data import Subset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import build_model
from losses import CombinedLoss
from eval import evaluate
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def kfold_split(dataset, k=5, seed=42):
    from sklearn.model_selection import StratifiedKFold
    if hasattr(dataset, "labels"):
        targets = [dataset.labels[sid] for sid in dataset.samples]
    else:
        targets = [s["grade"] for s in dataset.samples]
    targets = np.array(targets)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in skf.split(np.zeros(len(targets)), targets):
        splits.append((train_idx.tolist(), val_idx.tolist()))
    return splits


def train_one_fold(args, fold, train_indices, val_indices, device):
    defaults = {
        "df": {"batch_size": 32, "epochs": 100, "lr_backbone": 1e-4, "lr_head": 1e-3,
               "weight_decay": 0.05, "stage_1_epochs": 5, "stage_2_epochs": 30, "warmup_epochs": 5},
        "sf": {"batch_size": 16, "epochs": 150, "lr_backbone": 1e-4, "lr_head": 1e-3,
               "weight_decay": 0.05, "stage_1_epochs": 10, "stage_2_epochs": 45, "warmup_epochs": 10},
    }[args.task]
    defs = defaults

    epochs = args.epochs or defs["epochs"]
    batch_size = args.batch_size or defs["batch_size"]
    stage_1 = args.stage_1_epochs or defs["stage_1_epochs"]
    stage_2 = args.stage_2_epochs or defs["stage_2_epochs"]

    torch.manual_seed(args.seed + fold)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed + fold)
        torch.cuda.manual_seed_all(args.seed + fold)

    from data.dataset import DFDataset, SFDataset, get_transforms
    DS = DFDataset if args.task == "df" else SFDataset
    train_ds = DS(args.data_root, split="train",
                  transform=get_transforms(args.task, is_train=True))
    test_ds = DS(args.data_root, split="test",
                 transform=get_transforms(args.task, is_train=False))

    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(train_ds, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    import torch.nn as nn
    import torch.nn.functional as F

    model = build_model(task=args.task)
    model.to(device)

    if args.mode == "ce":
        class CELossWrapper(nn.Module):
            def forward(self, alpha, z, targets, epoch=None):
                loss = F.nll_loss(F.log_softmax(z, dim=-1), targets)
                return loss, {"stage": 0, "L_ce": loss.item()}
        criterion = CELossWrapper()
    elif args.mode == "edl":
        from losses.edl_loss import EDLLoss
        edl = EDLLoss(num_classes=4, kl_lambda=args.lambda_kl)
        class EDLWrapper(nn.Module):
            def forward(self, alpha, z, targets, epoch=None):
                loss = edl(alpha, targets, epoch, epochs)
                return loss, {"stage": 0, "L_edl": loss.item()}
        criterion = EDLWrapper()
    elif args.mode == "orcu":
        from losses.orcu_loss import ORCULoss
        orcu = ORCULoss(num_classes=4, t=args.orcu_t, lambda_reg=args.orcu_lambda_reg)
        class ORCUWrapper(nn.Module):
            def forward(self, alpha, z, targets, epoch=None):
                loss = orcu(z, targets)
                return loss, {"stage": 0, "L_orcu": loss.item()}
        criterion = ORCUWrapper()
    else:
        criterion = CombinedLoss(
            num_classes=4, lambda_orcu=args.lambda_orcu, lambda_kl=args.lambda_kl,
            orcu_t=args.orcu_t, stage_1_epochs=stage_1, stage_2_epochs=stage_2,
            total_epochs=epochs,
        )

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and n.startswith("backbone")]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("backbone")]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone or defs["lr_backbone"]},
        {"params": head_params, "lr": args.lr_head or defs["lr_head"]},
    ], weight_decay=defs["weight_decay"])

    # LR scheduler (matches train.py)
    warmup_epochs = defs["warmup_epochs"]
    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            alpha, z = model(images)
            loss, _ = criterion(alpha, z, targets, epoch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    scalar_metrics = {k: v for k, v in test_metrics.items()
                      if isinstance(v, (float, int))}
    scalar_metrics["fold"] = fold
    scalar_metrics["best_val_acc"] = best_val_acc
    return scalar_metrics


def aggregate_results(all_fold_results, metric_keys):
    summary = {}
    for key in metric_keys:
        values = [r[key] for r in all_fold_results if key in r]
        if len(values) < 2:
            continue
        summary[key] = {"mean": np.mean(values), "std": np.std(values), "values": values}
    return summary


def paired_ttest(method_results, metric, alpha=0.05):
    method_names = list(method_results.keys())
    results = []
    for m1, m2 in combinations(method_names, 2):
        v1 = [r[metric] for r in method_results[m1] if metric in r]
        v2 = [r[metric] for r in method_results[m2] if metric in r]
        if len(v1) < 3 or len(v2) < 3:
            continue
        t_stat, p_val = stats.ttest_rel(v1, v2)
        results.append({
            "methods": [m1, m2], "metric": metric,
            "t_statistic": float(t_stat), "p_value": float(p_val),
            "significant": bool(p_val < alpha),
        })
    return results


def main():
    p = argparse.ArgumentParser(description="5-Fold CV for EDL+ORCU fluorosis")
    p.add_argument("--task", type=str, default="df", choices=["df", "sf"])
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr_backbone", type=float, default=None)
    p.add_argument("--lr_head", type=float, default=None)
    p.add_argument("--lambda_orcu", type=float, default=0.5)
    p.add_argument("--lambda_kl", type=float, default=0.1)
    p.add_argument("--orcu_t", type=float, default=3.0)
    p.add_argument("--stage_1_epochs", type=int, default=None)
    p.add_argument("--stage_2_epochs", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", nargs="+", default=["ce", "edl", "orcu", "edl_orcu"],
                   choices=["ce", "edl", "orcu", "edl_orcu"])
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "results", f"cv_{args.task}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[CV] Task={args.task} | K={args.k} | Methods={args.methods}")

    from data.dataset import DFDataset, SFDataset, get_transforms
    DS = DFDataset if args.task == "df" else SFDataset
    train_ds = DS(args.data_root, split="train",
                  transform=get_transforms(args.task, is_train=True))
    splits = kfold_split(train_ds, k=args.k, seed=args.seed)
    print(f"[CV] {len(splits)} folds created")

    all_method_results = {}
    for method in args.methods:
        print(f"\n{'='*50}\n  Method: {method}\n{'='*50}")
        fold_results = []
        method_args = copy.deepcopy(args)
        method_args.mode = method

        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\n  Fold {fold+1}/{args.k} ...")
            metrics = train_one_fold(method_args, fold, train_idx, val_idx, device)
            fold_results.append(metrics)
            print(f"    Val Acc={metrics['best_val_acc']:.4f} | "
                  f"Test Acc={metrics['acc']:.4f} | F1={metrics['macro_f1']:.4f}")

        all_method_results[method] = fold_results

    metric_keys = ["acc", "macro_f1", "qwk", "ece", "sce", "pct_unimodal",
                   "u_ece", "auroc_u", "mean_uncertainty"]
    summary = {}
    for method in args.methods:
        summary[method] = aggregate_results(all_method_results[method], metric_keys)

    significance = {}
    for metric in metric_keys:
        for r in paired_ttest(all_method_results, metric):
            key = f"{r['methods'][0]}_vs_{r['methods'][1]}"
            significance.setdefault(key, {})[metric] = r

    output = {
        "task": args.task, "k": args.k, "methods": args.methods,
        "fold_results": all_method_results, "summary": summary,
        "significance": significance,
    }
    with open(os.path.join(output_dir, "cv_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*70}")
    print(f"CV Summary — {args.task.upper()} ({args.k}-fold)")
    for method in args.methods:
        print(f"\n  {method}:")
        for m in metric_keys:
            if m in summary.get(method, {}):
                s = summary[method][m]
                print(f"    {m:20s}: {s['mean']:.4f} ± {s['std']:.4f}")

    print(f"\n[CV] Results saved to {output_dir}/cv_results.json")


if __name__ == "__main__":
    main()
