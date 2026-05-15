"""
MC Dropout and Deep Ensemble uncertainty baselines for EDL comparison.

Converts MC variance and ensemble disagreement to pseudo-evidence/uncertainty
for fair comparison with EDL Dirichlet uncertainty via shared metrics.

Usage:
  python uncertainty_baselines.py --task df --data_root ../.. --method mc_dropout
  python uncertainty_baselines.py --task df --data_root ../.. --method both
"""
import os
import sys
import json
import copy
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import build_model
from eval import compute_metrics


# ---- MC Dropout ----

class MCDropoutModel(nn.Module):
    def __init__(self, backbone, num_classes=4, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(backbone.dim, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(self.backbone(x)))


@torch.no_grad()
def mc_predict(model, x, n_samples=30):
    model.train()  # keep dropout active
    stack = []
    for _ in range(n_samples):
        stack.append(F.softmax(model(x), dim=-1))
    probs = torch.stack(stack, dim=0)           # (N, B, K)
    mean = probs.mean(dim=0)                     # (B, K)
    var = probs.var(dim=0).mean(dim=-1)          # (B,) mean class variance
    u = var / (var + 1.0)                        # normalize to (0,1)
    return mean.argmax(dim=-1), mean, u


def train_mc_model(model, train_loader, val_loader, device, epochs=100):
    backbone_lr, head_lr = 1e-4, 1e-3
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith("backbone")], "lr": backbone_lr},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("backbone")], "lr": head_lr},
    ], weight_decay=0.05)

    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            loss = F.cross_entropy(model(images), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            total += len(targets)
        if total > 0 and correct / total > best_acc:
            best_acc = correct / total
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def eval_mc(model, loader, device, n_samples=30):
    all_p, all_t, all_u = [], [], []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        preds, probs, u = mc_predict(model, images, n_samples)
        all_p.append(probs.cpu())
        all_t.append(targets.cpu())
        all_u.append(u.cpu())

    probs = torch.cat(all_p, dim=0)
    targets = torch.cat(all_t, dim=0)
    u = torch.cat(all_u, dim=0)

    S = (1.0 / u.clamp(min=1e-6)).unsqueeze(-1)
    alpha = probs * S + 1.0
    z = torch.log(probs.clamp(min=1e-8))

    m = compute_metrics(alpha, z, targets)
    m["mean_uncertainty"] = u.mean().item()
    return m


# ---- Deep Ensemble ----

def train_ensemble(task, data_root, n_models=5, device="cuda", seed=42):
    from data import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root, task=task, batch_size=32 if task == "df" else 16)

    models = []
    for i in range(n_models):
        torch.manual_seed(seed + i * 100)
        model = build_model(task=task)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        best_acc, best_state = 0.0, None
        for epoch in range(50):
            model.train()
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                _, z = model(images)
                loss = F.cross_entropy(z, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            correct, total = 0, 0
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                _, z = model(images)
                correct += (z.argmax(dim=-1) == targets).sum().item()
                total += len(targets)
            if total > 0 and correct / total > best_acc:
                best_acc = correct / total
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        models.append(model)
        print(f"  Ensemble {i+1}/{n_models}: Val Acc={best_acc:.4f}")

    return models, test_loader


@torch.no_grad()
def eval_ensemble(models, loader, device):
    all_t, all_p = [], [[] for _ in range(len(models))]
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        all_t.append(targets.cpu())
        for i, model in enumerate(models):
            model.eval()
            _, z = model(images)
            all_p[i].append(F.softmax(z, dim=-1).cpu())

    targets = torch.cat(all_t, dim=0)
    stack = torch.stack([torch.cat(p, dim=0) for p in all_p], dim=0)  # (M, N, K)
    mean = stack.mean(dim=0)
    var = stack.var(dim=0).mean(dim=-1)
    u = var / (var + 1.0)

    S = (1.0 / u.clamp(min=1e-6)).unsqueeze(-1)
    alpha = mean * S + 1.0
    z = torch.log(mean.clamp(min=1e-8))

    m = compute_metrics(alpha, z, targets)
    m["mean_uncertainty"] = u.mean().item()
    return m


# ---- CLI ----

def main():
    p = argparse.ArgumentParser(description="Uncertainty baselines for EDL comparison")
    p.add_argument("--task", type=str, required=True, choices=["df", "sf"])
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--method", type=str, required=True,
                   choices=["mc_dropout", "deep_ensemble", "both"])
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "results", f"uncertainty_{args.task}")
    os.makedirs(output_dir, exist_ok=True)

    from data import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        args.data_root, task=args.task,
        batch_size=32 if args.task == "df" else 16)

    results = {}

    if args.method in ("mc_dropout", "both"):
        print(f"[MC Dropout] n={args.n_samples} forward passes")
        bb = build_model(task=args.task)
        mc_model = MCDropoutModel(bb.backbone, num_classes=4, dropout=0.1)
        mc_model.to(device)
        mc_model = train_mc_model(mc_model, train_loader, val_loader, device, epochs=100)
        mc_m = eval_mc(mc_model, test_loader, device, n_samples=args.n_samples)
        results["mc_dropout"] = {k: v for k, v in mc_m.items() if isinstance(v, (float, int))}
        print(f"  Acc={mc_m['acc']:.4f} ECE={mc_m['ece']:.4f} AUROC(u)={mc_m['auroc_u']:.4f}")

    if args.method in ("deep_ensemble", "both"):
        print(f"\n[Deep Ensemble] M={args.n_samples} models")
        models, _ = train_ensemble(args.task, args.data_root, n_models=args.n_samples,
                                    device=device, seed=args.seed)
        ens_m = eval_ensemble(models, test_loader, device)
        results["deep_ensemble"] = {k: v for k, v in ens_m.items() if isinstance(v, (float, int))}
        print(f"  Acc={ens_m['acc']:.4f} ECE={ens_m['ece']:.4f} AUROC(u)={ens_m['auroc_u']:.4f}")

    rp = os.path.join(output_dir, "uncertainty_baselines.json")
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] {rp}")


if __name__ == "__main__":
    main()
