"""
Ablation experiment runner for EDL+ORCU fluorosis diagnosis.

Runs 6 planned ablation studies (A1–A6) by invoking train.py as subprocess.
Produces consolidated ablation_summary.json with rankings.

Usage:
  python ablation_runner.py --task df --data_root ../..  [--subset A1,A3]
"""
import os
import sys
import json
import subprocess
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ABLATION_CONFIGS = {
    "A1": {
        "description": "Loss function: CE vs EDL vs EDL+ORCU",
        "runs": [
            {"name": "A1_ce", "mode": "ce"},
            {"name": "A1_edl", "mode": "edl"},
            {"name": "A1_edl_orcu", "mode": "edl_orcu", "lambda_orcu": 0.5},
        ],
    },
    "A3": {
        "description": "Lambda ORCU sweep",
        "runs": [
            {"name": f"A3_lam_{lam}", "mode": "edl_orcu", "lambda_orcu": lam}
            for lam in [0.1, 0.3, 0.5, 0.7, 1.0]
        ],
    },
    "A4": {
        "description": "ORCU temperature (tau) sweep",
        "runs": [
            {"name": f"A4_tau_{tau}", "mode": "edl_orcu", "orcu_t": tau}
            for tau in [1.0, 3.0, 5.0, 10.0]
        ],
    },
    "A5": {
        "description": "KL annealing on/off",
        "runs": [
            {"name": "A5_kl_on",  "mode": "edl", "lambda_kl": 0.1},
            {"name": "A5_kl_off", "mode": "edl", "lambda_kl": 0.0},
        ],
    },
}


def run_single(train_script, data_root, task, run_cfg, output_dir, defaults):
    name = run_cfg["name"]
    cmd = [
        sys.executable, train_script,
        "--task", task,
        "--data_root", data_root,
        "--mode", run_cfg.get("mode", "edl_orcu"),
        "--output_dir", os.path.join(output_dir, name),
        "--seed", "42",
    ]
    for p in ["batch_size", "epochs", "lr_backbone", "lr_head",
              "stage_1_epochs", "stage_2_epochs"]:
        if p in defaults and defaults[p] is not None:
            cmd += [f"--{p}", str(defaults[p])]
    for p in ["lambda_orcu", "lambda_kl", "orcu_t"]:
        if p in run_cfg:
            cmd += [f"--{p}", str(run_cfg[p])]

    print(f"\n[Ablation] {name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    mp = os.path.join(output_dir, name, "test_metrics.json")
    if os.path.exists(mp):
        with open(mp) as f:
            return json.load(f)
    print(f"  [WARN] No test_metrics.json; stderr: {result.stderr[-300:]}")
    return {}


def compute_rankings(all_results):
    metrics = ["acc", "macro_f1", "qwk", "ece", "pct_unimodal", "u_ece"]
    directions = {"acc": 1, "macro_f1": 1, "qwk": 1, "ece": -1, "pct_unimodal": 1, "u_ece": -1}
    rankings = {}
    for m in metrics:
        sorted_runs = sorted(
            [(n, v.get(m)) for n, v in all_results.items() if isinstance(v, dict) and m in v],
            key=lambda x: x[1] or 0, reverse=directions[m] > 0)
        rankings[m] = [{"rank": i+1, "name": n, "value": val} for i, (n, val) in enumerate(sorted_runs)]
    return rankings


def main():
    p = argparse.ArgumentParser(description="Ablation runner for EDL+ORCU")
    p.add_argument("--task", type=str, required=True, choices=["df", "sf"])
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--subset", type=str, default="A1,A3,A4,A5")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--skip_missing", action="store_true")
    args = p.parse_args()

    defaults = {
        "df": {"batch_size": 32, "epochs": 100, "stage_1_epochs": 5, "stage_2_epochs": 30},
        "sf": {"batch_size": 16, "epochs": 150, "stage_1_epochs": 10, "stage_2_epochs": 45},
    }[args.task]
    if args.epochs:
        defaults["epochs"] = args.epochs

    train_script = os.path.join(os.path.dirname(__file__), "..", "train.py")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "results",
        f"ablation_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)

    ablation_ids = [a.strip() for a in args.subset.split(",")]
    print(f"[Ablation] Task={args.task} | IDs={ablation_ids}")

    all_results = {}
    for aid in ablation_ids:
        if aid in ("A2", "A6"):
            print(f"  [{aid}] Requires dataset-level GT change — see experiment plan.")
            continue
        if aid not in ABLATION_CONFIGS:
            print(f"  [SKIP] Unknown: {aid}")
            continue
        cfg = ABLATION_CONFIGS[aid]
        print(f"\n{'='*50}\n  {aid}: {cfg['description']}\n{'='*50}")
        for run_cfg in cfg["runs"]:
            out = os.path.join(output_dir, run_cfg["name"], "test_metrics.json")
            if args.skip_missing and os.path.exists(out):
                with open(out) as f:
                    all_results[run_cfg["name"]] = json.load(f)
                print(f"  [{run_cfg['name']}] SKIP (exists)")
                continue
            r = run_single(train_script, args.data_root, args.task, run_cfg, output_dir, defaults)
            all_results[run_cfg["name"]] = r
            if r:
                print(f"  [{run_cfg['name']}] Acc={r.get('acc',0):.4f} F1={r.get('macro_f1',0):.4f} QWK={r.get('qwk',0):.4f} ECE={r.get('ece',0):.4f}")
            else:
                print(f"  [{run_cfg['name']}] FAILED")

    rankings = compute_rankings(all_results)
    summary = {"task": args.task, "ablation_ids": ablation_ids,
               "results": all_results, "rankings": rankings}
    sp = os.path.join(output_dir, "ablation_summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}\nAblation Rankings — {args.task.upper()}")
    for m, entries in rankings.items():
        print(f"\n  {m}:")
        for e in entries[:5]:
            v = f"{e['value']:.4f}" if e['value'] is not None else "N/A"
            print(f"    {e['rank']:2d}. {e['name']:30s} = {v}")
    print(f"\n[Results] {sp}")


if __name__ == "__main__":
    main()
