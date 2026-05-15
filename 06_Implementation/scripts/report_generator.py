"""
Auto-diagnosis report generator based on EDL model outputs.

Generates clinical-grade reports from Dirichlet evidence parameters,
including per-class evidence, uncertainty, and clinical recommendations.

Based on: 03_Innovation/02_idea_proposals/auto_diagnosis_report.md
"""
import torch
import torch.nn.functional as F
from datetime import datetime


GRADE_NAMES = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}

DF_DESCRIPTIONS = {
    0: "No fluorosis lesions detected. Recommend routine dental check-up.",
    1: "Mild fluorosis — chalky streaks <25% of tooth surface. Recommend water quality improvement and monitoring.",
    2: "Moderate fluorosis — noticeable chalkiness with mild staining. Recommend professional diagnosis and aesthetic restoration evaluation.",
    3: "Severe fluorosis — extensive chalkiness, staining, enamel defects. Strongly recommend comprehensive treatment.",
}

SF_DESCRIPTIONS = {
    0: "No skeletal fluorosis changes observed. Normal bone density and structure.",
    1: "Mild skeletal fluorosis — slight trabecular thickening, possible interosseous membrane calcification. Recommend follow-up.",
    2: "Moderate skeletal fluorosis — evident ligament/ membrane calcification, increased bone density. Recommend clinical intervention assessment.",
    3: "Severe skeletal fluorosis — extensive ligament calcification, osteosclerosis, possible joint fusion. Multidisciplinary treatment needed.",
}


def generate_report(alpha, z, task="df", patient_id=None, extra_info=None):
    """Generate a formatted diagnosis report from EDL outputs.

    Args:
        alpha: Dirichlet params (K,) or (1, K)
        z: logits (K,) or (1, K)
        task: "df" or "sf"
        patient_id: optional patient identifier string
        extra_info: optional dict with task-specific metadata

    Returns:
        str: formatted diagnostic report
    """
    alpha = alpha.squeeze().detach().cpu()
    z = z.squeeze().detach().cpu()

    K = len(alpha)
    evidence = alpha - 1.0
    S = alpha.sum()
    probs = alpha / S
    u = K / S
    pred = torch.argmax(probs).item()

    H = -(probs * torch.log(probs + 1e-8)).sum().item()
    mode = (alpha - 1.0) / (S - K).clamp(min=1e-8)

    sorted_idx = torch.argsort(probs, descending=True)
    if len(sorted_idx) >= 2:
        confusion = 1.0 - (probs[sorted_idx[0]] - probs[sorted_idx[1]]).item()
    else:
        confusion = 0.0

    if u > 0.60:
        u_level = "HIGH"
    elif u > 0.30:
        u_level = "MEDIUM"
    else:
        u_level = "LOW"

    recommendation = _build_recommendation(pred, u, S, task)

    patient_str = f"PT-{patient_id}" if patient_id else "N/A"
    task_name = "Dental Fluorosis" if task == "df" else "Skeletal Fluorosis"
    backbone = "ViT-B" if task == "df" else "ResNet-50"

    lines = [
        "=" * 56,
        f"  Automated Diagnosis Report  |  {task_name}",
        "=" * 56,
        "",
        f"  Patient ID:    {patient_str}",
        f"  Report Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Model:         EDL-ORCU + {backbone}",
        "",
        "  --- Diagnosis Result ---",
        f"  Grade:         {GRADE_NAMES[pred]} (Grade {pred})",
        f"  Confidence:    {probs[pred].item():.1%}",
        f"  Uncertainty:   {u.item():.2%}  [{u_level}]",
        "",
        "  --- Class Probabilities ---",
    ]

    for k in range(K):
        bar = "#" * max(1, int(probs[k].item() * 40))
        marker = " <--" if k == pred else ""
        lines.append(f"  {GRADE_NAMES[k]:12s}: {probs[k].item():6.1%}  {bar}{marker}")

    lines += [
        "",
        "  --- Evidence Distribution ---",
        f"  Evidence(N)={evidence[0].item():.1f}  "
        f"E(Mi)={evidence[1].item():.1f}  "
        f"E(Mo)={evidence[2].item():.1f}  "
        f"E(Se)={evidence[3].item():.1f}",
        f"  Total Evidence S = {S.item():.1f} (higher = more certain)",
        "",
        "  --- Derived Statistics ---",
        f"  Entropy H = {H:.4f}",
        f"  Adjacent Confusion = {confusion:.3f}",
        f"  Dirichlet Mode = [{mode[0].item():.3f}, {mode[1].item():.3f}, "
        f"{mode[2].item():.3f}, {mode[3].item():.3f}]",
        "",
        "  --- Clinical Recommendation ---",
        f"  {recommendation}",
    ]

    if extra_info:
        lines += ["", "  --- Additional Information ---"]
        for k, v in extra_info.items():
            lines.append(f"  {k}: {v}")

    lines += [
        "",
        "=" * 56,
        "  Disclaimer: Automated research tool.",
        "  Clinical decisions require physician review.",
        "=" * 56,
    ]

    return "\n".join(lines)


def _build_recommendation(pred, u, S, task):
    if u > 0.60:
        role = "dentist" if task == "df" else "radiologist"
        return (
            f"HIGH UNCERTAINTY — recommend review by senior {role}. "
            f"(u={u.item():.2%}, S={S.item():.1f})"
        )

    if u > 0.30:
        return (
            "MODERATE UNCERTAINTY — recommend correlation with clinical examination. "
            f"(u={u.item():.2%}, S={S.item():.1f})"
        )

    desc = DF_DESCRIPTIONS if task == "df" else SF_DESCRIPTIONS
    return desc.get(pred, "No recommendation.") + f" (u={u.item():.2%}, S={S.item():.1f})"


def compute_report_metrics(alpha, targets, theta=0.30):
    """Compute Recommendation Accuracy and Safe Prediction Rate.

    Args:
        alpha: Dirichlet params (N, K)
        targets: ground truth labels (N,)
        theta: uncertainty threshold for safe/unsafe split

    Returns:
        dict with rec_acc, safe_pred_rate, unsafe_rate, safe_rate, theta
    """
    K = alpha.shape[-1]
    S = alpha.sum(dim=-1)
    u = K / S
    probs = alpha / S.unsqueeze(-1)
    preds = torch.argmax(probs, dim=-1)
    correct = (preds == targets).float()

    high_u_mask = (u > theta)
    low_u_mask = (u <= theta)

    n_high = high_u_mask.sum().item()
    n_low = low_u_mask.sum().item()
    n_total = len(targets)

    rec_acc = (1.0 - correct[high_u_mask].mean().item()) if n_high > 0 else 0.0
    safe_pred_rate = correct[low_u_mask].mean().item() if n_low > 0 else 0.0

    return {
        "rec_acc": rec_acc,
        "safe_pred_rate": safe_pred_rate,
        "unsafe_rate": n_high / n_total,
        "safe_rate": n_low / n_total,
        "theta": theta,
    }


@torch.no_grad()
def generate_batch_reports(model, dataloader, task="df", device="cuda", max_reports=3):
    """Generate sample reports for cases from a dataloader."""
    model.eval()
    reports = []

    for images, targets in dataloader:
        images = images.to(device)
        alpha, z = model(images)

        for i in range(min(len(images), max_reports - len(reports))):
            report = generate_report(
                alpha[i], z[i], task=task,
                patient_id=f"{targets[i].item():03d}",
            )
            reports.append(report)

        if len(reports) >= max_reports:
            break

    return reports


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, "..")

    from models import build_model
    from data import create_dataloaders

    p = argparse.ArgumentParser(description="Generate EDL diagnosis reports")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--task", type=str, default="df", choices=["df", "sf"])
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--num_samples", type=int, default=3)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(task=args.task)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    _, _, test_loader = create_dataloaders(args.data_root, task=args.task, batch_size=1)

    reports = generate_batch_reports(model, test_loader, task=args.task, device=device, max_reports=args.num_samples)

    output = "\n\n".join(reports)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Reports saved to {args.output}")
    else:
        print(output)
