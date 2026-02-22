"""Visualization of evaluation results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_metric_comparison(
    metrics: dict[str, dict[str, float]],
    output_dir: Path,
    metric_names: list[str] | None = None,
) -> Path:
    """Bar chart comparing metrics across variants.

    Returns path to saved figure.
    """
    if metric_names is None:
        all_names: set[str] = set()
        for vm in metrics.values():
            all_names.update(vm.keys())
        metric_names = sorted(all_names)

    variants = sorted(metrics.keys())
    n_metrics = len(metric_names)
    n_variants = len(variants)

    x = np.arange(n_metrics)
    width = 0.8 / n_variants

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 1.5), 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, variant in enumerate(variants):
        values = [metrics[variant].get(m, 0.0) for m in metric_names]
        offset = (i - n_variants / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=variant, color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, rotation=45)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("System Variant Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / "metric_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_radar_chart(
    metrics: dict[str, dict[str, float]],
    output_dir: Path,
) -> Path:
    """Radar chart for multi-dimensional comparison.

    Normalizes all metrics to [0, 1] range where higher = better.
    """
    lower_is_better = {
        "responsibility_mae", "ece", "brier_score",
        "hallucination_rate", "avg_entropy",
    }

    # Get common metrics
    all_names: set[str] = set()
    for vm in metrics.values():
        all_names.update(vm.keys())
    metric_names = sorted(all_names)

    variants = sorted(metrics.keys())

    # Normalize to [0, 1]
    normalized: dict[str, list[float]] = {}
    for variant in variants:
        values = []
        for m in metric_names:
            raw = metrics[variant].get(m, 0.0)
            # Find min/max across variants
            all_vals = [metrics[v].get(m, 0.0) for v in variants]
            vmin, vmax = min(all_vals), max(all_vals)
            if vmax - vmin > 1e-9:
                norm = (raw - vmin) / (vmax - vmin)
            else:
                norm = 0.5
            if m in lower_is_better:
                norm = 1.0 - norm
            values.append(norm)
        normalized[variant] = values

    # Plot
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for i, variant in enumerate(variants):
        values = normalized[variant] + normalized[variant][:1]
        ax.plot(angles, values, "o-", linewidth=2, label=variant, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("Normalized Performance (higher = better)", size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()

    output_path = output_dir / "radar_chart.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_stability_heatmap(
    stability_data: dict,
    output_dir: Path,
) -> Path:
    """Heatmap of primary cause consistency across reruns."""
    rerun_results = stability_data["rerun_results"]
    n_accidents = len(rerun_results)
    n_reruns = stability_data["n_reruns"]

    # Build consistency matrix: 1 if matches mode, 0 otherwise
    matrix = np.zeros((n_accidents, n_reruns))
    accident_labels: list[str] = []

    for i, reruns in enumerate(rerun_results):
        causes = [r.get("primary_cause", "") for r in reruns]
        accident_id = reruns[0].get("accident_id", f"A{i}") if reruns else f"A{i}"
        accident_labels.append(accident_id[:15])

        # Mode cause
        from collections import Counter
        counts = Counter(causes)
        mode_cause = counts.most_common(1)[0][0] if counts else ""
        for j, cause in enumerate(causes):
            matrix[i, j] = 1.0 if cause == mode_cause else 0.0

    fig, ax = plt.subplots(figsize=(max(6, n_reruns), max(6, n_accidents * 0.5)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_reruns))
    ax.set_xticklabels([f"Run {j+1}" for j in range(n_reruns)])
    ax.set_yticks(range(n_accidents))
    ax.set_yticklabels(accident_labels, fontsize=8)
    ax.set_title(f"Cause Consistency ({stability_data['variant']})")
    ax.set_xlabel("Rerun")
    ax.set_ylabel("Accident")
    fig.colorbar(im, label="Matches mode cause")
    fig.tight_layout()

    output_path = output_dir / f"stability_heatmap_{stability_data['variant']}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_causation_confusion_matrix(
    predictions: list[dict],
    ground_truths: list[dict],
    output_dir: Path,
    variant: str = "",
) -> Path:
    """Confusion matrix of predicted vs ground truth cause taxonomy IDs."""
    from evaluation.metrics.cause_taxonomy import classify_cause

    # Classify all predictions and ground truths
    pred_labels = [classify_cause(p.get("primary_cause", "")) for p in predictions]
    gt_labels = [
        g.get("primary_cause_taxonomy_id", classify_cause(g.get("primary_cause", "")))
        for g in ground_truths
    ]

    # Get unique labels that actually appear
    all_labels = sorted(set(pred_labels) | set(gt_labels))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    n = len(all_labels)

    # Build matrix
    matrix = np.zeros((n, n))
    for pl, gl in zip(pred_labels, gt_labels):
        matrix[label_to_idx.get(gl, 0), label_to_idx.get(pl, 0)] += 1

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.6)))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(all_labels, fontsize=7)
    ax.set_xlabel("Predicted Cause")
    ax.set_ylabel("Ground Truth Cause")
    title = "Causation Confusion Matrix"
    if variant:
        title += f" ({variant})"
    ax.set_title(title)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = int(matrix[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8,
                        color="white" if val > matrix.max() / 2 else "black")

    fig.colorbar(im)
    fig.tight_layout()

    suffix = f"_{variant}" if variant else ""
    output_path = output_dir / f"causation_confusion{suffix}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_per_category_metrics(
    per_category: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
    metric_name: str = "accuracy",
) -> Path:
    """Grouped bar chart of a metric broken down by accident category."""
    categories = sorted(per_category.keys())
    variants = sorted(
        set(v for cat_data in per_category.values() for v in cat_data.keys())
    )

    x = np.arange(len(categories))
    n_variants = len(variants)
    width = 0.8 / n_variants
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 2), 6))

    for i, variant in enumerate(variants):
        values = [
            per_category[cat].get(variant, {}).get(metric_name, 0.0)
            for cat in categories
        ]
        offset = (i - n_variants / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=variant, color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, rotation=45)

    ax.set_xlabel("Accident Category")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"{metric_name.replace('_', ' ').title()} by Category")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path = output_dir / f"per_category_{metric_name}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_retrieval_vs_accuracy(
    predictions: list[dict],
    ground_truths: list[dict],
    output_dir: Path,
    variant: str = "",
) -> Path:
    """Scatter plot of retrieval quality vs causation accuracy."""
    from evaluation.metrics.cause_taxonomy import classify_cause

    retrieval_scores: list[float] = []
    accuracies: list[float] = []

    for pred, gt in zip(predictions, ground_truths):
        chunks = pred.get("retrieved_chunks", [])
        if not chunks:
            continue
        max_score = max(c.get("score", 0.0) for c in chunks)
        retrieval_scores.append(max_score)

        pred_taxonomy = classify_cause(pred.get("primary_cause", ""))
        gt_taxonomy = gt.get("primary_cause_taxonomy_id", classify_cause(gt.get("primary_cause", "")))
        accuracies.append(1.0 if pred_taxonomy == gt_taxonomy else 0.0)

    if not retrieval_scores:
        # Nothing to plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No retrieval data available", ha="center", va="center", transform=ax.transAxes)
        output_path = output_dir / f"retrieval_vs_accuracy_{variant}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    fig, ax = plt.subplots(figsize=(8, 6))
    colors_map = ["#E53935" if a == 0 else "#43A047" for a in accuracies]

    # Add jitter to y for visibility (accuracy is binary 0/1)
    jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(accuracies))
    ax.scatter(retrieval_scores, np.array(accuracies) + jitter, c=colors_map, alpha=0.7, s=60, edgecolors="white")

    ax.set_xlabel("Max Retrieval Score")
    ax.set_ylabel("Correct (1) / Incorrect (0)")
    title = "Retrieval Quality vs Causation Accuracy"
    if variant:
        title += f" ({variant})"
    ax.set_title(title)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Incorrect", "Correct"])
    ax.grid(alpha=0.3)
    fig.tight_layout()

    suffix = f"_{variant}" if variant else ""
    output_path = output_dir / f"retrieval_vs_accuracy{suffix}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--text", action="store_true", help="Use text-mode result files")
    args = parser.parse_args()

    results_dir = args.results
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot comparative metrics if available
    for comp_name in ["comparative_metrics.json", "comparative_text_metrics.json"]:
        comp_path = results_dir / comp_name
        if comp_path.exists():
            with open(comp_path) as f:
                metrics = json.load(f)
            print(f"Saved: {plot_metric_comparison(metrics, results_dir)}")
            print(f"Saved: {plot_radar_chart(metrics, results_dir)}")

    # Plot stability if available
    for stab_path in results_dir.glob("*_stability_*.json"):
        with open(stab_path) as f:
            stab_data = json.load(f)
        print(f"Saved: {plot_stability_heatmap(stab_data, results_dir)}")

    # Plot confusion matrices and retrieval vs accuracy for text mode
    pred_suffix = "_text_predictions.json" if args.text else "_predictions.json"
    gt_suffix = "_text_ground_truths.json" if args.text else "_ground_truths.json"

    for variant in ["S1", "S2", "S3"]:
        pred_path = results_dir / f"{variant}{pred_suffix}"
        gt_path = results_dir / f"{variant}{gt_suffix}"
        if pred_path.exists() and gt_path.exists():
            with open(pred_path) as f:
                preds = json.load(f)
            with open(gt_path) as f:
                gts = json.load(f)
            print(f"Saved: {plot_causation_confusion_matrix(preds, gts, results_dir, variant)}")
            if variant in ("S2", "S3"):
                print(f"Saved: {plot_retrieval_vs_accuracy(preds, gts, results_dir, variant)}")

    # Per-category plot
    cat_path = results_dir / "per_category_comparison.json"
    if cat_path.exists():
        with open(cat_path) as f:
            per_cat = json.load(f)
        print(f"Saved: {plot_per_category_metrics(per_cat, results_dir, 'accuracy')}")
        print(f"Saved: {plot_per_category_metrics(per_cat, results_dir, 'responsibility_mae')}")
