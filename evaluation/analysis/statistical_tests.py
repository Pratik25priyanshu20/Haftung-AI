"""Statistical significance tests for comparing system variants."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def paired_t_test(
    values_a: list[float],
    values_b: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired t-test for two sets of per-sample metrics.

    Returns test statistic, p-value, and significance flag.
    """
    a = np.array(values_a)
    b = np.array(values_b)
    t_stat, p_value = stats.ttest_rel(a, b)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "n": len(values_a),
        "mean_diff": float(np.mean(a - b)),
        "std_diff": float(np.std(a - b, ddof=1)),
    }


def wilcoxon_test(
    values_a: list[float],
    values_b: list[float],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Wilcoxon signed-rank test (non-parametric alternative)."""
    a = np.array(values_a)
    b = np.array(values_b)

    diff = a - b
    if np.all(diff == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "alpha": alpha,
            "n": len(values_a),
        }

    stat, p_value = stats.wilcoxon(a, b)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "n": len(values_a),
    }


def cohens_d(values_a: list[float], values_b: list[float]) -> float:
    """Compute Cohen's d effect size for paired samples."""
    a = np.array(values_a)
    b = np.array(values_b)
    diff = a - b
    d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
    return d


def bonferroni_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Apply Bonferroni correction for multiple comparisons.

    Returns list of dicts with original p-value, corrected p-value,
    and significance flag.
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests if n_tests > 0 else alpha

    return [
        {
            "original_p": p,
            "corrected_p": min(p * n_tests, 1.0),
            "significant": p < corrected_alpha,
            "corrected_alpha": corrected_alpha,
            "n_tests": n_tests,
        }
        for p in p_values
    ]


def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap confidence interval for the mean."""
    arr = np.array(values)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = (1 + confidence) / 2 * 100

    return {
        "mean": float(arr.mean()),
        "ci_lower": float(np.percentile(boot_means, lower_pct)),
        "ci_upper": float(np.percentile(boot_means, upper_pct)),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def compute_per_sample_accuracy(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> list[float]:
    """Compute per-sample binary accuracy for primary cause."""
    return [
        1.0 if p.get("primary_cause", "").lower() == g.get("primary_cause", "").lower() else 0.0
        for p, g in zip(predictions, ground_truths)
    ]


def compute_per_sample_taxonomy_accuracy(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> list[float]:
    """Compute per-sample binary accuracy using cause taxonomy."""
    from evaluation.metrics.cause_taxonomy import classify_cause

    results: list[float] = []
    for pred, gt in zip(predictions, ground_truths):
        pred_taxonomy = classify_cause(pred.get("primary_cause", ""))
        gt_taxonomy = gt.get("primary_cause_taxonomy_id", "")
        if not gt_taxonomy:
            gt_taxonomy = classify_cause(gt.get("primary_cause", ""))
        results.append(1.0 if pred_taxonomy == gt_taxonomy else 0.0)
    return results


def compute_per_sample_responsibility_ae(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> list[float]:
    """Compute per-sample absolute error for responsibility assignment."""
    errors: list[float] = []
    for pred, gt in zip(predictions, ground_truths):
        pred_resp = {r.get("party", ""): r.get("percentage", 0.0) for r in pred.get("responsibility", [])}
        gt_resp = {r.get("party", ""): r.get("percentage", 0.0) for r in gt.get("responsibility", [])}

        all_parties = set(pred_resp) | set(gt_resp)
        if not all_parties:
            errors.append(0.0)
            continue
        ae = sum(abs(pred_resp.get(p, 0.0) - gt_resp.get(p, 0.0)) for p in all_parties) / len(all_parties)
        errors.append(ae)
    return errors


def compute_per_sample_hallucination(
    predictions: list[dict[str, Any]],
) -> list[float]:
    """Compute per-sample hallucination rate (fraction of unsupported claims)."""
    results: list[float] = []
    for pred in predictions:
        claims = pred.get("claims", [])
        if not claims:
            results.append(0.0)
            continue
        unsupported = sum(1 for c in claims if not c.get("supported", True))
        results.append(unsupported / len(claims))
    return results


def compute_per_sample_factors_f1(
    predictions: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> list[float]:
    """Compute per-sample F1 for contributing factors."""
    results: list[float] = []
    for pred, gt in zip(predictions, ground_truths):
        pred_factors = {f.get("category", "").lower() for f in pred.get("contributing_factors", [])}
        gt_factors = {f.get("category", "").lower() for f in gt.get("contributing_factors", [])}

        tp = len(pred_factors & gt_factors)
        fp = len(pred_factors - gt_factors)
        fn = len(gt_factors - pred_factors)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.append(f1)
    return results


def run_all_tests(
    results_dir: Path,
    alpha: float = 0.05,
    text_mode: bool = False,
) -> dict[str, Any]:
    """Run significance tests between all variant pairs.

    Expects {variant}_predictions.json and {variant}_ground_truths.json files,
    or {variant}_text_predictions.json for text mode.
    """
    variants = ["S1", "S2", "S3"]
    pred_suffix = "_text_predictions.json" if text_mode else "_predictions.json"
    gt_suffix = "_text_ground_truths.json" if text_mode else "_ground_truths.json"

    per_sample: dict[str, dict[str, list[float]]] = {}

    for variant in variants:
        pred_path = results_dir / f"{variant}{pred_suffix}"
        gt_path = results_dir / f"{variant}{gt_suffix}"
        if not pred_path.exists():
            continue

        with open(pred_path) as f:
            preds = json.load(f)
        with open(gt_path) as f:
            gts = json.load(f)

        per_sample[variant] = {
            "accuracy": compute_per_sample_accuracy(preds, gts),
            "taxonomy_accuracy": compute_per_sample_taxonomy_accuracy(preds, gts),
            "responsibility_ae": compute_per_sample_responsibility_ae(preds, gts),
            "hallucination": compute_per_sample_hallucination(preds),
            "factors_f1": compute_per_sample_factors_f1(preds, gts),
        }

    # Pairwise tests
    test_results: dict[str, Any] = {
        "pairwise": {},
        "confidence_intervals": {},
        "effect_sizes": {},
        "bonferroni": {},
    }
    available = [v for v in variants if v in per_sample]

    # Collect all p-values for Bonferroni correction
    all_p_values: list[float] = []
    all_p_labels: list[str] = []

    for i, va in enumerate(available):
        # CIs per variant
        for metric_name in per_sample[va]:
            key = f"{va}_{metric_name}"
            test_results["confidence_intervals"][key] = bootstrap_confidence_interval(
                per_sample[va][metric_name]
            )

        for vb in available[i + 1:]:
            pair_key = f"{va}_vs_{vb}"
            pair_results: dict[str, Any] = {}

            for metric_name in per_sample[va]:
                if metric_name not in per_sample[vb]:
                    continue
                vals_a = per_sample[va][metric_name]
                vals_b = per_sample[vb][metric_name]
                if len(vals_a) != len(vals_b):
                    continue

                t_test = paired_t_test(vals_a, vals_b, alpha)
                w_test = wilcoxon_test(vals_a, vals_b, alpha)
                d = cohens_d(vals_a, vals_b)

                pair_results[metric_name] = {
                    "paired_t_test": t_test,
                    "wilcoxon": w_test,
                    "cohens_d": d,
                }

                all_p_values.append(t_test["p_value"])
                all_p_labels.append(f"{pair_key}_{metric_name}")

                # Effect sizes
                test_results["effect_sizes"][f"{pair_key}_{metric_name}"] = {
                    "cohens_d": d,
                    "interpretation": (
                        "large" if abs(d) >= 0.8 else
                        "medium" if abs(d) >= 0.5 else
                        "small" if abs(d) >= 0.2 else
                        "negligible"
                    ),
                }

            test_results["pairwise"][pair_key] = pair_results

    # Bonferroni correction
    if all_p_values:
        corrections = bonferroni_correction(all_p_values, alpha)
        test_results["bonferroni"] = {
            label: correction
            for label, correction in zip(all_p_labels, corrections)
        }

    # Save
    suffix = "_text" if text_mode else ""
    output_path = results_dir / f"statistical_tests{suffix}.json"
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)

    return test_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run statistical tests")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--text", action="store_true", help="Use text-mode result files")
    args = parser.parse_args()

    results = run_all_tests(args.results, args.alpha, text_mode=args.text)

    # Summary
    for pair, tests in results["pairwise"].items():
        print(f"\n{pair}:")
        for metric, test_set in tests.items():
            t_test = test_set["paired_t_test"]
            d = test_set["cohens_d"]
            sig = "***" if t_test["significant"] else "n.s."
            print(f"  {metric}: t={t_test['t_statistic']:.3f}, p={t_test['p_value']:.4f} {sig}, d={d:.3f}")
