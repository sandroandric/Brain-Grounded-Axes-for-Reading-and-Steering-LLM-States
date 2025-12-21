"""
Box 6 Scientific Validation

Rigorous statistical analysis of steering experiments:
1. Test if steering affects semantic shifts (paired t-tests)
2. Compute effect sizes (Cohen's d)
3. Test reading projection validity
4. Check for specificity (target vs off-target)
5. Report confidence intervals

Run:
python analysis/validate_box6.py --results outputs/steering_eval/steering_results.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import numpy as np
from scipy import stats


def load_results(path: pathlib.Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)


def bootstrap_ci(data: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    data = np.array(data)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    return np.percentile(means, alpha * 100), np.percentile(means, (1 - alpha) * 100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = args.results.parent

    results = load_results(args.results)

    print("=" * 70)
    print("BOX 6 SCIENTIFIC VALIDATION")
    print("=" * 70)

    validation_results = {
        "steering_tests": {},
        "reading_tests": {},
        "issues": [],
        "limitations": [],
    }

    # ===== PART 1: STEERING VALIDATION =====
    print("\n" + "=" * 70)
    print("1. STEERING EFFECT VALIDATION")
    print("=" * 70)

    for contrast_name, contrast_data in results["contrasts"].items():
        print(f"\n[{contrast_name.upper()}]")

        # Extract semantic shifts for negative vs positive steering
        neg_shifts = [g["semantic_shifts"][contrast_name]
                      for g in contrast_data["generations"]["strength_-1.5"]]
        pos_shifts = [g["semantic_shifts"][contrast_name]
                      for g in contrast_data["generations"]["strength_1.5"]]
        baseline_shifts = [g["semantic_shifts"][contrast_name]
                           for g in contrast_data["generations"]["strength_0.0"]]

        # Paired t-test: Does steering change the target dimension?
        t_stat, p_value = stats.ttest_ind(pos_shifts, neg_shifts)
        d = cohens_d(pos_shifts, neg_shifts)

        # Bootstrap CIs
        neg_ci = bootstrap_ci(neg_shifts)
        pos_ci = bootstrap_ci(pos_shifts)
        delta = np.mean(pos_shifts) - np.mean(neg_shifts)
        delta_ci = (pos_ci[0] - neg_ci[1], pos_ci[1] - neg_ci[0])

        print(f"  Mean shift (neg): {np.mean(neg_shifts):+.3f} 95% CI [{neg_ci[0]:.3f}, {neg_ci[1]:.3f}]")
        print(f"  Mean shift (pos): {np.mean(pos_shifts):+.3f} 95% CI [{pos_ci[0]:.3f}, {pos_ci[1]:.3f}]")
        print(f"  Δ (pos - neg):    {delta:+.3f}")
        print(f"  t-statistic:      {t_stat:.3f}")
        print(f"  p-value:          {p_value:.4f}")
        print(f"  Cohen's d:        {d:.3f}")

        sig = "✓ SIGNIFICANT" if p_value < 0.05 else "✗ NOT SIGNIFICANT"
        print(f"  Result:           {sig}")

        # Check off-target effects (specificity)
        print(f"\n  Off-target effects:")
        off_target_effects = {}
        for other_contrast in results["contrasts"].keys():
            if other_contrast != contrast_name:
                other_neg = [g["semantic_shifts"][other_contrast]
                             for g in contrast_data["generations"]["strength_-1.5"]]
                other_pos = [g["semantic_shifts"][other_contrast]
                             for g in contrast_data["generations"]["strength_1.5"]]
                other_delta = np.mean(other_pos) - np.mean(other_neg)
                other_t, other_p = stats.ttest_ind(other_pos, other_neg)
                off_target_effects[other_contrast] = {
                    "delta": float(other_delta),
                    "t": float(other_t),
                    "p": float(other_p),
                }
                off_sig = "⚠ AFFECTED" if other_p < 0.05 else "OK"
                print(f"    {other_contrast}: Δ={other_delta:+.3f}, p={other_p:.4f} {off_sig}")

        # Perplexity analysis
        neg_ppl = [g["perplexity"] for g in contrast_data["generations"]["strength_-1.5"] if g["perplexity"]]
        pos_ppl = [g["perplexity"] for g in contrast_data["generations"]["strength_1.5"] if g["perplexity"]]
        base_ppl = [g["perplexity"] for g in contrast_data["generations"]["strength_0.0"] if g["perplexity"]]

        ppl_t, ppl_p = stats.ttest_ind(pos_ppl + neg_ppl, base_ppl)
        print(f"\n  Perplexity impact:")
        print(f"    Baseline:       {np.mean(base_ppl):.3f}")
        print(f"    Steered (avg):  {np.mean(pos_ppl + neg_ppl):.3f}")
        print(f"    p-value:        {ppl_p:.4f}")

        validation_results["steering_tests"][contrast_name] = {
            "neg_mean": float(np.mean(neg_shifts)),
            "neg_ci": neg_ci,
            "pos_mean": float(np.mean(pos_shifts)),
            "pos_ci": pos_ci,
            "delta": float(delta),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(d),
            "significant": p_value < 0.05,
            "off_target": off_target_effects,
            "ppl_baseline": float(np.mean(base_ppl)),
            "ppl_steered": float(np.mean(pos_ppl + neg_ppl)),
            "ppl_p": float(ppl_p),
        }

    # ===== PART 2: READING VALIDATION =====
    print("\n" + "=" * 70)
    print("2. READING (PROJECTION) VALIDATION")
    print("=" * 70)

    reading = results["reading"]
    texts = reading["texts"]
    projections = reading["projections"]

    # Expected directions based on text content
    expected = [
        {"concreteness": +1, "arousal": 0, "valence": 0},  # concrete bridge
        {"concreteness": -1, "arousal": 0, "valence": 0},  # abstract concepts
        {"concreteness": 0, "arousal": +1, "valence": 0},  # exciting, thrilling
        {"concreteness": 0, "arousal": -1, "valence": +1},  # peaceful, calm
        {"concreteness": 0, "arousal": 0, "valence": +1},  # happy, wonderful
        {"concreteness": 0, "arousal": 0, "valence": -1},  # angry, terrible
    ]

    print("\nValidation of semantic projections:")
    print(f"{'Text':<50} {'Dim':<12} {'Expected':<10} {'Actual':<10} {'Match':<6}")
    print("-" * 88)

    correct = 0
    total = 0
    for i, (text, proj, exp) in enumerate(zip(texts, projections, expected)):
        for dim, exp_dir in exp.items():
            if exp_dir != 0:
                actual_val = proj[dim]
                match = (exp_dir > 0 and actual_val > 0) or (exp_dir < 0 and actual_val < 0)
                correct += 1 if match else 0
                total += 1
                exp_str = "+" if exp_dir > 0 else "-"
                act_str = f"{actual_val:+.2f}"
                match_str = "✓" if match else "✗"
                print(f"{text[:48]:<50} {dim:<12} {exp_str:<10} {act_str:<10} {match_str:<6}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nReading accuracy: {correct}/{total} = {accuracy:.1%}")

    # Binomial test: Is accuracy better than chance (50%)?
    binom_result = stats.binomtest(correct, total, 0.5, alternative='greater')
    binom_p = binom_result.pvalue
    print(f"Binomial test (vs 50% chance): p = {binom_p:.4f}")

    validation_results["reading_tests"] = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "binom_p": float(binom_p),
        "significant": binom_p < 0.05,
    }

    # ===== PART 3: IDENTIFY ISSUES =====
    print("\n" + "=" * 70)
    print("3. IDENTIFIED ISSUES")
    print("=" * 70)

    issues = []

    # Small sample size
    n_prompts = len(results["contrasts"]["concreteness"]["prompts"])
    if n_prompts < 10:
        issues.append(f"Small sample size: Only {n_prompts} prompts tested")
        print(f"⚠ Small sample size: Only {n_prompts} prompts tested")

    # Check for non-significant contrasts
    for contrast, test in validation_results["steering_tests"].items():
        if not test["significant"]:
            issues.append(f"Steering not significant for {contrast} (p={test['p_value']:.4f})")
            print(f"⚠ Steering not significant for {contrast} (p={test['p_value']:.4f})")

    # Check for off-target effects
    for contrast, test in validation_results["steering_tests"].items():
        for other, effect in test["off_target"].items():
            if effect["p"] < 0.05:
                issues.append(f"{contrast} steering affects {other} (p={effect['p']:.4f})")
                print(f"⚠ {contrast} steering affects {other} off-target (p={effect['p']:.4f})")

    # Check perplexity degradation
    for contrast, test in validation_results["steering_tests"].items():
        ppl_increase = (test["ppl_steered"] - test["ppl_baseline"]) / test["ppl_baseline"]
        if ppl_increase > 0.20:
            issues.append(f"{contrast} steering increases perplexity by {ppl_increase:.0%}")
            print(f"⚠ {contrast} steering increases perplexity by {ppl_increase:.0%}")

    if not issues:
        print("✓ No major issues identified")

    validation_results["issues"] = issues

    # ===== PART 4: LIMITATIONS =====
    print("\n" + "=" * 70)
    print("4. LIMITATIONS")
    print("=" * 70)

    limitations = [
        f"Small sample: Only {n_prompts} prompts, limiting statistical power",
        "Steering vectors from English words, not brain data directly",
        "Single model tested (TinyLlama 1.1B)",
        "Single layer steering (layer 11 of 22)",
        "No human evaluation of output quality",
        "Steering strength range limited to [-1.5, +1.5]",
        "No control for prompt-specific effects",
        "Perplexity as only fluency metric",
    ]

    for lim in limitations:
        print(f"  • {lim}")

    validation_results["limitations"] = limitations

    # ===== PART 5: SUMMARY =====
    print("\n" + "=" * 70)
    print("5. SUMMARY")
    print("=" * 70)

    # Count validations
    n_steering_sig = sum(1 for t in validation_results["steering_tests"].values() if t["significant"])
    n_contrasts = len(validation_results["steering_tests"])

    print(f"\nSteering effects: {n_steering_sig}/{n_contrasts} contrasts significant (p<0.05)")
    print(f"Reading accuracy: {accuracy:.1%} (p={binom_p:.4f})")
    print(f"Issues identified: {len(issues)}")
    print(f"Limitations: {len(limitations)}")

    # Effect size summary
    print("\nEffect sizes (Cohen's d):")
    for contrast, test in validation_results["steering_tests"].items():
        d = test["cohens_d"]
        size = "LARGE" if abs(d) > 0.8 else ("MEDIUM" if abs(d) > 0.5 else "SMALL")
        print(f"  {contrast}: d = {d:.3f} ({size})")

    validation_results["summary"] = {
        "n_steering_significant": n_steering_sig,
        "n_contrasts": n_contrasts,
        "reading_accuracy": accuracy,
        "reading_p": float(binom_p),
        "n_issues": len(issues),
        "n_limitations": len(limitations),
    }

    # Save validation results (convert numpy types)
    def convert_np(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_np(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_np(v) for v in obj]
        return obj

    out_path = args.out_dir / "box6_validation.json"
    with open(out_path, "w") as f:
        json.dump(convert_np(validation_results), f, indent=2)

    print(f"\n✓ Validation saved to {out_path}")


if __name__ == "__main__":
    main()
