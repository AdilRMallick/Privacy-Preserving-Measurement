"""
Privacy-utility tradeoff analysis.

Sweeps over a list of noise_multiplier values and compares final model
accuracy against the (epsilon, delta)-DP budget consumed.  A no-DP
baseline is included for reference.
"""
from typing import Any, Dict, List, Tuple

from experiments.federated_mnist import run_federated_experiment


def run_tradeoff_analysis(
    fed_config: Dict[str, Any],
    dp_config: Dict[str, Any],
    tradeoff_config: Dict[str, Any],
) -> List[Tuple[str, float, float]]:
    """
    Run one experiment per noise setting (plus a no-DP baseline).

    Returns a list of (label, final_accuracy, final_epsilon) tuples.
    """
    noise_multipliers: List[float] = tradeoff_config["noise_multipliers"]
    num_rounds: int = tradeoff_config["num_rounds"]
    delta: float = dp_config["delta"]

    results: List[Tuple[str, float, float]] = []

    # --- Baseline: no DP ---
    print("\n  [Baseline] No differential privacy")
    r = run_federated_experiment(
        dict(fed_config, num_rounds=num_rounds),
        dict(dp_config, add_noise=False),
        verbose=False,
    )
    results.append(("no-DP", r["final_accuracy"], float("inf")))
    print(f"  → Accuracy: {r['final_accuracy']:.4f}")

    # --- DP sweep ---
    for sigma in noise_multipliers:
        print(f"\n  [σ = {sigma:.1f}]  noise_multiplier = {sigma}")
        r = run_federated_experiment(
            dict(fed_config, num_rounds=num_rounds),
            dict(dp_config, add_noise=True, noise_multiplier=sigma),
            verbose=False,
        )
        results.append((f"σ={sigma:.1f}", r["final_accuracy"], r["final_epsilon"]))
        print(f"  → Accuracy: {r['final_accuracy']:.4f} | ε = {r['final_epsilon']:.3f}")

    # --- Summary table ---
    _print_table(results, delta)
    return results


def _print_table(results: List[Tuple[str, float, float]], delta: float) -> None:
    bar = "─" * 52
    print(f"\n{bar}")
    print(f"  {'Setting':<12}  {'Accuracy':>10}  {'ε':>10}  (δ={delta:.0e})")
    print(bar)
    for label, acc, eps in results:
        eps_str = f"{eps:.3f}" if eps < 1e6 else "∞"
        print(f"  {label:<12}  {acc:>10.4f}  {eps_str:>10}")
    print(bar)
    _print_privacy_risk_notes(results)


def _print_privacy_risk_notes(results: List[Tuple[str, float, float]]) -> None:
    print("\n  Privacy Risk / Mitigation Notes")
    print("  --------------------------------")
    print("  ε < 1   : Strong privacy — membership inference extremely difficult.")
    print("  ε 1–10  : Moderate privacy — acceptable for many production deployments.")
    print("  ε > 10  : Weak formal guarantee — consider increasing noise_multiplier.")
    print("  no-DP   : No formal guarantee; establishes accuracy upper bound.")
    print()
    print("  Deployment considerations:")
    print("  • Use secure aggregation to prevent the server seeing raw client updates.")
    print("  • Combine with shuffling / local DP for stronger amplification.")
    print("  • Re-audit the privacy budget if the model is fine-tuned or retrained.")
