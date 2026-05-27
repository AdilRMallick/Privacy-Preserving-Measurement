"""
Privacy-Preserving Measurement & Federated Learning Prototype
=============================================================

Pipeline
--------
1. Partition MNIST across 100 simulated clients (IID split).
2. Run DP-FedAvg: each round, 10 clients train locally, clip their weight
   updates to bound sensitivity, and the server injects calibrated Gaussian
   noise before aggregating — preserving (epsilon, delta)-DP at the user level.
3. Track the privacy budget with an RDP accountant (Mironov et al., 2019).
4. Sweep noise_multiplier values to profile the privacy-utility tradeoff.

Run
---
    pip install -r requirements.txt
    python main.py
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF info/warning logs

import config
from experiments.federated_mnist import run_federated_experiment
from experiments.privacy_utility_tradeoff import run_tradeoff_analysis


def main() -> None:
    _section(
        "1 / 2  DP-FedAvg on MNIST",
        f"σ = {config.DP['noise_multiplier']}  |  "
        f"{config.FEDERATED['num_rounds']} rounds  |  "
        f"{config.FEDERATED['num_clients']} clients  |  "
        f"{config.FEDERATED['clients_per_round']} sampled/round",
    )

    result = run_federated_experiment(config.FEDERATED, config.DP, verbose=True)

    print(f"\n  Final test accuracy : {result['final_accuracy']:.4f}")
    print(
        f"  Privacy budget     : ε = {result['final_epsilon']:.3f}  "
        f"(δ = {config.DP['delta']:.0e})"
    )

    _section(
        "2 / 2  Privacy-Utility Tradeoff",
        f"{config.TRADEOFF['num_rounds']} rounds per setting  |  "
        f"σ ∈ {config.TRADEOFF['noise_multipliers']}",
    )

    run_tradeoff_analysis(config.FEDERATED, config.DP, config.TRADEOFF)


def _section(title: str, subtitle: str = "") -> None:
    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(bar)


if __name__ == "__main__":
    main()
