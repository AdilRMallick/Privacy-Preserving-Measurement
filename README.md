# Privacy-Preserving Measurement & Federated Learning

An end-to-end prototype demonstrating **differentially private federated learning (DP-FedAvg)** on MNIST. Simulates private client updates, injects calibrated Gaussian noise, tracks the privacy budget with an RDP accountant, and evaluates privacy-utility tradeoffs — all without centralising raw user data.

---

## What It Does

| Component | Description |
|---|---|
| **Federated Learning** | 100 simulated clients each hold a local shard of MNIST. Each round, 10 are sampled, train locally, and send weight updates to the server. |
| **Differential Privacy** | Client updates are clipped to bound sensitivity, then the server injects Gaussian noise before aggregating. Achieves **(ε, δ)-DP at the user (client) level**. |
| **Privacy Accounting** | An RDP accountant (Mironov et al., 2019) tracks the privacy budget consumed across rounds and converts it to (ε, δ)-DP. |
| **Tradeoff Analysis** | Sweeps over noise multiplier values to compare final model accuracy against the privacy budget spent. |

---

## Project Structure

```
Privacy-Preserving-Measurement/
├── main.py                          # Entry point — runs both experiments
├── config.py                        # All hyperparameters
├── requirements.txt
│
├── federated/
│   ├── model.py                     # 784 → 128 → 64 → 10 MLP
│   ├── data_utils.py                # MNIST loading & client partitioning (IID / non-IID)
│   ├── client.py                    # Local SGD + L2 gradient clipping
│   └── server.py                    # FedAvg aggregation + server-side DP noise
│
├── differential_privacy/
│   ├── mechanisms.py                # Gaussian & Laplace noise mechanisms
│   └── accountant.py               # RDP accountant for Sampled Gaussian Mechanism
│
└── experiments/
    ├── federated_mnist.py           # Full FL training loop with per-round logging
    └── privacy_utility_tradeoff.py  # Noise multiplier sweep + summary table
```

---

## Setup

**Requirements:** Python 3.8+, pip

```bash
git clone https://github.com/AdilRMallick/Privacy-Preserving-Measurement.git
cd Privacy-Preserving-Measurement
pip install -r requirements.txt
```

---

## Running

### Full demo (both experiments)

```bash
python main.py
```

This runs two back-to-back experiments and prints results to stdout. Total runtime is roughly **5–15 minutes** on CPU depending on hardware.

---

### Experiment 1 — DP-FedAvg on MNIST

Trains a federated model for 50 rounds with noise multiplier σ = 1.1 and logs accuracy + privacy budget every 5 rounds:

```
Round  5/50 | Acc: 0.7821 | Loss: 0.7043 | ε =   1.234  (δ = 1e-05)
Round 10/50 | Acc: 0.8340 | Loss: 0.5612 | ε =   2.087  (δ = 1e-05)
...
Round 50/50 | Acc: 0.9102 | Loss: 0.3041 | ε =   4.871  (δ = 1e-05)

  Final test accuracy : 0.9102
  Privacy budget      : ε = 4.871  (δ = 1e-05)
```

---

### Experiment 2 — Privacy-Utility Tradeoff

Runs 20 rounds at each noise setting (including a no-DP baseline) and prints a comparison table:

```
────────────────────────────────────────────────────
  Setting        Accuracy           ε  (δ=1e-05)
────────────────────────────────────────────────────
  no-DP            0.9231            ∞
  σ=0.5            0.8934       18.432
  σ=1.0            0.9011        6.203
  σ=1.5            0.8876        2.714
  σ=2.0            0.8643        1.531
────────────────────────────────────────────────────
```

Higher σ → stronger privacy (lower ε) → slightly lower accuracy.

---

## Configuration

All hyperparameters live in `config.py`:

```python
FEDERATED = {
    "num_clients": 100,        # total simulated clients
    "clients_per_round": 10,   # sampled each round (10% participation)
    "num_rounds": 50,          # communication rounds
    "local_epochs": 3,         # SGD epochs per client per round
    "batch_size": 32,
    "learning_rate": 0.01,
    "clip_norm": 1.0,          # L2 sensitivity bound per client update
    "iid": True,               # False → non-IID label skew across clients
    "seed": 42,
}

DP = {
    "add_noise": True,
    "noise_multiplier": 1.1,   # σ — higher = more private, less accurate
    "delta": 1e-5,             # δ for (ε, δ)-DP
}

TRADEOFF = {
    "num_rounds": 20,
    "noise_multipliers": [0.5, 1.0, 1.5, 2.0],
}
```

To disable DP entirely (baseline), set `"add_noise": False`.  
To simulate label-skewed clients (non-IID), set `"iid": False`.

---

## How the Privacy Guarantee Works

```
Client update  →  clip to L2 norm ≤ C  →  server aggregates m updates
                                        →  add N(0, (σC/m)²) noise per weight
                                        →  RDP accountant tracks ε over T rounds
```

1. **Clipping** bounds the maximum influence of any single client (sensitivity = `clip_norm / m`).
2. **Gaussian noise** at the server satisfies the Gaussian mechanism with noise multiplier σ.
3. **Subsampling amplification**: only `q = clients_per_round / num_clients` of clients participate each round, which tightens the privacy guarantee.
4. **RDP composition**: privacy accumulates as `T × RDP_per_round`; the accountant optimises the conversion to (ε, δ) over a set of Rényi orders.

### Privacy risk guide

| ε range | Interpretation |
|---|---|
| < 1 | Strong — membership inference is extremely difficult |
| 1 – 10 | Moderate — acceptable for many production ML deployments |
| > 10 | Weak formal guarantee — increase `noise_multiplier` |
| ∞ (no DP) | No guarantee — use only as accuracy upper-bound baseline |

---

## Running Individual Modules

```python
# Just the federated experiment
from experiments.federated_mnist import run_federated_experiment
import config

result = run_federated_experiment(config.FEDERATED, config.DP)
print(result["final_accuracy"], result["final_epsilon"])
```

```python
# Just the Gaussian mechanism (standalone)
from differential_privacy.mechanisms import gaussian_mechanism
import numpy as np

noisy = gaussian_mechanism(np.array([1.0, 2.0, 3.0]), sensitivity=1.0, noise_multiplier=1.1)
```

```python
# Just the privacy accountant
from differential_privacy.accountant import RDPAccountant

acc = RDPAccountant(noise_multiplier=1.1, sampling_rate=0.1, delta=1e-5)
for _ in range(50):
    eps = acc.accumulate(1)
print(f"ε after 50 rounds: {eps:.3f}")
```

---

## References

- McMahan et al. (2017) — [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) *(FedAvg)*
- Abadi et al. (2016) — [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) *(DP-SGD)*
- Mironov (2017) — [Rényi Differential Privacy](https://arxiv.org/abs/1702.07476)
- Mironov et al. (2019) — [R&eacute;nyi Differential Privacy of the Sampled Gaussian Mechanism](https://arxiv.org/abs/1702.07476)
