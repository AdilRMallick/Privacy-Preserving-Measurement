"""
Rényi Differential Privacy (RDP) accountant for the Sampled Gaussian Mechanism.

Privacy guarantee: after T rounds of DP-FedAvg with:
  - noise_multiplier sigma
  - client sampling rate q = clients_per_round / total_clients
  - failure probability delta

the mechanism satisfies (epsilon, delta)-DP where epsilon is returned by
RDPAccountant.epsilon after T calls to .accumulate().

References:
  Mironov (2017)  — "Rényi Differential Privacy"
  Mironov et al. (2019) — "RDP of the Sampled Gaussian Mechanism"
  Balle et al. (2020) — "Hypothesis Testing Interpretations and Rényi DP"
"""
from __future__ import annotations

import math
from typing import Dict, List

# RDP orders to evaluate; the tightest (epsilon, delta) conversion is the min over these
_ORDERS: List[int] = [2, 4, 8, 16, 32, 64, 128, 256]


class RDPAccountant:
    def __init__(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        delta: float = 1e-5,
    ):
        if not (0 < sampling_rate <= 1):
            raise ValueError("sampling_rate must be in (0, 1]")
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")

        self.sigma = noise_multiplier
        self.q = sampling_rate
        self.delta = delta
        self._rdp: Dict[int, float] = {a: 0.0 for a in _ORDERS}
        self.steps: int = 0

    def accumulate(self, num_steps: int = 1) -> float:
        """Account for num_steps FL rounds and return the current epsilon."""
        for alpha in _ORDERS:
            self._rdp[alpha] += num_steps * _rdp_sgm(self.q, self.sigma, alpha)
        self.steps += num_steps
        return self.epsilon

    @property
    def epsilon(self) -> float:
        """Current (epsilon, delta)-DP guarantee; optimised over RDP orders."""
        return float(
            min(_rdp_to_eps(a, self._rdp[a], self.delta) for a in _ORDERS)
        )


# ---------------------------------------------------------------------------
# Core RDP computation (integer orders, Poisson subsampling)
# ---------------------------------------------------------------------------

def _rdp_sgm(q: float, sigma: float, alpha: int) -> float:
    """
    RDP at integer order alpha >= 2 for one step of the Sampled Gaussian
    Mechanism with Poisson subsampling rate q and noise multiplier sigma.

    Formula (Theorem 3, Mironov 2019):
      (1/(alpha-1)) * log  sum_{j=0}^{alpha}
          C(alpha,j) * q^j * (1-q)^(alpha-j) * exp(j*(j-1) / (2*sigma^2))
    """
    log_sum = -math.inf
    for j in range(alpha + 1):
        log_term = (
            _log_comb(alpha, j)
            + j * math.log(q)
            + (alpha - j) * math.log(1.0 - q)
            + j * (j - 1) / (2.0 * sigma ** 2)
        )
        log_sum = _log_add(log_sum, log_term)
    return log_sum / (alpha - 1)


def _rdp_to_eps(alpha: int, rdp: float, delta: float) -> float:
    """Convert accumulated RDP to (eps, delta)-DP: eps = rdp + log(1/delta)/(alpha-1)."""
    return rdp + math.log(1.0 / delta) / (alpha - 1)


def _log_comb(n: int, k: int) -> float:
    """log C(n,k) via the log-gamma function (avoids integer overflow)."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))
