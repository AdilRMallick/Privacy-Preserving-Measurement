import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple


class FederatedServer:
    """
    Central coordinator implementing weighted FedAvg with optional server-side
    Gaussian noise for (epsilon, delta)-DP at the client level.

    Noise model: sensitivity of the weighted average w.r.t. one client's update
    is clip_norm / m  (m = number of participants).  Adding N(0, (sigma*C/m)^2)
    per-coordinate achieves noise_multiplier = sigma for the Gaussian mechanism.
    """

    def __init__(
        self,
        model_fn: Callable,
        dp_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model_fn()
        self._weights: List[np.ndarray] = self.model.get_weights()
        self.dp = dp_config or {}

    def aggregate(
        self,
        updates: List[List[np.ndarray]],
        counts: List[int],
    ) -> None:
        """Apply weighted FedAvg and inject DP noise into the aggregated update."""
        total = float(sum(counts))
        m = len(updates)

        agg = [np.zeros_like(w) for w in self._weights]
        for delta, count in zip(updates, counts):
            w = count / total
            for i, d in enumerate(delta):
                agg[i] += w * d

        if self.dp.get("add_noise"):
            sigma = self.dp["noise_multiplier"]
            C = self.dp["clip_norm"]
            # Noise std calibrated to L2 sensitivity of the weighted average
            noise_std = sigma * C / m
            agg = [a + np.random.normal(0.0, noise_std, a.shape) for a in agg]

        self._weights = [g + a for g, a in zip(self._weights, agg)]
        self.model.set_weights(self._weights)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        loss, acc = self.model.evaluate(x, y, verbose=0)
        return float(loss), float(acc)

    def get_weights(self) -> List[np.ndarray]:
        return [w.copy() for w in self._weights]
