import numpy as np
from typing import Any, Callable, Dict, List, Tuple


class FederatedClient:
    """
    A single federated participant that trains a local model and returns
    a sensitivity-bounded weight delta (client-level DP clipping).
    """

    def __init__(
        self,
        client_id: int,
        data: Tuple[np.ndarray, np.ndarray],
        model_fn: Callable,
        config: Dict[str, Any],
    ):
        self.client_id = client_id
        self.x, self.y = data
        self.model = model_fn()
        self.config = config

    def compute_update(
        self, global_weights: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], int]:
        """
        Synchronise with the global model, run local SGD, and return the
        clipped weight delta together with the local sample count.
        """
        self.model.set_weights(global_weights)
        init_weights = [w.copy() for w in global_weights]

        self.model.fit(
            self.x,
            self.y,
            epochs=self.config["local_epochs"],
            batch_size=self.config["batch_size"],
            verbose=0,
        )

        delta = [nw - iw for nw, iw in zip(self.model.get_weights(), init_weights)]
        delta = _clip_by_global_norm(delta, self.config["clip_norm"])
        return delta, len(self.x)

    @property
    def num_samples(self) -> int:
        return len(self.x)


def _clip_by_global_norm(tensors: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
    """Clip the concatenated update vector so its L2 norm <= max_norm."""
    global_norm = float(np.sqrt(sum(np.sum(t ** 2) for t in tensors)))
    scale = min(1.0, max_norm / (global_norm + 1e-10))
    return [t * scale for t in tensors]
