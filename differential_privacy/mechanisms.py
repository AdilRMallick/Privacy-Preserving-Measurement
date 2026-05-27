"""
Standalone DP noise mechanisms (Gaussian and Laplace).

These are used independently of federated learning for single-query
privacy analysis and unit testing of the accounting logic.
"""
import numpy as np


def gaussian_mechanism(
    value: np.ndarray,
    sensitivity: float,
    noise_multiplier: float,
) -> np.ndarray:
    """
    Add calibrated Gaussian noise: std = noise_multiplier * sensitivity.
    Satisfies (eps, delta)-DP where sigma = sqrt(2*ln(1.25/delta)) / eps
    when noise_multiplier matches that formula.
    """
    std = noise_multiplier * sensitivity
    return value + np.random.normal(0.0, std, value.shape)


def laplace_mechanism(
    value: np.ndarray,
    sensitivity: float,
    epsilon: float,
) -> np.ndarray:
    """Add Laplace noise for pure epsilon-DP (scale = sensitivity/epsilon)."""
    return value + np.random.laplace(0.0, sensitivity / epsilon, value.shape)


def clip_to_norm(value: np.ndarray, max_norm: float) -> np.ndarray:
    """Project value onto the L2 ball of radius max_norm."""
    norm = float(np.linalg.norm(value))
    return value * min(1.0, max_norm / (norm + 1e-10))


def gaussian_sigma_for_target(
    sensitivity: float, epsilon: float, delta: float
) -> float:
    """Compute the Gaussian noise std achieving (eps, delta)-DP analytically."""
    return sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
