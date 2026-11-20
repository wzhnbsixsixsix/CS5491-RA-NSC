"""Specification-only FFN residual capacity and its shortcut ablation."""
from functools import lru_cache
import math

import numpy as np

from nsc import attention_capacity, layers

SIGMA = 0.02
GATE_PROBABILITY = 0.5
DRAWS = 32
SEED = 202627


def logdet_spd(matrix):
    """Evaluate a positive-definite log determinant via its Cholesky diagonal."""
    diagonal = np.linalg.cholesky(matrix).diagonal()
    return 2 * np.log(diagonal).sum()


def capacities(branch, identity):
    """Return centered residual and plain capacity using the same branch draw.

    Average the residual capacities for B and -B to cancel odd terms.
    Each sign has the same distribution. Dividing the residual Gram update
    by two accounts for the pure shortcut's d * log(2) capacity.
    """
    gram = branch.T @ branch
    even = identity + 0.5 * gram
    odd = 0.5 * (branch + branch.T)
    residual = 0.5 * (logdet_spd(even + odd) + logdet_spd(even - odd))
    return residual, logdet_spd(identity + gram)


def branch_samples(d, f, draws=DRAWS, seed=SEED):
    """Sample paired capacities given hidden width d and FFN width f."""
    rng = np.random.default_rng([seed, d, f])
    identity = np.eye(d)
    samples = np.empty((draws, 2))
    for i in range(draws):
        # Exchangeable gates let us sample just the active count and its weights.
        active = rng.binomial(f, GATE_PROBABILITY)
        down = rng.standard_normal((d, active))
        up = rng.standard_normal((active, d))
        branch = (down @ up) * SIGMA**2
        samples[i] = capacities(branch, identity)
    return samples


@lru_cache(maxsize=None)
def ffn_capacity(d, f):
    """Cache both mean capacities for each architecture-independent FFN shape."""
    residual, plain = branch_samples(d, f).mean(axis=0)
    return float(residual), float(plain)


def architecture_score(config, component):
    """Keep the reference attention term and sum the chosen FFN contribution."""
    d, heads, inner = layers(config)
    return math.fsum(
        attention_capacity(d, h) + ffn_capacity(d, f)[component]
        for h, f in zip(heads, inner)
    )


def score(config):
    """Return the score with the centered residual FFN contribution."""
    return architecture_score(config, 0)


def ablation(config):
    """Score the same FFN branch and attention with the shortcut removed."""
    return architecture_score(config, 1)
