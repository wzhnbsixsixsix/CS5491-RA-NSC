#!/usr/bin/env python3
"""Check residual scoring identities and compare Monte Carlo draw counts."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from student_score import (
    DRAWS, GATE_PROBABILITY, SEED, SIGMA, ablation, branch_samples, capacities, score,
)


def svd_capacity(matrix):
    """Independent capacity reference using singular values."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return np.log1p(singular_values**2).sum()


def literal_gate_reference(d, f, draws, seed):
    """Use explicit Bernoulli masks and SVD for an independent toy reference."""
    rng = np.random.default_rng(seed)
    identity = np.eye(d)
    samples = np.empty((draws, 2))
    for i in range(draws):
        mask = rng.random(f) < GATE_PROBABILITY
        up = rng.normal(0, SIGMA, (f, d))
        down = rng.normal(0, SIGMA, (d, f))
        branch = (down * mask) @ up
        residual = 0.5 * (svd_capacity(identity + branch) + svd_capacity(identity - branch))
        samples[i] = residual - d * math.log(2), svd_capacity(branch)
    return samples


def check_identities():
    """Check zero/weak branches, exact capacities and architecture invariants."""
    identity = np.eye(2)
    assert capacities(np.zeros((2, 2)), identity) == (0.0, 0.0)
    rng = np.random.default_rng(SEED)
    branch = rng.normal(0, 0.3, (2, 2))
    residual, plain = capacities(branch, identity)
    expected = 0.5 * (svd_capacity(identity + branch) + svd_capacity(identity - branch))
    np.testing.assert_allclose(
        [residual, plain], [expected - 2 * math.log(2), svd_capacity(branch)], atol=1e-13,
        rtol=0,
    )

    # Equal C(B), different C(I+B), because the addition sees branch direction.
    t = 1e-3
    skew = t * np.array([[0, 1], [-1, 0]])
    symmetric_result = capacities(t * identity, identity)
    skew_result = capacities(skew, identity)
    np.testing.assert_allclose(
        [symmetric_result[0], skew_result[0]], [math.log1p(t**4 / 4), 2 * math.log1p(t*t / 2)],
        atol=1e-14, rtol=0,
    )
    np.testing.assert_allclose(symmetric_result[1], skew_result[1], atol=1e-14, rtol=0)

    forward = {'d_model': 8, 'n_layer': 2, 'n_head': [2, 4], 'd_inner': [12, 20]}
    reverse = {**forward, 'n_head': [4, 2], 'd_inner': [20, 12]}
    broadcast = {**forward, 'n_head': 2, 'd_inner': 12}
    expanded = {**forward, 'n_head': [2, 2], 'd_inner': [12, 12]}
    for scorer in (score, ablation):
        assert math.isfinite(scorer(forward))
        assert scorer(forward) == scorer(reverse)
        assert scorer(broadcast) == scorer(expanded)
    assert score(forward) != ablation(forward)
    return {'strength': t, 'symmetric': list(symmetric_result), 'skew': list(skew_result)}


def compare_draws(d, f, reference, reference_draws):
    """Record error against an independent reference, including sampling uncertainty."""
    start = time.perf_counter()
    estimate = branch_samples(d, f)
    estimate_seconds = time.perf_counter() - start
    start = time.perf_counter()
    high_draw = reference(d, f, reference_draws, SEED + 1)
    reference_seconds = time.perf_counter() - start
    estimates, targets = estimate.mean(axis=0), high_draw.mean(axis=0)
    combined_se = np.sqrt(
        estimate.var(axis=0, ddof=1) / DRAWS + high_draw.var(axis=0, ddof=1) / reference_draws
    )
    return {
        'hidden_width': d, 'ffn_width': f, 'reference_draws': reference_draws,
        'estimate_seconds': estimate_seconds, 'reference_seconds': reference_seconds,
        'scores': {
            name: {
                'estimate': float(estimates[i]), 'reference': float(targets[i]),
                'absolute_error': float(abs(estimates[i] - targets[i])),
                'relative_error': float(abs(estimates[i] - targets[i]) / abs(targets[i])),
                'combined_standard_error': float(combined_se[i]),
            }
            for i, name in enumerate(('residual', 'ablation'))
        },
    }


def main():
    """Run the checks and optionally archive the measured comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    weak_branch = check_identities()
    comparisons = [compare_draws(8, 16, literal_gate_reference, 4096)]
    for d, f in [(64, 512), (128, 1024), (256, 1536), (512, 2048)]:
        comparisons.append(compare_draws(d, f, branch_samples, 256))
    payload = {
        'checks': ['zero branch', 'Cholesky vs SVD', 'weak branch',
                   'layer reorder', 'scalar vs list configuration', 'ablation separation'],
        'draws': DRAWS, 'seed': SEED, 'reference_seed': SEED + 1,
        'weak_branch': weak_branch,
        'comparisons': comparisons,
    }
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print('PASS: ' + ', '.join(payload['checks']))
    for row in comparisons:
        errors = ', '.join(f"{name} {scores['relative_error']:.2%}"
                           for name, scores in row['scores'].items())
        print(f"d={row['hidden_width']:3} f={row['ffn_width']:4}: {errors}")


if __name__ == '__main__':
    main()
