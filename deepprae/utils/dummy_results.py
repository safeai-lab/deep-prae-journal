"""
Dummy/test results generator for quick testing without running full algorithm.

Produces results with paper-accurate order-of-magnitude values for each experiment.
Useful for testing the pipeline, figure generation, and understanding output format.

Paper reference values:
  Example 1 (gamma=1.8): P ~ 4.1e-24, RE ~ 0.40, 2 dominating points
  Example 2 (gamma=4.75, h=10): P_true ~ 1.18e-8, UB ~ 5e-8, RE ~ 0.10, 49 DPs
  Example 3 (gamma=11): P ~ 1e-4, UB ~ 2.5e-4, RE ~ 0.05, ~20 DPs
  Example 4 (gamma=20): P ~ 1.23e-5, UB ~ 1.77e-5, RE ~ 0.10, ~32 DPs
  Example 5 (gamma=1.0): P ~ 1e-6, UB ~ 1.46e-6, RE ~ 0.15, ~29 DPs
  Example 6 (gamma=1.0): P ~ 1e-3, UB ~ 5e-3, RE ~ 0.30, ~42 DPs
"""

import numpy as np
from typing import Dict


def _generate_realistic_dominating_points(
    dimension: int,
    n_points: int,
    mu: np.ndarray,
    sigma: float,
    offset_scale: float = 2.0
) -> np.ndarray:
    """Generate realistic-looking dominating points offset from mean."""
    offsets = np.random.randn(n_points, dimension) * offset_scale * sigma
    return mu.reshape(1, -1) + offsets


def _training_history(n_iters: int = 500):
    """Generate plausible training loss/accuracy curves."""
    t = np.linspace(0, 5, n_iters)
    return {
        'losses': list(np.exp(-t) * 0.7 + 0.01),
        'accuracies': list(0.5 + 0.48 * (1 - np.exp(-t))),
    }


# ── Example 1: 2D Sigmoid ──────────────────────────────────────────────

def generate_dummy_results_example1(
    gamma: float = 1.8,
    n1: int = 10000,
    n2: int = 20000
) -> Dict:
    """
    Dummy results for Example 1: 2D Sigmoid.

    Paper (Table 2, gamma=1.8): P(UB) ~ 4.09e-24, RE ~ 0.40, 2 DPs.
    Probability decays super-exponentially with gamma.
    """
    # Paper-calibrated probability decay (quadratic fit to Table 2)
    # gamma: 1.0→2.53e-3, 1.2→3.16e-6, 1.4→3.86e-10, 1.6→2.96e-16, 1.8→4.09e-24
    g = gamma - 1.0
    log10_p = -2.6 - 8.0 * g - 22.5 * g ** 2
    prob = 10 ** log10_p * np.random.uniform(0.85, 1.15)
    n_dp = max(2, int(2 + 0.5 * g + np.random.randint(-1, 2)))
    re = 0.30 + 0.12 * (gamma - 1.0) + np.random.uniform(-0.03, 0.03)

    mu = np.array([5.0, 5.0])
    sigma = 0.5

    return {
        'experiment': 'Example 1: 2D Sigmoid',
        'gamma': gamma,
        'probability': prob,
        'relative_error': float(np.clip(re, 0.05, 1.0)),
        'n_dominating_points': n_dp,
        'num_dominating_points': n_dp,
        'dominating_points': _generate_realistic_dominating_points(2, n_dp, mu, sigma, 3.0),
        'n1': n1,
        'n2': n2,
        'total_samples': n1 + n2,
        'stage1_time': np.random.uniform(15, 30),
        'stage2_time': np.random.uniform(5, 10),
        'total_time': np.random.uniform(20, 40),
        'training_history': _training_history(500),
        '_dummy': True,
        '_note': 'Dummy output calibrated to paper Table 2. Run with real algorithm for actual results.',
    }


# ── Example 2: 5D Ball Complement ──────────────────────────────────────

def generate_dummy_results_example2(
    gamma: float = 4.75,
    hidden_dim: int = 10,
    n1: int = 2000,
    n2: int = 8000
) -> Dict:
    """
    Dummy results for Example 2: Complement of 5D Ball.

    Paper (Table 3, gamma=4.75, h=10): true P ~ 1.18e-8, UB ~ 5.11e-8,
    RE ~ 0.10, 49 DPs.
    """
    from scipy.stats import chi2

    sigma2 = 0.5
    true_prob = 1 - chi2.cdf((gamma ** 2) / sigma2, df=5)

    # Upper bound is typically 2-5x the true probability for this example
    ub_ratio = 3.0 + 1.5 * (hidden_dim - 10) / 10.0 + np.random.uniform(-0.5, 0.5)
    ub_prob = true_prob * max(1.5, ub_ratio)

    # Number of DPs depends on hidden dim: h=10→49, h=15→87, h=20→130
    n_dp = int(30 + 2.0 * hidden_dim + np.random.randint(-3, 4))
    re = 0.08 + 0.02 * (gamma - 4.0) + np.random.uniform(-0.02, 0.02)

    mu = np.zeros(5)
    sigma = np.sqrt(sigma2)

    return {
        'experiment': 'Example 2: 5D Ball Complement',
        'gamma': gamma,
        'hidden_dim': hidden_dim,
        'probability': ub_prob,
        'true_probability': true_prob,
        'relative_error': float(np.clip(re, 0.02, 0.5)),
        'n_dominating_points': n_dp,
        'num_dominating_points': n_dp,
        'dominating_points': _generate_realistic_dominating_points(5, n_dp, mu, sigma, 3.0),
        'n1': n1,
        'n2': n2,
        'total_samples': n1 + n2,
        'stage1_time': np.random.uniform(10, 20),
        'stage2_time': np.random.uniform(3, 8),
        'total_time': np.random.uniform(13, 28),
        'training_history': _training_history(2000),
        '_dummy': True,
        '_note': 'Dummy output calibrated to paper Table 3.',
    }


# ── Example 3: Random Walk Excursion ───────────────────────────────────

def generate_dummy_results_example3(
    T: int = 10,
    gamma: float = 11.0,
    n1: int = 10000,
    n2: int = 20000
) -> Dict:
    """
    Dummy results for Example 3: Random Walk.

    Paper (Table 4, gamma=11, n1=10k): UB ~ 2.53e-4, RE ~ 0.0477, ~20 DPs.
    NMC reference: ~1e-4.
    """
    # Paper-calibrated: UB ~ 2.5e-4 for gamma=11
    base_ub = 2.5e-4
    scale = np.exp(-0.5 * (gamma - 11.0))           # rough scaling around gamma=11
    prob = base_ub * scale * np.random.uniform(0.9, 1.1)

    n_dp = int(18 + np.random.randint(-3, 5))
    re = 0.045 + np.random.uniform(-0.01, 0.01)

    mu = np.zeros(T)
    sigma = 1.0

    return {
        'experiment': 'Example 3: Random Walk Excursion',
        'T': T,
        'gamma': gamma,
        'probability': prob,
        'relative_error': float(np.clip(re, 0.01, 0.3)),
        'n_dominating_points': n_dp,
        'num_dominating_points': n_dp,
        'dominating_points': _generate_realistic_dominating_points(T, n_dp, mu, sigma, 2.0),
        'n1': n1,
        'n2': n2,
        'total_samples': n1 + n2,
        'stage1_time': np.random.uniform(20, 35),
        'stage2_time': np.random.uniform(8, 15),
        'total_time': np.random.uniform(28, 50),
        'training_history': _training_history(1000),
        '_dummy': True,
        '_note': 'Dummy output calibrated to paper Table 4.',
    }


# ── Example 4: Non-Gaussian ────────────────────────────────────────────

def generate_dummy_results_example4(
    gamma: float = 20.0,
    n1: int = 5000,
    n2: int = 495000
) -> Dict:
    """
    Dummy results for Example 4: Non-Gaussian (Expo + GenPareto).

    Paper (Section 5.4, gamma_r=20): NMC ~ 1.23e-5, UB ~ 1.77e-5,
    RE ~ 0.10, ~32 DPs.
    """
    # Paper-calibrated
    base_ub = 1.77e-5
    scale = np.exp(-0.3 * (gamma - 20.0))
    prob = base_ub * scale * np.random.uniform(0.9, 1.1)

    n_dp = int(30 + np.random.randint(-3, 5))
    re = 0.10 + np.random.uniform(-0.02, 0.02)

    mu = np.zeros(6)
    sigma = 1.0

    return {
        'experiment': 'Example 4: Non-Gaussian Distribution',
        'gamma': gamma,
        'probability': prob,
        'nmc_reference': 1.23e-5,
        'relative_error': float(np.clip(re, 0.03, 0.5)),
        'n_dominating_points': n_dp,
        'num_dominating_points': n_dp,
        'dominating_points': _generate_realistic_dominating_points(6, n_dp, mu, sigma, 2.5),
        'n1': n1,
        'n2': n2,
        'total_samples': n1 + n2,
        'stage1_time': np.random.uniform(25, 40),
        'stage2_time': np.random.uniform(150, 200),
        'total_time': np.random.uniform(175, 240),
        'training_history': _training_history(1000),
        '_dummy': True,
        '_note': 'Dummy output calibrated to paper Section 5.4.',
    }


# ── Example 5: Hollow Set with Hole ────────────────────────────────────

def generate_dummy_results_example5(
    gamma: float = 1.0,
    n1: int = 2000,
    n2: int = 8000
) -> Dict:
    """
    Dummy results for Example 5: Rare-event Set with Hole.

    Paper (Section 5.5): UB ~ 1.46e-6, RE ~ 0.15, ~29 DPs.
    """
    base_ub = 1.46e-6
    scale = np.exp(-2.0 * (gamma - 1.0))
    prob = base_ub * scale * np.random.uniform(0.9, 1.1)

    n_dp = int(27 + np.random.randint(-3, 5))
    re = 0.15 + np.random.uniform(-0.03, 0.03)

    mu = np.zeros(2)
    sigma = 0.5

    return {
        'experiment': 'Example 5: Rare-event Set with Hole',
        'gamma': gamma,
        'probability': prob,
        'relative_error': float(np.clip(re, 0.05, 0.6)),
        'n_dominating_points': n_dp,
        'num_dominating_points': n_dp,
        'dominating_points': _generate_realistic_dominating_points(2, n_dp, mu, sigma, 4.0),
        'hole_radius': 1.0,
        'hole_center': [1.5, 5.0],
        'n1': n1,
        'n2': n2,
        'total_samples': n1 + n2,
        'stage1_time': np.random.uniform(12, 25),
        'stage2_time': np.random.uniform(5, 10),
        'total_time': np.random.uniform(17, 35),
        'training_history': _training_history(500),
        '_dummy': True,
        '_note': 'Dummy output calibrated to paper Section 5.5.',
    }


# ── Example 6: Intelligent Driving ─────────────────────────────────────

def generate_dummy_results_example6(
    gamma: float = 1.0,
    n1: int = 2000,
    n2: int = 8000
) -> Dict:
    """
    Dummy results for Example 6: Intelligent Driving (IDM).

    Paper (Section 5.6, gamma=1.0): UB ~ 5e-3, RE ~ 0.30, ~42 DPs.
    """
    # Crash probability decreases with higher braking capability gamma
    base_ub = 5e-3
    scale = np.exp(-1.5 * (gamma - 1.0))
    prob = base_ub * scale * np.random.uniform(0.85, 1.15)

    n_dp = int(40 + np.random.randint(-5, 6))
    re = 0.30 + np.random.uniform(-0.05, 0.05)

    mu = np.ones(15) * 10.0
    sigma = 1.0

    return {
        'experiment': 'Example 6: Intelligent Driving Safety',
        'gamma': gamma,
        'a_max': 2.0 * gamma,
        'd_max': 2.0 * gamma,
        'probability': prob,
        'relative_error': float(np.clip(re, 0.1, 0.8)),
        'n_dominating_points': n_dp,
        'num_dominating_points': n_dp,
        'dominating_points': _generate_realistic_dominating_points(15, n_dp, mu, sigma, 1.5),
        'n1': n1,
        'n2': n2,
        'total_samples': n1 + n2,
        'stage1_time': np.random.uniform(30, 50),
        'stage2_time': np.random.uniform(10, 20),
        'total_time': np.random.uniform(40, 70),
        'training_history': _training_history(1000),
        '_dummy': True,
        '_note': 'Dummy output calibrated to paper Section 5.6.',
    }


# ── Registry ────────────────────────────────────────────────────────────

DUMMY_GENERATORS = {
    1: generate_dummy_results_example1,
    2: generate_dummy_results_example2,
    3: generate_dummy_results_example3,
    4: generate_dummy_results_example4,
    5: generate_dummy_results_example5,
    6: generate_dummy_results_example6,
}


def get_dummy_results(example_num: int, **kwargs) -> Dict:
    """
    Get dummy results for any example.

    Args:
        example_num: Example number (1-6)
        **kwargs: Example-specific parameters (gamma, n1, n2, etc.)

    Returns:
        Dictionary with dummy results

    Raises:
        ValueError: If example_num is invalid
    """
    if example_num not in DUMMY_GENERATORS:
        raise ValueError(f"Invalid example number: {example_num}. Must be 1-6.")
    return DUMMY_GENERATORS[example_num](**kwargs)
