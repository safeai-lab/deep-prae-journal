"""
Example 6: Intelligent Driving (IDM Car-Following).

Estimates crash probability for an IDM-controlled autonomous vehicle
following a human-driven lead vehicle with stochastic actions.

Paper specifications:
- State: [x_follow, x_lead, v_follow, v_lead, a_follow, a_lead]
- Input: 15 Gaussian random actions (4-second epochs over T=60s)
- IDM parameters: s_0=2m, v_0=30m/s, a=2*gamma, b=1.67, d=2*gamma,
                   T_bar=1.5s, delta=4, L=4m
- Crash condition: r_t = x_lead - x_follow - L < 0
- Architecture: 6 hidden layers (6, 16, 32, 16, 4, 2 nodes), ReLU
- n1=2,000 (uniform on [0,1]^15), n2=8,000
- Gamma: controls max acceleration/deceleration
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Dict, Optional

from ..core.algorithm import DeepPrAE
from ..configs.experiment_configs import EXAMPLE6_CONFIG


class Example6_IntelligentDriving:
    """Example 6: Autonomous vehicle crash probability estimation."""

    # IDM parameters from paper (Table)
    IDM_PARAMS = {
        's0': 2.0,       # Safety distance (m)
        'v0': 30.0,      # Free traffic speed (m/s)
        'b': 1.67,       # Comfortable deceleration (m/s^2)
        'T_bar': 1.5,    # Safe time headway (s)
        'delta_exp': 4,  # Acceleration exponent
        'L': 4.0,        # Car length (m)
    }

    def __init__(self, gamma: float = 1.0):
        """
        Initialize Example 6.

        Args:
            gamma: Braking capability parameter.
                   a_max = 2*gamma m/s^2 (max acceleration)
                   d_max = 2*gamma m/s^2 (max deceleration)
        """
        self.gamma = gamma
        self.config = EXAMPLE6_CONFIG
        self.dimension = self.config.dimension  # 15

        # Distribution parameters
        self.mu = self.config.mu  # ones(15) * 10 (u_0 = 10)
        self.sigma = self.config.sigma  # 1.0

        # IDM gamma-dependent parameters
        self.a_max = 2.0 * gamma  # Max acceleration
        self.d_max = 2.0 * gamma  # Max deceleration

        # Simulation parameters
        self.T = 60.0       # Simulation horizon (seconds)
        self.dt = 0.1       # Timestep (seconds)
        self.epoch_dt = 4.0 # Action epoch (seconds)
        self.n_epochs = int(self.T / self.epoch_dt)  # 15 epochs

        # Initial conditions
        self.x_follow_0 = 0.0
        self.x_lead_0 = 50.0   # Lead vehicle starts 50m ahead
        self.v_follow_0 = 20.0 # Initial follower speed (m/s)
        self.v_lead_0 = 20.0   # Initial leader speed (m/s)

    def idm_acceleration(self, v_follow, v_lead, gap):
        """
        Compute IDM acceleration for following vehicle.

        Args:
            v_follow: Follower speed (m/s)
            v_lead: Leader speed (m/s)
            gap: Bumper-to-bumper gap (m)

        Returns:
            IDM acceleration (m/s^2), clipped to [-d_max, a_max]
        """
        p = self.IDM_PARAMS
        delta_v = v_follow - v_lead

        # Desired gap
        s_star = p['s0'] + max(0.0, v_follow * p['T_bar'] +
                                v_follow * delta_v / (2 * np.sqrt(self.a_max * p['b'])))

        # IDM acceleration
        a_free = self.a_max * (1 - (v_follow / p['v0']) ** p['delta_exp'])
        a_int = -self.a_max * (s_star / max(gap, 0.1)) ** 2
        a_idm = a_free + a_int

        # Clip to physical limits
        return np.clip(a_idm, -self.d_max, self.a_max)

    def simulate_trajectory(self, lv_actions: np.ndarray):
        """
        Simulate car-following trajectory given lead vehicle actions.

        Args:
            lv_actions: Lead vehicle action parameters, shape [15,]
                       These are accelerations applied at each 4s epoch.

        Returns:
            Tuple of (crashed: bool, min_gap: float, trajectory_info: dict)
        """
        p = self.IDM_PARAMS
        dt = self.dt
        n_steps = int(self.T / dt)

        # Initialize state
        x_f = self.x_follow_0
        x_l = self.x_lead_0
        v_f = self.v_follow_0
        v_l = self.v_lead_0

        min_gap = x_l - x_f - p['L']

        for step in range(n_steps):
            t = step * dt
            epoch = min(int(t / self.epoch_dt), self.n_epochs - 1)

            # Lead vehicle acceleration from input (scaled)
            a_l = (lv_actions[epoch] - self.mu[0]) * 1.0  # Center at mu=10, scale=1.0

            # Gap (bumper to bumper)
            gap = x_l - x_f - p['L']

            # Check crash
            if gap < 0:
                return True, gap, {'crash_time': t}

            min_gap = min(min_gap, gap)

            # IDM acceleration for follower
            a_f = self.idm_acceleration(v_f, v_l, gap)

            # Update positions and velocities (Euler integration)
            x_f += v_f * dt + 0.5 * a_f * dt ** 2
            x_l += v_l * dt + 0.5 * a_l * dt ** 2

            v_f = max(0.0, v_f + a_f * dt)  # Speed >= 0
            v_l = max(0.0, v_l + a_l * dt)

        return False, min_gap, {}

    def indicator_function(self, x: np.ndarray) -> np.ndarray:
        """
        Crash indicator: simulate trajectory and check if crash occurs.

        Args:
            x: Lead vehicle action parameters, shape [N, 15] or [15,]

        Returns:
            Binary indicators, shape [N,]
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        N = x.shape[0]
        indicators = np.zeros(N)

        for i in range(N):
            crashed, _, _ = self.simulate_trajectory(x[i])
            indicators[i] = float(crashed)

        return indicators

    def original_pdf(self, x: np.ndarray) -> np.ndarray:
        """
        PDF of original distribution: N(mu, sigma^2 * I_15).

        Args:
            x: Input of shape [N, 15] or [15,]

        Returns:
            PDF values, shape [N,]
        """
        cov = (self.sigma ** 2) * np.eye(self.dimension)
        return multivariate_normal.pdf(x, mean=self.mu, cov=cov)

    def generate_stage1_samples(self, n1: int) -> tuple:
        """
        Generate Stage 1 samples: uniform on [0, 1]^15 scaled appropriately.

        Per paper: Stage 1 uses 2,000 uniform samples.

        Args:
            n1: Number of Stage 1 samples.

        Returns:
            Tuple of (X_stage1, labels)
        """
        # Uniform sampling scaled to cover the input space
        # Center around mu with range proportional to sigma
        low = self.mu - 3 * self.sigma
        high = self.mu + 3 * self.sigma
        X_stage1 = np.random.uniform(low, high, size=(n1, self.dimension))

        labels = self.indicator_function(X_stage1)
        return X_stage1, labels

    def run(
        self,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        verbose: bool = True,
        _test_mode: bool = False
    ) -> Dict:
        """
        Run Deep-PrAE for Example 6.

        Args:
            n1: Stage 1 samples (default: 2000)
            n2: Stage 2 samples (default: 8000)
            verbose: Print progress
            _test_mode: Return dummy results for testing

        Returns:
            Dictionary with results
        """
        n1 = n1 or self.config.n1
        n2 = n2 or self.config.n2

        if _test_mode:
            from ..utils.dummy_results import generate_dummy_results_example6
            return generate_dummy_results_example6(
                gamma=self.gamma, n1=n1, n2=n2
            )

        if verbose:
            print(f"\n{'='*70}")
            print(f"Example 6: Intelligent Driving Safety")
            print(f"{'='*70}")
            print(f"Gamma: {self.gamma} (a_max={self.a_max}, d_max={self.d_max})")
            print(f"Dimension: {self.dimension}")
            print(f"Simulation: T={self.T}s, epoch={self.epoch_dt}s, dt={self.dt}s")
            print(f"Stage 1 (n1): {n1}, Stage 2 (n2): {n2}")
            print(f"{'='*70}\n")

        # Generate Stage 1 samples
        if verbose:
            print("Generating Stage 1 samples (simulating trajectories)...")
        X_stage1, labels = self.generate_stage1_samples(n1)

        if verbose:
            num_crashes = int(labels.sum())
            print(f"Crashes in Stage 1: {num_crashes}/{n1} ({100*num_crashes/n1:.2f}%)")

        # Run Deep-PrAE
        deep_prae = DeepPrAE(
            indicator_function=self.indicator_function,
            original_pdf=self.original_pdf,
            dimension=self.dimension,
            mu=self.mu,
            sigma=self.sigma
        )

        results = deep_prae.run(
            X_stage1=X_stage1,
            Y_stage1=labels,
            n2=n2,
            hidden_dims=self.config.hidden_dims,
            n_iters=self.config.n_iters,
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            class_weights=self.config.class_weights,
            l2_reg=self.config.l2_reg,
            verbose=verbose
        )

        results['experiment'] = 'Example 6: Intelligent Driving'
        results['gamma'] = self.gamma
        results['n1'] = n1
        results['n2'] = n2

        return results
