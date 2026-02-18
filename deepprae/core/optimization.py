"""
Dominating point solver using mixed-integer nonlinear programming.

This module implements the optimization problem to find dominating points
that characterize the rare-event set for importance sampling.

Supports arbitrary-depth ReLU neural networks by encoding each hidden layer
as mixed-integer linear constraints using the big-M formulation.
"""

import numpy as np
import pyomo.environ as pyomo
from pyomo.opt import SolverFactory
from typing import List, Dict, Optional


class DominatingPointSolver:
    """
    Solver for finding dominating points via MINLP.

    Formulates and solves the optimization problem to find points that
    dominate the outer approximation of the rare-event set, using the
    iterative cutting plane algorithm.

    Supports multi-layer ReLU networks: each hidden layer is encoded via
    big-M constraints with binary variables for ReLU activation states.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        mu: Optional[np.ndarray] = None,
        sigma: float = 1.0,
        solver_name: str = 'gurobi',
        max_iterations: int = 100,
        eps_ignore: float = 0.1,
        big_M: float = 10000.0
    ):
        """
        Initialize dominating point solver.

        Args:
            input_dim: Dimension of input space
            hidden_dims: List of hidden layer dimensions
            mu: Mean of original distribution (defaults to zeros)
            sigma: Standard deviation of original distribution
            solver_name: Name of solver to use ('gurobi', 'ipopt', etc.)
            max_iterations: Maximum number of dominating points to find
            eps_ignore: Small value for cutting plane constraint
            big_M: Big-M constant for ReLU encoding
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 2  # Binary classification

        self.mu = mu if mu is not None else np.zeros(input_dim)
        self.sigma = sigma

        self.solver_name = solver_name
        self.max_iterations = max_iterations
        self.eps_ignore = eps_ignore
        self.big_M = big_M

        self.dominating_points = None
        self.model = None

    def build_model(self, network_params: Dict) -> pyomo.ConcreteModel:
        """
        Build Pyomo optimization model encoding a multi-layer ReLU network.

        The network_params dict is keyed by layer index (as strings):
          '0', '1', ..., 'L' where L = num_hidden_layers.
        Layers 0..L-1 are hidden layers (with ReLU), layer L is the output layer.

        Args:
            network_params: Dictionary of network weights and biases from
                           NeuralNetworkClassifier.extract_params()

        Returns:
            Pyomo ConcreteModel
        """
        model = pyomo.ConcreteModel()

        num_layers = len(network_params)
        num_hidden = num_layers - 1  # Last layer is output (no ReLU)

        # --- Input variables ---
        model.input_indices = pyomo.Set(initialize=list(range(self.input_dim)))
        model.x = pyomo.Var(model.input_indices, domain=pyomo.Reals)

        # --- Distribution parameters ---
        model.mu = pyomo.Param(
            model.input_indices,
            initialize={i: float(self.mu[i]) for i in range(self.input_dim)}
        )
        model.sigma = pyomo.Param(initialize=self.sigma)
        model.bigM = pyomo.Param(initialize=self.big_M)

        # Store network params for reference
        self.network_params = network_params

        # --- Hidden layers with ReLU ---
        # For each hidden layer k (0..num_hidden-1):
        #   h_k[i] = ReLU(W_k @ input_k + b_k)
        # where input_0 = x, input_k = h_{k-1} for k > 0

        hidden_vars = {}  # h_k variables
        binary_vars = {}  # z_k binary variables

        for k in range(num_hidden):
            layer_key = str(k)
            layer_dim = len(network_params[layer_key]['bias'])

            # Create index set for this layer
            idx_set = pyomo.Set(initialize=list(range(layer_dim)))
            setattr(model, f'hidden_indices_{k}', idx_set)

            # Hidden layer output variables (non-negative due to ReLU)
            h_var = pyomo.Var(idx_set, domain=pyomo.NonNegativeReals)
            setattr(model, f'h_{k}', h_var)
            hidden_vars[k] = h_var

            # Binary variables for ReLU activation
            z_var = pyomo.Var(idx_set, domain=pyomo.Binary)
            setattr(model, f'z_{k}', z_var)
            binary_vars[k] = z_var

            # Determine input variables for this layer
            if k == 0:
                input_var = model.x
                input_indices = model.input_indices
            else:
                input_var = hidden_vars[k - 1]
                input_indices = getattr(model, f'hidden_indices_{k - 1}')

            W = network_params[layer_key]['weight']
            b = network_params[layer_key]['bias']

            # Pre-activation: a_k[i] = sum_j W[i][j] * input[j] + b[i]
            def _make_pre_activation(W_layer, b_layer, in_var, in_idx):
                def pre_act(i):
                    return sum(
                        W_layer[str(i)][str(j)] * in_var[j]
                        for j in in_idx
                    ) + b_layer[str(i)]
                return pre_act

            pre_act = _make_pre_activation(W, b, input_var, input_indices)

            # Big-M ReLU constraints:
            # (1) h_k[i] <= pre_activation[i] + M * (1 - z_k[i])
            # (2) h_k[i] >= pre_activation[i]
            # (3) h_k[i] <= M * z_k[i]
            #
            # When z_k[i]=1 (neuron active): h_k[i] = pre_activation[i] (constraints 1+2 force equality)
            # When z_k[i]=0 (neuron inactive): h_k[i] = 0 (constraint 3), and pre_activation <= 0 (constraint 2)

            def _make_rule1(k_idx, pa_func, h_v, z_v):
                def rule(model, i):
                    return h_v[i] <= pa_func(i) + model.bigM * (1 - z_v[i])
                return rule

            def _make_rule2(k_idx, pa_func, h_v):
                def rule(model, i):
                    return h_v[i] >= pa_func(i)
                return rule

            def _make_rule3(k_idx, h_v, z_v):
                def rule(model, i):
                    return h_v[i] <= model.bigM * z_v[i]
                return rule

            setattr(model, f'relu_upper_{k}', pyomo.Constraint(
                idx_set, rule=_make_rule1(k, pre_act, h_var, z_var)
            ))
            setattr(model, f'relu_lower_{k}', pyomo.Constraint(
                idx_set, rule=_make_rule2(k, pre_act, h_var)
            ))
            setattr(model, f'relu_bigm_{k}', pyomo.Constraint(
                idx_set, rule=_make_rule3(k, h_var, z_var)
            ))

        # --- Output layer (no ReLU) ---
        output_key = str(num_hidden)
        output_dim = len(network_params[output_key]['bias'])

        model.output_indices = pyomo.Set(initialize=list(range(output_dim)))
        model.y = pyomo.Var(model.output_indices, domain=pyomo.Reals)

        # Input to output layer is the last hidden layer
        if num_hidden > 0:
            last_hidden_var = hidden_vars[num_hidden - 1]
            last_hidden_indices = getattr(model, f'hidden_indices_{num_hidden - 1}')
        else:
            # Edge case: no hidden layers (just input -> output)
            last_hidden_var = model.x
            last_hidden_indices = model.input_indices

        W_out = network_params[output_key]['weight']
        b_out = network_params[output_key]['bias']

        def output_layer_rule(model, i):
            return model.y[i] == sum(
                W_out[str(i)][str(j)] * last_hidden_var[j]
                for j in last_hidden_indices
            ) + b_out[str(i)]

        model.output_constraint = pyomo.Constraint(
            model.output_indices, rule=output_layer_rule
        )

        # --- Rare-event constraint: y[1] >= y[0] (classifier predicts rare) ---
        def rare_event_rule(model):
            return model.y[1] - model.y[0] >= 0

        model.rare_event_constraint = pyomo.Constraint(rule=rare_event_rule)

        # --- Objective: minimize Mahalanobis distance from distribution mean ---
        # For Gaussian: minimize ||x - mu||^2 / (2 * sigma^2)
        def objective_rule(model):
            return sum(
                ((model.x[i] - model.mu[i]) ** 2) / (2 * model.sigma ** 2)
                for i in model.input_indices
            )

        model.obj = pyomo.Objective(rule=objective_rule, sense=pyomo.minimize)

        self.model = model
        return model

    def solve(
        self,
        network_params: Dict,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Solve for dominating points using iterative cutting plane algorithm.

        Args:
            network_params: Dictionary of network weights and biases
            verbose: Whether to print solver output

        Returns:
            Array of dominating points of shape [num_points, input_dim]
        """
        # Build model
        self.build_model(network_params)

        # Set up solver
        solver = SolverFactory(self.solver_name)

        if self.solver_name == 'gurobi':
            solver.options['NonConvex'] = 2
            solver.options['MIPGap'] = 1e-4
            solver.options['TimeLimit'] = 300  # 5 min per solve

        # Solve initial problem
        results = solver.solve(self.model, tee=verbose)
        status = results.solver.status

        if verbose:
            print(f"Initial solve status: {status}")

        if status != 'ok':
            print("Warning: Initial optimization failed")
            return np.array([])

        # Extract first dominating point
        x_sol = np.array([self.model.x[i]() for i in range(self.input_dim)])
        dominating_points = [x_sol]

        if verbose:
            obj_val = self.model.obj()
            print(f"Point 1: obj={obj_val:.4f}, x={x_sol}")

        # Iterative cutting plane algorithm
        self.model.cuts = pyomo.ConstraintList()

        for iteration in range(1, self.max_iterations):
            # Add cutting plane: s_{x*}^T (x - x*) < 0
            # where s_{x*} = (x* - mu) / sigma^2 is the tilting direction
            x_prev = x_sol.copy()

            def cutting_plane_expr():
                return self.eps_ignore + sum(
                    (x_prev[i] - self.mu[i]) * (self.model.x[i] - x_prev[i])
                    for i in range(self.input_dim)
                )

            self.model.cuts.add(cutting_plane_expr() <= 0)

            # Solve again
            results = solver.solve(self.model, tee=False)
            status = results.solver.status

            if status != 'ok':
                if verbose:
                    print(f"Stopped at iteration {iteration}: no more feasible solutions")
                break

            # Extract new dominating point
            x_sol = np.array([self.model.x[i]() for i in range(self.input_dim)])
            dominating_points.append(x_sol)

            if verbose:
                obj_val = self.model.obj()
                print(f"Point {iteration + 1}: obj={obj_val:.4f}, x={x_sol}")

        self.dominating_points = np.array(dominating_points)

        if verbose:
            print(f"\nFound {len(self.dominating_points)} dominating points")

        return self.dominating_points

    def get_dominating_points(self) -> Optional[np.ndarray]:
        """Get the dominating points found by the solver."""
        return self.dominating_points
