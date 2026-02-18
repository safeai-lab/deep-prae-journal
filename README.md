# Deep-PrAE: Deep Probabilistic Rare Event Estimation

A neural network-based importance sampling framework for estimating extremely small probabilities of rare events (as small as $10^-24$). The method combines deep learning with dominating points identification for optimal importance sampling with theoretical guarantees.


## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mansurarief/DeepPrAE-dev.git

python -m venv .venv
source .venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```


## Six Experiments

### 1. 2D Sigmoid Functions
- **Problem**: $P(g(X) > \gamma)$ with some nonlinear $g$ and $X \sim N([5,5], 0.25I^{2 \times 2}$)

```bash
python run_all_examples.py --examples 1 --gamma 1.8 --visualize
```

### 2. Complement of 5D Ball
- **Problem**: $P(\|X\| \geq \gamma)$ in 5 dimensions, $X \sim N(0, 0.5I^{5 \times 5}$)

```bash
python run_all_examples.py --examples 2 --gamma 4.75
```

### 3. Random Walk Excursion
- **Problem**: $P(\max_t S_t \geq \gamma)$ where $S_t =$ cumulative sum, $X_i \sim N(0, 1), T=10$ time steps

```bash
python run_all_examples.py --examples 3
```

### 4. Non-Gaussian Distribution
- **Problem**: Exponential + Generalized Pareto marginals with Gaussian copula (Non-Gaussian Example)

```bash
python run_all_examples.py --examples 4
```

### 5. Rare-event Set with Hole
- **Problem**: Set with hole, non-monotonic set (disconnected rare-event regions)

```bash
python run_all_examples.py --examples 5
```

### 6. Intelligent Driving Safety
- **Problem**: Autonomous vehicle crash probability (high dimensional, complex dynamics)

```bash
python run_all_examples.py --examples 6
```

## Repository Structure

```
Deep-PRAE/
├── deepprae/                  # Main package
│   ├── core/                  # Core algorithm components
│   │   ├── algorithm.py       # Main Deep-PrAE algorithm
│   │   ├── networks.py        # Neural network architectures
│   │   ├── optimization.py    # Dominating point optimization
│   │   ├── sampling.py        # Proposal distribution sampling
│   │   └── estimation.py      # Probability estimation
│   ├── configs/               # Experiment configurations
│   │   └── experiment_configs.py
│   ├── experiments/           # Six benchmark experiments
│   │   ├── example1_2d_sigmoid.py
│   │   ├── example2_ball_complement.py
│   │   ├── example3_random_walk.py
│   │   ├── example4_non_gaussian.py
│   │   ├── example5_hole.py
│   │   └── example6_intelligent_driving.py
│   └── utils/                 # Utility functions
├── run_all_examples.py        # Unified script to run examples
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```


## Dependency

- Gurobi 13.0, with unlimited size model (needed to solve MIP model)