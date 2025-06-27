# Data generation for PRL Figure 3
import os, sys
import numpy as np
import jax.numpy as jnp
from jax import random

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Lohe import *

# Create output directory
save_dir = os.path.join(project_root, "experiments", "results", "figure3")
os.makedirs(save_dir, exist_ok=True)

# ------------------ Simulation parameters ------------------
N = 1000             # Number of oscillators
K = 1.0              # Coupling strength
T = 200.0            # Final time
steps = 20000        # Number of time steps
alpha = 0.7          # Initial concentration towards the northern hemisphere
seed = 0             # Random seed

# ------------------ Natural frequency distributions ------------------
distributions = {
    "omega_05": lambda key: 0.5 * jnp.ones(shape= (N,)),
    "uniform_01": lambda key: random.uniform(key, (N,), minval=0.0, maxval=1.0),
    "normal_05_sigma_0288": lambda key: random.normal(key, (N,)) * 0.288 + 0.5,
    "Lorentzian_05_gamma_01": lambda key: sample_lorentzian(key, N, loc=0.5, scale=0.1),
}

# ------------------ Run simulations and save results ------------------
x0 = random_points_on_s2(N, alpha=alpha, seed=0) # Initial positions on S^2
t_eval = jnp.linspace(0.0, T, steps)             # Time evaluation grid

for name, sampler in distributions.items():
    omega_array = sampler(random.key(seed))      # Natural frequencies
    position = run_simulation(N, K, omega_array, x0, t1=T, steps=steps)
    out_path = os.path.join(save_dir, f"{name}.npy")
    np.save(out_path, position)
    print(f"Saved: {out_path}")