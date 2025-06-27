# Data generation for PRL Figure 2
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
save_dir = os.path.join(project_root, "experiments", "results", "figure2")
os.makedirs(save_dir, exist_ok=True)

# ------------------ Simulation parameters ------------------
N = 20               # Number of oscillators
K = 1                # Coupling strength
T = 200.0            # Final time
steps = 20000        # Number of time steps
alpha = 0.6          # Initial concentration towards the northern hemisphere
mu = 0.5             # Mean frequency
sigma = 0.288        # Frequency std dev
seed = 5             # Random seed

# ------------------ Initialize state ------------------
omega_array = random.normal(random.key(0), shape=(N,)) * sigma + mu
x0 = random_points_on_s2(N, alpha=alpha, seed=seed) 
y_init = x0.reshape(-1)
t_eval = jnp.linspace(0.0, T, steps)

# ------------------ Run simulations and save results ------------------
position = run_simulation(N, K, omega_array, x0, t1=T, steps=steps)
np.save(save_dir +"/trajectory.npy", position)
np.save(save_dir +"/t_eval.npy", t_eval)
print(f"Saved")