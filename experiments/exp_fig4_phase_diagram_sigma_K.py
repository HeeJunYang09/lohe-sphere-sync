# Data generation for PRL Figure 4
import os, sys
import numpy as np
import jax.numpy as jnp
from jax import random

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Lohe import *

# Create output directory
save_dir = os.path.join(project_root, "experiments", "results", "figure4")
os.makedirs(save_dir, exist_ok=True)

# ------------------ Simulation parameters ------------------
N = 1000                               # Number of oscillators
T = 200.0                              # Final time
steps = 20000                          # Time steps
K_vals = np.linspace(1, 20, 20)        # Coupling strengths
sigma_vals = np.linspace(0, 0.5, 101)  # Frequency std deviations
num_seeds = 5                          # Number of seeds per setting
alpha = 0.7                            # Initial concentration towards the northern hemisphere
seed = 42                              # Global random seed  

# Output file name
save_name = f"phi_mean_array_N{N}_T{int(T)}_K{len(K_vals)}_sigma{len(sigma_vals)}_numseed{num_seeds}.npy"
out_path = os.path.join(save_dir, save_name)

# ------------------ Run simulations ------------------
def simulate_phi_mean_array():
    phi_mean_array = np.zeros((len(K_vals), len(sigma_vals)))
    default_key = random.key(seed)
    keys = random.split(default_key, num_seeds * len(K_vals) * len(sigma_vals))
    x0 = random_points_on_s2(N, alpha=alpha, seed=0)
    
    key_idx = 0
    for i_K, K in enumerate(K_vals):
        for j_sigma, sigma in enumerate(sigma_vals):
            phi_total = 0
            for _ in range(num_seeds):
                omega_key = keys[key_idx]; key_idx += 1
                omega_array = random.normal(omega_key, shape=(N,)) * sigma + 0.5
                position = run_simulation(N, K, omega_array, x0, t1=T, steps=steps)
                z_mean = jnp.mean(position[:, 2, :], axis=0)
                phi_last = jnp.arccos(jnp.clip(z_mean[-1], -1.0, 1.0)) # Final mean polar angle
                phi_total += phi_last
            phi_mean_array[i_K, j_sigma] = phi_total / num_seeds
    return phi_mean_array

# Save result
phi_mean_array = simulate_phi_mean_array()
np.save(out_path, phi_mean_array)
print(f"Saved: {out_path}")

