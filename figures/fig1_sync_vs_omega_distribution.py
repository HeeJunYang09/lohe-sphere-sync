# Generate Figure 1 for the PRL paper
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.plot_utils import apply_figure1_style, save_figure, compute_phi_variance, compute_R
apply_figure1_style()

# ------------------ Load simulation data ------------------
base_dir = os.path.join(project_root, "experiments", "results", "figure1")
out_path = os.path.join(project_root, "figures", "outputs", "fig1_sync_vs_omega_distribution.png")
files = [
    "omega_05.npy",
    "uniform_01.npy",
    "normal_05_sigma_0288.npy",
    "Lorentzian_05_gamma_01.npy",
]
legend_list = [
    r'$\omega_i \equiv 0.5$',
    r'$\omega_i \sim U(0,1)$',
    r'$\omega_i \sim \mathcal{N}(0.5,\ \sigma^2),\ \sigma=0.288$',
    r'$\omega_i \sim \text{Lorentzian}(0.5,\ \gamma),\ \gamma=0.1$'
]
results = [np.load(os.path.join(base_dir, f)) for f in files]

# ------------------ Plot figure ------------------
fig, ax = plt.subplots(2,1, figsize=(3.375, 2.0), sharex=True)
real_ticks = np.arange(0, 22, 2)
tick_locations = real_ticks / 0.01

for data, label in zip(results, legend_list):
    phi = compute_phi_variance(data)
    R = compute_R(data)
    ax[0].plot(phi, label=label)
    ax[1].plot(1 - R)

ax[0].legend()
ax[0].set_ylabel(r"$\mathrm{Var}[\phi_i(t)]$")
ax[0].grid(True)
ax[1].set_ylabel(r"$1 - R(t)$")
ax[1].set_xlabel("Time $t$")
ax[1].set_ylim(0, 1.05)
ax[1].set_xlim([0, 200])
ax[1].set_xticks(tick_locations)
ax[1].set_xticklabels([f"{t:.0f}" for t in real_ticks])
ax[1].grid(True)

# ------------------ Save figure ------------------
save_figure(out_path)
print(f"Figure saved to {out_path}")
