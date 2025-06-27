# Generate Figure 3 for the PRL paper
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.plot_utils import apply_figure3_style, save_figure, compute_phi_mean
apply_figure3_style()

# ------------------ Load simulation data ------------------
base_dir = os.path.join(project_root, "experiments", "results", "figure3")
out_path = os.path.join(project_root, "figures", "outputs", "fig3_avg_polar_angle_vs_time.png")
files = [
    "omega_05.npy",
    "uniform_01.npy",
    "normal_05_sigma_0288.npy",
    "Lorentzian_05_gamma_01.npy"
]
legend_list = [
    r'$\omega_i \equiv 0.5$',
    r'$\omega_i \sim U(0,1)$',
    r'$\omega_i \sim \mathcal{N}(0.5,\ \sigma^2),\ \sigma=0.288$',
    r'$\omega_i \sim \text{Lorentzian}(0.5,\ \gamma),\ \gamma=0.1$'
]
results = [np.load(os.path.join(base_dir, f)) for f in files]

# ------------------ Plot figure ------------------
fig, ax = plt.subplots(figsize=(3.375, 1.5))
tick_times = np.arange(0, 225, 25)
tick_locs = tick_times / 0.01

for result, label in zip(results, legend_list):
    phi = compute_phi_mean(result)
    ax.plot(phi, label=label)

phi_limit = 0.0
ax.set_xlim([0, 20000])
ax.set_ylabel(r"$\phi$ (radians)")
ax.set_xlabel(r"Time $t$")
ax.set_xticks(tick_locs)
ax.set_xticklabels([f"{t:.0f}" for t in tick_times])
ax.axhline(y=phi_limit, color='red', linestyle='--', linewidth=1)
ax.set_ylabel(r'$\frac{1}{N}\sum_i \phi_i(t)$')
ax.text(
    x=1.19 * ax.get_xlim()[1],
    y=phi_limit - 0.02,
    s=r"$\phi = {:.2f}$".format(phi_limit),
    color='red',
    fontsize=8,
    ha='right'
)
ax.legend()
ax.grid(True)

# ------------------ Save figure ------------------
save_figure(out_path)
print(f"Figure saved to {out_path}")