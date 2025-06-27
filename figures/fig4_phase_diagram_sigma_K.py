# Generate Figure 4 for the PRL paper
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.plot_utils import apply_figure4_style, save_figure
apply_figure4_style()

# ------------------ Load simulation data ------------------
base_dir = os.path.join(project_root, "experiments", "results", "figure4")
out_path = os.path.join(project_root, "figures", "outputs", "fig4_phase_diagram_sigma_K.png")

phi_mean_array = np.load(os.path.join(base_dir, "phi_mean_array_N1000_T200_K20_sigma101_numseed5.npy"))
K, S =  phi_mean_array.shape
K_vals = np.linspace(1, K, K)
sigma_vals = np.linspace(0.5, 0, S)

# ------------------ Plot figure ------------------
fig, (ax_slice, ax_main) = plt.subplots(1, 2, figsize=(3.375, 1.5), gridspec_kw={'width_ratios': [2, 3]})

# Contour plot: K vs sigma → mean polar angle
cf = ax_main.contourf(sigma_vals, K_vals, phi_mean_array, levels=50, cmap="viridis")
cbar = fig.colorbar(cf, ax=ax_main, format="%.3f")

ax_main.set_ylim(1, 20)
ax_main.set_yticks([1, 5, 10, 15, 20])
ax_main.set_xlabel(r'$\sigma\ \left(\omega_i \sim \mathcal{N}(0.5,\ \sigma^2)\right)$')
ax_main.set_ylabel(r'$K$')
ax_main.set_title(r'Mean polar angle: $\frac{1}{N} \sum_{i} \phi_i(T)$')

# Slice plot: sigma vs polar angle for selected K
K_slices = [2, 4, 8]
colors = ['C0', 'C1', 'C2']

for K_val, color in zip(K_slices, colors):
    ax_main.axhline(y=K_val, color=color, linestyle='--')
    idx = np.argmin(np.abs(K_vals - K_val))
    ax_slice.plot(sigma_vals, phi_mean_array[idx], label=f"$K={K_val}$", color=color)

ax_slice.set_xlabel(r'$\sigma$')
ax_slice.set_ylabel(r'$\frac{1}{N} \sum_i \phi_i(T)$')
ax_slice.set_xlim([0, 0.5])
ax_slice.set_ylim([-1e-4, 0.1])
ax_slice.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.10])
ax_slice.grid(True)
ax_slice.legend()
plt.tight_layout(pad=2.0)

# ------------------ Save figure ------------------
save_figure(out_path)
print(f"Figure saved to {out_path}")