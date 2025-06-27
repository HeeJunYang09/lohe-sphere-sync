# Generate Figure 2 for the PRL paper
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.plot_utils import apply_figure2_style, save_figure, plot_tail_with_highlighted_ends
apply_figure2_style()

# ------------------ Load simulation data ------------------
base_dir = os.path.join(project_root, "experiments", "results", "figure2")
position = np.load(os.path.join(base_dir, "trajectory.npy"))
t_eval = np.load(os.path.join(base_dir, "t_eval.npy"))
out_path = os.path.join(project_root, "figures", "outputs", "fig2_3d_trajectory_on_sphere_.png")

# ------------------ Plot figure ------------------
center_time_list = [0, 1, 3, 12, 25, 100]
fig = plt.figure(figsize=(3.375, 2.9))
axs = [fig.add_subplot(2, 3, i + 1, projection='3d') for i in range(6)]
labels = [r'(a) $t=0$', r'(b) $t=1$', r'(c) $t=3$', r'(d) $t=12$', r'(e) $t=25$', r'(f) $t=100$']

for idx, center_time in enumerate(center_time_list):
    axs[idx] = plot_tail_with_highlighted_ends(
        positions=position[0:20], 
        t_eval=t_eval, 
        center_time=center_time, 
        ax=axs[idx],
        tail_length=300
    )
    axs[idx].text2D(0.5, -0.05, labels[idx], transform=axs[idx].transAxes, ha='center', va='top', fontsize=6)

fig.subplots_adjust(wspace=0.05, hspace=0.05)

# ------------------ Save figure ------------------
save_figure(out_path)
print(f"Figure saved to {out_path}")

