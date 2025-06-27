import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps 
import os

# ------------------ Plot style settings for each figure ------------------

def apply_figure1_style():
    """Set matplotlib rcParams for Figure 1."""
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 5,
        "figure.titlesize": 8,
        "lines.linewidth": 0.6,
        "lines.markersize": 3,
        "font.family": "serif",        
    })

def apply_figure2_style():
    """Set matplotlib rcParams for Figure 2 (small font for 3D subplots)."""
    plt.rcParams.update({
        "font.size": 3,
        "axes.labelsize": 4,
        "axes.titlesize": 4,
        "xtick.labelsize": 3,
        "ytick.labelsize": 3,
        "legend.fontsize": 2,
        "figure.titlesize": 4,
        "lines.linewidth": 1.0,
        "lines.markersize": 1,
        "font.family": "serif",        
    })

def apply_figure3_style():
    """Set matplotlib rcParams for Figure 3."""
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,           
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "figure.titlesize": 8,
        "lines.linewidth": 0.6,
        "lines.markersize": 3,
        "font.family": "serif",
    })

def apply_figure4_style():
    """Set matplotlib rcParams for Figure 4 (compact layout)."""
    plt.rcParams.update({
        "font.size": 5.5,
        "axes.labelsize": 5.5,
        "axes.titlesize": 5.5,
        "xtick.labelsize": 4.5,
        "ytick.labelsize": 4.5,
        "legend.fontsize": 3.5,
        "figure.titlesize": 5.5,
        "lines.linewidth": 0.5,
        "lines.markersize": 0.5,
        "font.family": "serif",
    })

# ------------------ Analysis functions ------------------

def compute_phi_variance(position):
    """
    Compute variance of the polar angle φ_i(t) over agents.
    position: shape (N, 3, T)
    returns: shape (T,)
    """
    z = position[:, 2, :]
    phi = jnp.arccos(jnp.clip(z, -1.0, 1.0))
    return jnp.var(phi, axis=0)

def compute_R(position):
    """
    Compute order parameter R(t) from azimuthal angle θ_i(t).
    position: shape (N, 3, T)
    returns: shape (T,)
    """
    theta = jnp.arctan2(position[:, 1, :], position[:, 0, :])
    cos_sum = jnp.sum(jnp.cos(theta), axis=0)
    sin_sum = jnp.sum(jnp.sin(theta), axis=0)
    N = position.shape[0]
    return jnp.sqrt((cos_sum/N)**2 + (sin_sum/N)**2)

def compute_phi_mean(position):
    """
    Compute mean of the polar angle φ_i(t) over agents.
    position: shape (N, 3, T)
    returns: shape (T,)
    """
    z = position[:, 2, :]
    phi = jnp.arccos(jnp.clip(z, -1.0, 1.0))
    return jnp.mean(phi, axis=0)


# ------------------ 3D trajectory plotting ------------------

def plot_tail_with_highlighted_ends(positions, t_eval, center_time, tail_length=300, colors=None, ax = None, file_path=None):
    """
    Plot 3D trajectories on a sphere with fading tails and highlighted endpoints.
    Used in PRL Figure 2.
    """
    # T = positions.shape[-1]
    t_array = np.asarray(t_eval)
    
    center_idx = np.argmin(np.abs(t_array - center_time))
    start_idx = max(center_idx - tail_length, 0)
    tail_idxs = np.arange(start_idx, center_idx + 1)

    # Generate sphere surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

    # Apply colormap to sphere
    normed_z = (z_sphere - z_sphere.min()) / (z_sphere.max() - z_sphere.min()) - 0.1
    face_color = colormaps["ocean"](normed_z)
    ax.plot_surface(
        x_sphere, y_sphere, z_sphere,
        facecolors=np.tile(face_color, (x_sphere.shape[0], x_sphere.shape[1], 1)),
        rstride=1, cstride=1,
        antialiased=True,
        linewidth=0,
        shade=False,
        alpha = 0.4,
        zorder =1,
        rasterized=True
    )
    N = positions.shape[0]
    if colors is None:
        colors_tail = ['blue'] * N
        colors_head = ['red'] * N

    # Plot trajectories (fading tail)
    for i in range(N):
        traj = positions[i]
        x = traj[0, tail_idxs]
        y = traj[1, tail_idxs]
        z = traj[2, tail_idxs]
        ax.plot(x, y, z, color=colors_tail[i], alpha=0.8, linewidth=0.3)

    # Highlight final position
    for i in range(N):
        traj = positions[i]  # (3, T)
        x = traj[0, tail_idxs]
        y = traj[1, tail_idxs]
        z = traj[2, tail_idxs]
        x_end, y_end, z_end = x[-1], y[-1], z[-1]
        ax.scatter(x_end, y_end, z_end, color=colors_head[i], s=0.1, label=f'Agent {i+1}', depthshade=False, zorder=2)

    # Set viewpoint and axis formatting
    ax.view_init(elev=25, azim=35)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xticklabels([]) 
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(axis='both', which='major', width=0.05, length=0.5)
    ax.grid(True) 

    # Thin axis lines and grid
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['grid']['linewidth'] = 0.3 

        axis._axinfo['axisline']['linewidth'] = 0.01 
        axis._axinfo['axisline']['color'] = (0.5, 0.5, 0.5) 
    return ax

# ------------------ Save utility ------------------

def save_figure(path, dpi=600):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=dpi)
