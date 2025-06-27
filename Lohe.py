
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
from diffrax import ODETerm, SaveAt, Dopri5, diffeqsolve, PIDController

# ------------------ Lohe Model Dynamics ------------------

def ode_wrapper(t, y_flat, N, omega_array, K=1.0):
    Xi = y_flat.reshape(N, 3)
    Xi = Xi / jnp.linalg.norm(Xi, axis=1, keepdims=True) # ensure unit vectors
    
    # Generator of z-axis rotation
    L = jnp.array([[0.0, -1.0, 0.0],
                   [1.0,  0.0, 0.0],
                   [0.0,  0.0, 0.0]])

    # Rotation term due to natural frequencies
    def omega_term(i):
        return omega_array[i] * (L @ Xi[i])
    term1 = vmap(omega_term)(jnp.arange(N))

    # Alignment interaction term
    def alignment_term(i):
        xi = Xi[i]
        norm_xi_sq = jnp.dot(xi, xi)
        def inner(k):
            xk = Xi[k]
            return norm_xi_sq * xk - jnp.dot(xi, xk) * xi
        return vmap(inner)(jnp.arange(N)).sum(axis=0)

    term2 = vmap(alignment_term)(jnp.arange(N))
    dx = term1 + (K / N) * term2
    return dx.reshape(-1)

# ------------------ Initialization Utilities ------------------

def random_points_on_s2(n, alpha=0.5, seed=0):
    """
    Sample n points on S^2 with optional bias toward the northern hemisphere.
    """
    rng = np.random.default_rng(seed)
    n_north = int(n * alpha)
    n_south = n - n_north

    # Sample polar and azimuthal angles
    theta_n = rng.uniform(0, 2 * np.pi, n_north)
    phi_n = np.arccos(rng.uniform(0, 1, n_north))
    theta_s = rng.uniform(0, 2 * np.pi, n_south)
    phi_s = np.arccos(-rng.uniform(0, 1, n_south))

    theta = np.concatenate([theta_n, theta_s])
    phi = np.concatenate([phi_n, phi_s])

    # Convert to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return jnp.stack([x, y, z], axis=1)

def sample_lorentzian(key, N, loc=0.0, scale=1.0):
    """
    Sample N values from a Lorentzian distribution.
    """
    return random.cauchy(key, shape=(N,)) * scale + loc

# ------------------ Simulation Wrapper ------------------

def run_simulation(N, K, omega_array, x0, t0=0.0, t1=200.0, steps=20000):
    """
    Simulate the ODE system using a 5th-order Dormand–Prince solver.
    """
    y_init = x0.reshape(-1)
    t_eval = jnp.linspace(t0, t1, steps)

    def rhs(t, y, args):
        N, omega_array, K = args
        return ode_wrapper(t, y, N, omega_array, K)
    
    term = ODETerm(rhs)
    sol = diffeqsolve(
        terms=term,
        solver=Dopri5(),
        t0=t0,
        t1=t1,
        dt0=0.01,
        y0=y_init,
        saveat=SaveAt(ts=t_eval),
        args=(N, omega_array, K),
        stepsize_controller=PIDController(rtol=1e-5, atol=1e-7),
        max_steps=8 * 8192
    )
    return sol.ys.T.reshape(N, 3, -1)
