import numpy as np
from scipy.integrate import solve_ivp

from src.dynamics import pendulum_rhs

def sample_initial_conditions(
        n_samples,
        theta_range=(-np.pi, np.pi),
        omega_range=(-1.5,1.5),
        seed=None
):
    rng = np.random.default_rng(seed)

    theta0 = rng.uniform(theta_range[0], theta_range[1], size=n_samples)
    omega0 = rng.uniform(omega_range[0], omega_range[1], size=n_samples)

    x0 = np.stack([theta0, omega0], axis=1)
    return x0

def simulate_trajectory(x0, t_span, t_eval, rtol=1e-9, atol=1e-9):
    sol = solve_ivp(
        pendulum_rhs,
        t_span,
        x0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")

    # shape: (T, 2)
    x = sol.y.T
    return x


def generate_dataset(
    n_trajectories,
    t_span,
    n_time_points,
    theta_range=(-np.pi, np.pi),
    omega_range=(-1.5, 1.5),
    seed=None,
):
    t_eval = np.linspace(t_span[0], t_span[1], n_time_points)
    x0_all = sample_initial_conditions(
        n_trajectories,
        theta_range=theta_range,
        omega_range=omega_range,
        seed=seed,
    )

    trajectories = []
    for x0 in x0_all:
        x = simulate_trajectory(x0, t_span, t_eval)
        trajectories.append(x)

    # shape: (N, T, 2)
    x_all = np.stack(trajectories, axis=0)

    return t_eval, x0_all, x_all