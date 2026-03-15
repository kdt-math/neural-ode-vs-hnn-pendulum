import numpy as np

def pendulum_rhs(t, state):
    """
    Nondimensionalized simple pendulum dynamics.

    state = [theta, omega]
    dtheta/dt = omega
    domega/dt = -sin(theta)
    """
    theta, omega = state
    dtheta = omega
    domega = -np.sin(theta)
    return np.array([dtheta, domega], dtype=float)

def pendulum_energy(state):
    """
    H(theta, omega) = 0.5 * omega^2 + (1 - cos(theta))
    """
    theta, omega = state
    return 0.5 * omega**2 + (1.0 - np.cos(theta))