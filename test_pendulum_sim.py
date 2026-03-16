import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from src.dynamics import pendulum_rhs, pendulum_energy


def main():
    t_span = (0.0, 20.0)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    x0 = np.array([1.0, 0.0])  # initial angle, initial angular velocity

    sol = solve_ivp(
        pendulum_rhs,
        t_span,
        x0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-9,
    )

    theta = sol.y[0]
    omega = sol.y[1]
    energy = np.array([pendulum_energy([th, om]) for th, om in zip(theta, omega)])

    plt.figure(figsize=(8, 4))
    plt.plot(sol.t, theta, label="theta(t)")
    plt.plot(sol.t, omega, label="omega(t)")
    plt.xlabel("t")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/sample_time_series.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(theta, omega)
    plt.xlabel("theta")
    plt.ylabel("omega")
    plt.tight_layout()
    plt.savefig("results/figures/sample_phase_portrait.png", dpi=150)
    plt.close()

    energy_error = energy - energy[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sol.t, energy_error)
    ax.set_xlabel("t")
    ax.set_ylabel("energy error")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    fig.tight_layout()
    fig.savefig("results/figures/sample_energy_error.png", dpi=150)
    plt.close(fig)

    plt.figure(figsize=(8, 4))
    plt.plot(sol.t, energy)
    plt.xlabel("t")
    plt.ylabel("energy")
    plt.ticklabel_format(axis="y", style="plain", useOffset=False)
    plt.tight_layout()
    plt.savefig("results/figures/sample_energy.png", dpi=150)
    plt.close()

    print("Saved figures to results/figures/")


if __name__ == "__main__":
    main()