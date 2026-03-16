import os
import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_dataset
from src.dynamics import pendulum_energy


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    train_t, train_x0, train_x = generate_dataset(
        n_trajectories=200,
        t_span=(0.0, 10.0),
        n_time_points=200,
        theta_range=(-np.pi, np.pi),
        omega_range=(-1.5, 1.5),
        seed=1,
    )

    test_t, test_x0, test_x = generate_dataset(
        n_trajectories=50,
        t_span=(0.0, 10.0),
        n_time_points=200,
        theta_range=(-np.pi, np.pi),
        omega_range=(-1.5, 1.5),
        seed=1,
    )

    np.save("data/train_t.npy", train_t)
    np.save("data/train_x0.npy", train_x0)
    np.save("data/train_x.npy", train_x)

    np.save("data/test_t.npy", test_t)
    np.save("data/test_x0.npy", test_x0)
    np.save("data/test_x.npy", test_x)

    print("Saved:")
    print("  data/train_t.npy")
    print("  data/train_x0.npy")
    print("  data/train_x.npy")
    print("  data/test_t.npy")
    print("  data/test_x0.npy")
    print("  data/test_x.npy")
    print()
    print(f"train_x shape: {train_x.shape}")
    print(f"test_x shape:  {test_x.shape}")

    # Sanity-check plot: a few trajectories in phase space
    plt.figure(figsize=(6, 6))
    for i in range(min(10, train_x.shape[0])):
        theta = train_x[i, :, 0]
        omega = train_x[i, :, 1]
        plt.plot(theta, omega, alpha=0.8)
    plt.xlabel("theta")
    plt.ylabel("omega")
    plt.tight_layout()
    plt.savefig("results/figures/train_phase_samples.png", dpi=150)
    plt.close()

    # Sanity-check energy conservation on one trajectory
    energy = np.array([
        pendulum_energy(state) for state in train_x[0]
    ])
    energy_error = energy - energy[0]

    plt.figure(figsize=(8, 4))
    plt.plot(train_t, energy_error)
    plt.xlabel("t")
    plt.ylabel("energy error")
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    plt.tight_layout()
    plt.savefig("results/figures/train_sample_energy_error.png", dpi=150)
    plt.close()

    print("Saved sanity-check figures in results/figures/")


if __name__ == "__main__":
    main()