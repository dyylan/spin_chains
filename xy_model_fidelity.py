import numpy as np
import matplotlib.pyplot as plt

from spin_chains.functions.protocols import xy_model_exp, xy_model_exp_noise
from spin_chains.functions.fits import sqrt_power_fit
from spin_chains.quantum import iontrap, chains
from spin_chains.data_analysis import data_handling


def xy_model_fidelity(n_ions, target_alpha, time, t2, samples=100, dt=1e-7):
    data = data_handling.read_data_spin(
        protocol="experimental/always_on_fast_xy",
        chain="open",
        alpha=target_alpha,
        save_tag="optimum_gammas_end_n_mu_min",
        spins=n_ions,
    )
    mu = np.ones(1) * data["mu"]  # 8 ions
    z_trap_frequency = data["z_trap_frequency"]

    times, xy_states, _ = xy_model_exp(
        n_ions,
        time,
        mu=mu,
        omega=np.array([2 * np.pi * 6e6, 2 * np.pi * 5e6, z_trap_frequency]),
        dt=dt,
    )
    times_noisy, xy_noisy_states, _ = xy_model_exp_noise(
        n_ions,
        time,
        mu=mu,
        omega=np.array([2 * np.pi * 6e6, 2 * np.pi * 5e6, z_trap_frequency]),
        t2=t2,
        samples=samples,
        dt=dt,
    )

    assert times == times_noisy

    fidelities = []
    for i, xy_state in enumerate(xy_states):
        fidelities.append(
            np.mean(
                chains.Chain.overlaps_evolution(
                    xy_state, [xy_noisy_state[i] for xy_noisy_state in xy_noisy_states]
                )
            )
        )

    return times, fidelities, mu, z_trap_frequency


if __name__ == "__main__":
    n_ions = 8
    target_alpha = 0.2
    phonon_mode_number = 1
    time = 0.01
    t2 = 0.0030
    samples = 1000
    update_data = False

    times, fidelities, mu, z_trap_frequency = xy_model_fidelity(
        n_ions=n_ions,
        target_alpha=target_alpha,
        time=time,
        t2=t2,
        samples=samples,
        dt=1e-5,
    )

    if update_data:
        if phonon_mode_number == 1:
            data_handling.update_data(
                protocol="spin_boson_single_mode",
                chain="open",
                alpha=target_alpha,
                save_tag=f"xy_model_fidelity_n={n_ions}_t2={t2}_mu={mu[0]}_z_trap={z_trap_frequency}",
                data_dict={"time": times, "fidelity": fidelities},
                replace=True,
            )
        elif phonon_mode_number == 2:
            data_handling.update_data(
                protocol="spin_boson_two_mode",
                chain="open",
                alpha=target_alpha,
                save_tag=f"xy_model_fidelity_n={n_ions}_t2={t2}_mu={mu[0]}_z_trap={z_trap_frequency}",
                data_dict={"time": times, "fidelity": fidelities},
                replace=True,
            )

    fig, ax = plt.subplots(figsize=[8, 8])
    ax.plot(times, fidelities)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fidelity")

    ax.plot()
    plt.show()
