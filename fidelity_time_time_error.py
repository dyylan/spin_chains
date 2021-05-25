from chains import Chain1d, Chain1dSubspace, Chain1dSubspaceLongRange
from hamiltonians import Hamiltonian
from states import (
    PeriodicState,
    FourierState,
    SubspaceFourierState,
    SingleExcitationState,
    SuperpositionState,
    SingleState,
)
from numerics import (
    eigenvalue_numba,
    s_parameter,
    s_k_parameter,
    delta_n,
    delta_k,
    first_order_fidelity_correction,
)
from plots import (
    plot_deltas,
    plot_always_on_fidelity,
    plot_always_on_time,
    plot_fidelity,
    plot_time,
    plot_time_comparisons,
    plot_fidelity_time_error_comparisons,
)
from fidelity_time_comparisons import fidelity_time_fast
from fits import power_fit
from data_analysis.data_handling import update_data, read_data
from numba import jit
import numpy as np

np.set_printoptions(threshold=np.inf)
import peakutils
import scipy.special
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def main(protocol, chain, alpha, save_tag):
    data = read_data(protocol, chain, alpha, save_tag)

    time_noise = 0.5
    time_error_data = {
        "spins": data["spins"],
        "optimum_gamma": data["optimum_gamma"],
        "naive_fidelity": data["naive_fidelity"],
        "fidelity": data["fidelity"],
        "fidelity_te": [],
        "time": data["time"],
        "time_noise": [],
    }

    for i, spins in enumerate(data["spins"]):
        time_error_data["fidelity_te"].append(
            time_error_sampling(
                spins,
                alpha,
                data["optimum_gamma"][i],
                data["time"][i],
                time_noise=time_noise,
                samples=5,
                open_chain=True,
                always_on=False,
            )
        )
        time_error_data["time_noise"].append(time_noise)

        print(
            f"---- Computed\n\tfidelity = {time_error_data['fidelity_te'][i]} (no noise fidelity = {time_error_data['fidelity'][i]})"
            f"\n\ttime noise = {time_noise}"
            f"\n\tfor {spins} dimensions (up to {time_error_data['spins'][-1]})"
        )

    update_data(protocol, chain, alpha, save_tag + "_time_error", time_error_data)

    fast_spins, fast_fidelities, fast_times = fidelity_time_fast(
        protocol="A",
        steps=5,
        alpha=1,
        open_chain=True,
        dt=0.01,
        rescalings=0.25,
        noise=0,
        samples=1,
        time_noise=0,
    )

    fast_spins, fast_fidelities_te, fast_times_te = fidelity_time_fast(
        protocol="A",
        steps=5,
        alpha=1,
        open_chain=True,
        dt=0.01,
        rescalings=0.25,
        noise=0,
        samples=5,
        time_noise=time_noise,
    )

    ys = [time_error_data["fidelity"], time_error_data["fidelity_te"]]
    fig, ax = plt.subplots()
    linestyles = ["dashed", "solid", "dashed", "solid"]
    for i, y in enumerate(ys):
        ax.plot(time_error_data["spins"], y, linestyle=linestyles[i])
    ax.plot(fast_spins, fast_fidelities, linestyle=linestyles[2])
    ax.plot(fast_spins, fast_fidelities_te, linestyle=linestyles[3])
    ax.legend(
        [
            f"Fidelity for RS protocol",
            f"Fidelity for RS with time noise = {time_noise}",
            f"Fidelity for fast protocol",
            f"Fidelity for fast with time noise = {time_noise}",
        ]
    )
    ax.set(xlabel="$n$")
    ax.grid()
    # plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    plt.show()

    fig, ax = plt.subplots()
    linestyles = ["solid", "solid"]

    ax.plot(time_error_data["spins"], time_error_data["time"], linestyle=linestyles[0])
    ax.plot(fast_spins, fast_times, linestyle=linestyles[1])

    ax.legend(
        [
            f"Time for RS protocol",
            f"Time for fast algorithm",
        ],
        loc=0,
    )

    ax.set(xlabel="$n$")
    ax.grid()
    # plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    plt.show()


def quantum_communication(
    spins,
    marked_strength,
    switch_time,
    time_noise,
    alpha=1,
    open_chain=False,
    start_site=1,
    final_site=15,
    always_on=False,
    long_time=False,
    dt=0.01,
):

    switch_time1 = switch_time + np.random.normal(0, time_noise)
    switch_time2 = switch_time + np.random.normal(0, time_noise)

    final_state = SingleState(spins, final_site, single_subspace=True)
    init_state = SingleState(spins, start_site, single_subspace=True)

    chain = Chain1dSubspaceLongRange(
        spins=spins,
        dt=dt,
        alpha=alpha,
        open_chain=open_chain,
    )

    chain.add_marked_site(start_site, marked_strength, gamma_rescale=True)
    chain.initialise(init_state)

    if always_on:
        switch_time2 = 3.5 * switch_time2
        chain.update_marked_site(final_site, 1)
    else:
        chain.time_evolution(time=switch_time1)
        chain.update_marked_site(start_site, -1)
        chain.update_marked_site(final_site, 1)

    times, states = chain.time_evolution(time=1.5 * switch_time2, reset_state=False)

    return times, states, final_state, chain


def fidelity_time(spins, alpha, gamma, evolution_time, time_noise, always_on):
    times, states, final_state, chain = quantum_communication(
        spins,
        gamma,
        evolution_time,
        time_noise,
        alpha=alpha,
        open_chain=True,
        start_site=1,
        final_site=spins,
        always_on=always_on,
    )
    qst_fidelity = chain.overlaps_evolution(final_state.subspace_ket, states)
    peaks, _ = find_peaks(qst_fidelity, height=(0.2, 1.05))
    try:
        time = times[peaks[0]]
        fidelity = qst_fidelity[peaks[0]]
    except IndexError:
        fidelity = 0
        time = 0
    return fidelity, time


def time_error_sampling(
    spins, alpha, gamma, time, time_noise, samples, open_chain, always_on=False
):
    if always_on:
        evolution_time = time
    else:
        evolution_time = time / 2
    fidelities = [
        fidelity_time(
            spins,
            alpha,
            gamma,
            evolution_time,
            time_noise,
            always_on=always_on,
        )[0]
        for _ in range(samples)
    ]
    mean_fidelity = np.mean(fidelities)
    return mean_fidelity


if __name__ == "__main__":
    alpha = 1
    protocol = "reverse_search"
    chain = "open"
    save_tag = "optimum_gammas"
    main(protocol, chain, alpha, save_tag)
