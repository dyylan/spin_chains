from spin_chains.quantum import chains
from spin_chains.quantum import hamiltonians
from spin_chains.quantum import states
from spin_chains.functions import numerics
from spin_chains.plots import plots
from spin_chains.data_analysis import data_handling

from numba import jit
import numpy as np

import scipy.special
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rc("text", usetex=True)
font = {"family": "serif", "size": 16, "serif": ["computer modern roman"]}
plt.rc("font", **font)


def main(
    spins,
    alpha,
    gamma,
    evolution_time,
    open_chain,
    always_on,
    mid_n=False,
    end_n=False,
    n_factor=0,
    n_diff=0,
):
    data = {
        "gamma": gamma,
        "fidelities": [],
        "times": [],
    }
    fidelities, times = fidelity_time(
        spins,
        alpha,
        gamma,
        evolution_time,
        open_chain,
        always_on,
        mid_n,
        end_n,
        n_factor,
        n_diff,
    )
    data["fidelities"].extend(fidelities)
    data["times"].extend(times)
    print(f"----> Computed fidelities for spins = {spins}, gamma = {gamma}")
    return data


def quantum_communication(
    spins,
    marked_strength,
    switch_time,
    alpha=1,
    open_chain=False,
    start_site=1,
    final_site=15,
    always_on=False,
    dt=0.01,
):
    final_state = states.SingleState(spins, final_site, single_subspace=True)
    init_state = states.SingleState(spins, start_site, single_subspace=True)

    chain = chains.Chain1dSubspaceLongRange(
        spins=spins,
        dt=dt,
        alpha=alpha,
        open_chain=open_chain,
    )

    chain.add_marked_site(start_site, marked_strength, gamma_rescale=True)
    chain.initialise(init_state)

    if always_on:
        if open_chain:
            switch_time = 1.5 * switch_time
        else:
            switch_time = 1.5 * switch_time
        chain.update_marked_site(final_site, 1)
    else:
        chain.time_evolution(time=switch_time)
        chain.update_marked_site(start_site, -1)
        chain.update_marked_site(final_site, 1)

    times, psi_states = chain.time_evolution(time=1.5 * switch_time, reset_state=False)

    return times, psi_states, final_state, chain


def fidelity_time(
    spins,
    alpha,
    gamma,
    evolution_time,
    open_chain,
    always_on,
    mid_n,
    end_n,
    n_factor,
    n_diff,
):
    if n_factor:
        final_site = spins // n_factor
    elif n_diff:
        final_site = spins - n_diff
    else:
        if mid_n:
            final_site = spins // 2
        elif end_n:
            final_site = spins
        elif open_chain:
            final_site = spins
        else:
            final_site = spins // 2
    times, psi_states, final_state, chain = quantum_communication(
        spins,
        gamma,
        evolution_time,
        alpha=alpha,
        open_chain=open_chain,
        start_site=1,
        final_site=final_site,
        always_on=always_on,
    )
    qst_fidelity = chain.overlaps_evolution(final_state.subspace_ket, psi_states)
    return qst_fidelity, times


if __name__ == "__main__":
    spins = 60
    alpha = 1
    protocol = "always_on_fast"
    chain = "open"
    mid_n = False
    end_n = True
    n_factor = 0  # 0 turns off n_factor (not 1)
    n_diff = 0
    save_tag = "optimum_gammas"

    save_tag += "_mid_n" if mid_n else ""
    save_tag += "_end_n" if end_n else ""
    save_tag += f"_n_over_{n_factor}" if n_factor else ""
    save_tag += f"_n_minus_{n_diff}" if n_diff else ""
    always_on = True if protocol == "always_on_fast" else False
    open_chain = True if chain == "open" else False

    orig_data = data_handling.read_data(protocol, chain, alpha, save_tag)
    index = orig_data["spins"].index(spins)
    analytical_gamma = orig_data["analytical_gamma"][index]
    optimum_gamma = orig_data["optimum_gamma"][index]
    evolution_time = orig_data["time"][index]
    analytical_fidelity = orig_data["naive_fidelity"][index]

    alphas = [
        alpha,
        1.1,
        1.2,
        1.3,
    ]
    gammas = [
        optimum_gamma / 2,
        optimum_gamma / 2,
        optimum_gamma / 2,
        optimum_gamma / 2,
    ]
    evolution_times = [
        orig_data["time"][index] * 15.2,
        orig_data["time"][index] * 26.7,
        orig_data["time"][index] * 44.3,
        orig_data["time"][index] * 73.4,
    ]

    data = []
    for i, alpha in enumerate(alphas):
        data.append(
            main(
                spins,
                alpha,
                gammas[i],
                evolution_times[i],
                open_chain,
                always_on,
                mid_n=mid_n,
                end_n=end_n,
                n_factor=n_factor,
                n_diff=n_diff,
            )
        )

    times_list = []
    fidelities_list = []
    for datum in data:
        times_list.append(datum["times"])
        fidelities_list.append(datum["fidelities"])

    plots.plot_fidelity_time_fast_slow_specific_n_2(
        times_list, fidelities_list, alphas, gammas
    )
