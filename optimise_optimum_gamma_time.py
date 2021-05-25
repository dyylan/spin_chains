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
    orig_data,
    alpha,
    gamma_range,
    gamma_steps,
    time_range,
    time_steps,
    open_chain,
    always_on,
    mid_n=False,
    end_n=False,
    n_factor=0,
    n_diff=0,
):
    data = {
        "spins": orig_data["spins"],
        "analytical_gamma": orig_data["analytical_gamma"],
        "start_gamma": [],
        "end_gamma": [],
        "optimum_gamma": [],
        "fidelity": [],
        "naive_fidelity": orig_data["naive_fidelity"],
        "time": [],
    }
    for i, spins in enumerate(orig_data["spins"]):
        optimum_gamma, fidelity, time, start_gamma, end_gamma = optimise_gamma(
            orig_data["time"][i],
            orig_data["fidelity"][i],
            spins,
            alpha,
            orig_data["optimum_gamma"][i],
            gamma_range,
            gamma_steps,
            time_range,
            time_steps,
            open_chain,
            always_on,
            mid_n,
            end_n,
            n_factor,
            n_diff,
        )
        data["start_gamma"].append(start_gamma)
        data["end_gamma"].append(end_gamma)
        data["optimum_gamma"].append(optimum_gamma)
        data["fidelity"].append(fidelity)
        data["time"].append(time)
        print(
            f"---- Computed\n\tfidelity = {fidelity} (original fidelity = {orig_data['fidelity'][i]}, naive fidelity = {orig_data['naive_fidelity'][i]})"
            f"\n\ttime = {time} (original time = {orig_data['time'][i]})"
            f"\n\toptimum gamma = {optimum_gamma} (original gamma = {orig_data['optimum_gamma'][i]}, approx_gamma = {orig_data['analytical_gamma'][i]})"
            f"\n\tfor {spins} dimensions (up to {orig_data['spins'][-1]})"
        )
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
            switch_time = 4 * switch_time
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
    peaks, _ = find_peaks(qst_fidelity, height=(0.2, 1.05))
    try:
        time = times[peaks[0]]
        fidelity = qst_fidelity[peaks[0]]
    except IndexError:
        fidelity = 0
        time = 0
    return fidelity, time


def optimise_gamma(
    orig_time,
    orig_fidelity,
    spins,
    alpha,
    gamma,
    gamma_range,
    gamma_steps,
    time_range,
    time_steps,
    open_chain,
    always_on,
    mid_n,
    end_n,
    n_factor,
    n_diff,
):
    if always_on:
        evolution_times = [np.sqrt(spins)]
    else:
        start_time = (orig_time / 2) - (time_range / 2)
        end_time = (orig_time / 2) + (time_range / 2)
        delta_time = (end_time - start_time) / time_steps
        evolution_times = [
            start_time + (step * delta_time) for step in range(time_steps + 1)
        ]
    gamma_delta = gamma_range / gamma_steps
    start_gamma = gamma - gamma_range / 2
    end_gamma = gamma - (gamma_range / 2) + (gamma_steps * gamma_delta)
    gammas = [start_gamma + (step * gamma_delta) for step in range(gamma_steps + 1)]
    optimum_gamma = gamma
    fidelity = orig_fidelity
    time = orig_time
    for gam in gammas:
        for evolution_time in evolution_times:
            _fidelity, _time = fidelity_time(
                spins,
                alpha,
                gam,
                evolution_time,
                open_chain,
                always_on,
                mid_n,
                end_n,
                n_factor,
                n_diff,
            )
            if _fidelity > fidelity:
                fidelity = _fidelity
                time = _time
                optimum_gamma = gam
    return optimum_gamma, fidelity, time, start_gamma, end_gamma


if __name__ == "__main__":
    alpha = 1
    protocol = "reverse_search"
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

    data = main(
        orig_data,
        alpha,
        gamma_range=0.02,
        gamma_steps=20,
        time_range=2,
        time_steps=80,
        open_chain=open_chain,
        always_on=always_on,
        mid_n=mid_n,
        end_n=end_n,
        n_factor=n_factor,
        n_diff=n_diff,
    )

    data_open = data_handling.update_data(
        protocol=protocol,
        chain=chain,
        alpha=alpha,
        save_tag=save_tag,
        data_dict=data,
        replace=True,
    )

    plots.plot_fidelity(
        data["spins"],
        data["naive_fidelity"],
        data["fidelity"],
    )

    plots.plot_time(data["spins"], data["time"], data["time"])
