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
    open_chain,
    always_on,
    noise,
    samples,
    mid_n=False,
    end_n=False,
    n_factor=0,
    n_diff=0,
):
    data = {
        "spins": orig_data["spins"],
        "analytical_gamma": orig_data["analytical_gamma"],
        "optimum_gamma": orig_data["optimum_gamma"],
        "fidelity": [],
        "noiseless_fidelity": orig_data["fidelity"],
        "naive_fidelity": orig_data["naive_fidelity"],
        "time": [],
        "noiseless_time": orig_data["time"],
    }
    for i, spins in enumerate(orig_data["spins"]):
        fidelity, time = optimise_gamma(
            spins,
            alpha,
            orig_data["time"][i],
            orig_data["optimum_gamma"][i],
            open_chain,
            always_on,
            noise,
            samples,
            mid_n,
            end_n,
            n_factor,
            n_diff,
        )
        data["fidelity"].append(fidelity)
        data["time"].append(time)
        print(
            f"---- Computed\n\tfidelity = {fidelity} (noiseless fidelity = {orig_data['fidelity'][i]})"
            f"\n\ttime = {time} (noiseless time = {orig_data['time'][i]})"
            f"\n\toptimum gamma = {orig_data['optimum_gamma'][i]}"
            f"\n\tfor {spins} dimensions (up to {orig_data['spins'][-1]})"
        )
    return data


def quantum_communication(
    spins,
    marked_strength,
    switch_time,
    alpha=1,
    open_chain=False,
    noise=0,
    samples=1,
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
        noise=noise,
        samples=samples,
    )

    chain.add_marked_site(start_site, marked_strength, gamma_rescale=True)
    chain.initialise(init_state, noisy_evolution=True)

    if always_on:
        if open_chain:
            switch_time = 4 * switch_time
        else:
            switch_time = 1.5 * switch_time
        chain.update_marked_site(final_site, 1)
    else:
        chain.noisy_time_evolution(time=switch_time)
        chain.update_marked_site(start_site, -1)
        chain.update_marked_site(final_site, 1)

    times, psi_states = chain.noisy_time_evolution(
        time=1.5 * switch_time, reset_state=False
    )

    return times, psi_states, final_state, chain


def fidelity_time(
    spins,
    alpha,
    gamma,
    evolution_time,
    open_chain,
    always_on,
    noise,
    samples,
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
        noise=noise,
        samples=samples,
        start_site=1,
        final_site=final_site,
        always_on=always_on,
    )
    qst_fidelity = chain.overlaps_noisy_evolution(final_state.subspace_ket, psi_states)
    peaks, _ = find_peaks(qst_fidelity, height=(0.2, 1.05))
    try:
        time = times[peaks[0]]
        fidelity = qst_fidelity[peaks[0]]
    except IndexError:
        fidelity = 0
        time = 0
    return fidelity, time


def optimise_gamma(
    spins,
    alpha,
    orig_time,
    gamma,
    open_chain,
    always_on,
    noise,
    samples,
    mid_n,
    end_n,
    n_factor,
    n_diff,
):
    if always_on:
        evolution_times = [np.sqrt(spins)]
    else:
        time_steps = 100
        start_time = 75 * orig_time / 100
        end_time = 110 * orig_time / 100
        delta_time = (end_time - start_time) / time_steps
        evolution_times = [
            start_time + (step * delta_time) for step in range(time_steps + 1)
        ]
    fidelity = 0
    time = 0
    for evolution_time in evolution_times:
        _fidelity, _time = fidelity_time(
            spins,
            alpha,
            gamma,
            evolution_time,
            open_chain,
            always_on,
            noise,
            samples,
            mid_n,
            end_n,
            n_factor,
            n_diff,
        )
        if _fidelity > fidelity:
            fidelity = _fidelity
            time = _time
    return fidelity, time


if __name__ == "__main__":
    alpha = 1
    protocol = "always_on_fast"
    chain = "open"
    noise = 0.005
    samples = 1000
    mid_n = False
    end_n = True
    n_factor = 0  # 0 turns of n_factor (not 1)
    n_diff = 0
    save_tag = "optimum_gammas"

    save_tag += "_mid_n" if mid_n else ""
    save_tag += "_end_n" if end_n else ""
    save_tag += f"_n_over_{n_factor}" if n_factor else ""
    save_tag += f"_n_minus_{n_diff}" if n_diff else ""
    orig_data = data_handling.read_data(protocol, chain, alpha, save_tag)

    save_tag += f"_noise={noise}" if noise else ""
    always_on = True if protocol == "always_on_fast" else False
    open_chain = True if chain == "open" else False

    spins_list = [x for x in range(12, 124, 4)]
    data = main(
        orig_data,
        alpha,
        open_chain=open_chain,
        always_on=always_on,
        noise=noise,
        samples=samples,
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
        spins_list,
        data["naive_fidelity"],
        data["fidelity"],
    )

    plots.plot_time(spins_list, data["time"], data["time"])
