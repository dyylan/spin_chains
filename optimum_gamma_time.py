from spin_chains.quantum import chains
from spin_chains.quantum import hamiltonians
from spin_chains.quantum import states
from spin_chains.functions import numerics
from spin_chains.plots import plots
from spin_chains.data_analysis import data_handling

from numba import jit
import numpy as np
import GPyOpt

import scipy.special
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rc("text", usetex=True)
font = {"family": "serif", "size": 16, "serif": ["computer modern roman"]}
plt.rc("font", **font)


def main(
    spins_list,
    alpha,
    gamma_range,
    gamma_steps,
    open_chain,
    always_on,
    mid_n=False,
    end_n=False,
    n_factor=0,
    n_diff=0,
):
    data = {
        "spins": spins_list,
        "analytical_gamma": [],
        "start_gamma": [],
        "end_gamma": [],
        "optimum_gamma": [],
        "fidelity": [],
        "naive_fidelity": [],
        "time": [],
    }
    for spins in spins_list:
        approx_gamma, naive_fidelity = approximate_gamma(spins, open_chain, alpha)
        optimum_gamma, fidelity, time, start_gamma, end_gamma = optimise_gamma_BO(
            spins,
            alpha,
            naive_fidelity,
            approx_gamma,
            gamma_range,
            gamma_steps,
            open_chain,
            always_on,
            mid_n,
            end_n,
            n_factor,
            n_diff,
        )
        data["analytical_gamma"].append(approx_gamma)
        data["start_gamma"].append(start_gamma)
        data["end_gamma"].append(end_gamma)
        data["optimum_gamma"].append(optimum_gamma)
        data["fidelity"].append(fidelity)
        data["naive_fidelity"].append(naive_fidelity)
        data["time"].append(time)
        print(
            f"---- Computed\n\tfidelity = {fidelity} (naive fidelity = {naive_fidelity})"
            f"\n\ttime = {time}"
            f"\n\toptimum gamma = {optimum_gamma} (approx_gamma = {approx_gamma})"
            f"\n\tfor {spins} dimensions (up to {spins_list[-1]})"
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
        dt=0.01,
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
    spins,
    alpha,
    naive_fidelity,
    gamma,
    gamma_range,
    gamma_steps,
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
        time_steps = 10
        naive_time = (np.pi / 2) * np.sqrt(spins / np.sqrt(naive_fidelity))
        start_time = 85 * naive_time / 100
        end_time = 115 * naive_time / 100
        delta_time = (end_time - start_time) / time_steps
        evolution_times = [
            start_time + (step * delta_time) for step in range(time_steps + 1)
        ]
    gamma_delta = gamma_range / gamma_steps
    start_gamma = gamma - (gamma_range / 2)
    end_gamma = gamma - (gamma_range / 2) + (gamma_steps * gamma_delta)
    gammas = [start_gamma + (step * gamma_delta) for step in range(gamma_steps + 1)]
    optimum_gamma = gamma
    fidelity = 0
    time = 0
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


def optimise_gamma_BO(
    spins,
    alpha,
    naive_fidelity,
    gamma,
    gamma_range,
    gamma_steps,
    open_chain,
    always_on,
    mid_n,
    end_n,
    n_factor,
    n_diff,
):
    if always_on:
        evolution_times = [np.pi * np.sqrt(spins / 2)]
    else:
        time_steps = 10
        naive_time = (np.pi / 2) * np.sqrt(spins / np.sqrt(naive_fidelity))
        start_time = 85 * naive_time / 100
        end_time = 115 * naive_time / 100
        delta_time = (end_time - start_time) / time_steps
        evolution_times = [
            start_time + (step * delta_time) for step in range(time_steps + 1)
        ]
    gamma_delta = gamma_range / gamma_steps
    start_gamma = gamma - (gamma_range / 2)
    end_gamma = gamma - (gamma_range / 2) + (gamma_steps * gamma_delta)

    bounds = [
        {
            "name": "gamma",
            "type": "continuous",
            "domain": (start_gamma, end_gamma),
        }
    ]

    def cost(params):
        parameters = params.tolist()
        gamma_trial = parameters[0]
        _fidelity, _time = fidelity_time(
            spins,
            alpha,
            gamma_trial,
            evolution_times[0],
            open_chain,
            always_on,
            mid_n,
            end_n,
            n_factor,
            n_diff,
        )
        return 1 - _fidelity

    optimisation = GPyOpt.methods.BayesianOptimization(
        cost,
        domain=bounds,
        model_type="GP",
        acquisition_type="EI",
        normalize_Y=True,
        acquisition_weight=2,
        maximize=False,
    )

    max_iter = gamma_steps
    max_time = 300
    optimisation.run_optimization(max_iter, max_time, verbosity=True)

    optimum_gamma = optimisation.x_opt[0]
    fidelity, time = fidelity_time(
        spins,
        alpha,
        optimum_gamma,
        evolution_times[0],
        open_chain,
        always_on,
        mid_n,
        end_n,
        n_factor,
        n_diff,
    )
    return optimum_gamma, fidelity, time, start_gamma, end_gamma


def approximate_gamma(spins, open_chain, alpha, dt=0.01):
    open_chain_for_s = chains.Chain1dSubspaceLongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )
    eigenvectors, eigenvalues = hamiltonians.Hamiltonian.spectrum(
        open_chain_for_s.hamiltonian.H_subspace
    )
    s_1 = hamiltonians.Hamiltonian.s_parameter(
        eigenvalues, 1, open_chain=open_chain, eigenvectors=eigenvectors
    )
    s_2 = hamiltonians.Hamiltonian.s_parameter(
        eigenvalues, 2, open_chain=open_chain, eigenvectors=eigenvectors
    )

    return s_1, ((s_1 ** 4) / (s_2 ** 2))


if __name__ == "__main__":
    alpha = 0.3
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

    spins_list = [x for x in range(4, 22, 2)]
    data = main(
        spins_list,
        alpha,
        gamma_range=0.05,
        gamma_steps=80,
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
        spins_list,
        data["naive_fidelity"],
        data["fidelity"],
    )

    plots.plot_time(spins_list, data["time"], data["time"])
