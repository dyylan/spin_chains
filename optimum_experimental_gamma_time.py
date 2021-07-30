from spin_chains.quantum import chains
from spin_chains.quantum import hamiltonians
from spin_chains.quantum import states
from spin_chains.quantum import iontrap
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
    target_alpha,
    gamma_range_factor,
    gamma_steps,
    open_chain,
    always_on,
    mid_n=False,
    end_n=False,
    n_factor=0,
    n_diff=0,
    gamma_rescale=False,
    mu_lower_bound=np.ones(1) * 2 * np.pi * 6.0000005e6,
    use_existing_mu=None,
):
    data = {
        "spins": spins_list,
        "minimum_mu": [],
        "mu": [],
        "alpha": [],
        "z_trap_frequency": [],
        "analytical_gamma": [],
        "start_gamma": [],
        "end_gamma": [],
        "optimum_gamma": [],
        "fidelity": [],
        "naive_fidelity": [],
        "time": [],
        "switch_time": [],
    }
    for spins in spins_list:
        if use_existing_mu:
            data_orig = data_handling.read_data_spin(
                "experimental/always_on_fast",
                "open",
                target_alpha,
                use_existing_mu,
                spins,
            )
            omega, mu_lower_bound, mu, alpha, approx_gamma, naive_fidelity = (
                np.array(
                    [2 * np.pi * 6e6, 2 * np.pi * 5e6, data_orig["z_trap_frequency"]]
                ),
                np.ones(1) * data_orig["minimum_mu"],
                np.ones(1) * data_orig["mu"],
                data_orig["alpha"],
                data_orig["analytical_gamma"],
                data_orig["naive_fidelity"],
            )
            z_trap_frequency = data_orig["z_trap_frequency"]
            approx_gamma = 1 / approx_gamma
        else:
            ion_trap = iontrap.IonTrap(spins, use_optimal_omega=True)
            omega = ion_trap.omega
            z_trap_frequency = ion_trap.omega[2]
            mu_lower_bound = ion_trap.calculate_minimum_mu()
            # ion_trap.plot_spin_interactions()

            # mu_bounds = [array([37730527.76961342]), array([38076102.96150829])]
            mu_bounds = [
                mu_lower_bound,
                # np.ones(1) * 37725148.80194278,  # For high N only
                np.ones(1) * 2 * np.pi * 6.25e6,
            ]
            steps = gamma_steps

            mu, alpha, _ = ion_trap.update_mu_for_target_alpha(
                target_alpha, mu_bounds, steps
            )
            approx_gamma, naive_fidelity = approximate_gamma(ion_trap.Js)

        gamma_range = [
            approx_gamma * gamma_range_factor,
            approx_gamma / gamma_range_factor,
        ]
        (
            optimum_gamma,
            fidelity,
            time,
            switch_time,
            start_gamma,
            end_gamma,
        ) = optimise_gamma_BO(
            spins,
            mu,
            omega,
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
            gamma_rescale,
        )
        data["minimum_mu"].append(mu_lower_bound[0])
        data["mu"].append(mu[0])
        data["alpha"].append(alpha)
        data["z_trap_frequency"].append(z_trap_frequency)
        data["analytical_gamma"].append(1 / approx_gamma)
        data["start_gamma"].append(start_gamma)
        data["end_gamma"].append(end_gamma)
        data["optimum_gamma"].append(optimum_gamma)
        data["fidelity"].append(fidelity)
        data["naive_fidelity"].append(naive_fidelity)
        data["time"].append(time)
        data["switch_time"].append(switch_time)
        print(
            f"---- Computed\n\tfidelity = {fidelity} (naive fidelity = {naive_fidelity})"
            f"\n\ttime = {time}"
            f"\n\toptimum gamma = {optimum_gamma} (approx_gamma = {1/approx_gamma})"
            f"\n\tmu = {mu[0]} (minimum_mu = {mu_lower_bound[0]})"
            f"\n\tfor {spins} dimensions (up to {spins_list[-1]})"
        )
    return data


def quantum_communication(
    spins,
    marked_strength,
    switch_time,
    mu=1,
    omega=2 * np.pi * np.array([6e6, 5e6, 1e6]),
    open_chain=False,
    start_site=1,
    final_site=15,
    always_on=False,
    dt=0.01,
    gamma_rescale=False,
):
    final_state = states.SingleState(spins, final_site, single_subspace=True)
    init_state = states.SingleState(spins, start_site, single_subspace=True)

    chain = chains.Chain1dSubspaceLongRangeExp(spins=spins, dt=dt, mu=mu, omega=omega)

    chain.add_marked_site(start_site, marked_strength, gamma_rescale=gamma_rescale)
    chain.initialise(init_state)

    update_strength = 1 if gamma_rescale else marked_strength

    if always_on:
        if open_chain:
            switch_time = 1.5 * switch_time
        else:
            switch_time = 1.5 * switch_time
        chain.update_marked_site(final_site, update_strength)
    else:
        chain.time_evolution(time=switch_time)
        chain.update_marked_site(start_site, -update_strength)
        chain.update_marked_site(final_site, update_strength)

    times, psi_states = chain.time_evolution(time=switch_time, reset_state=False)

    return times, psi_states, final_state, chain


def fidelity_time(
    spins,
    mu,
    omega,
    gamma,
    evolution_time,
    open_chain,
    always_on,
    mid_n,
    end_n,
    n_factor,
    n_diff,
    gamma_rescale,
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

    dt = 0.1 if gamma_rescale else 1e-7
    times, psi_states, final_state, chain = quantum_communication(
        spins,
        gamma,
        evolution_time,
        mu=mu,
        omega=omega,
        open_chain=open_chain,
        start_site=1,
        final_site=final_site,
        always_on=always_on,
        dt=dt,
        gamma_rescale=gamma_rescale,
    )
    qst_fidelity = chain.overlaps_evolution(final_state.subspace_ket, psi_states)
    peaks, _ = find_peaks(qst_fidelity, height=(0.2, 1.05))
    if spins == None:
        plot_states(times, qst_fidelity)
    try:
        time = times[peaks[0]]
        fidelity = qst_fidelity[peaks[0]]
    except IndexError:
        fidelity = 0
        time = 0
    return fidelity, time


def optimise_gamma(
    spins,
    mu,
    omega,
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
    gamma_rescale=False,
):
    rescaled_gamma = 1 / gamma if not gamma_rescale else gamma
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
    if not gamma_rescale:
        evolution_times = [t * gamma for t in evolution_times]
    gamma_delta = gamma_range / gamma_steps
    start_gamma = rescaled_gamma - (gamma_range / 2)
    end_gamma = rescaled_gamma - (gamma_range / 2) + (gamma_steps * gamma_delta)
    gammas = [start_gamma + (step * gamma_delta) for step in range(gamma_steps + 1)]
    optimum_gamma = rescaled_gamma
    fidelity = 0
    time = 0
    for gam in gammas:
        for evolution_time in evolution_times:
            _fidelity, _time = fidelity_time(
                spins,
                mu,
                omega,
                gam,
                evolution_time,
                open_chain,
                always_on,
                mid_n,
                end_n,
                n_factor,
                n_diff,
                gamma_rescale,
            )
            if _fidelity > fidelity:
                fidelity = _fidelity
                time = _time
                optimum_gamma = gam
    return optimum_gamma, fidelity, time, start_gamma, end_gamma


def optimise_gamma_BO(
    spins,
    mu,
    omega,
    naive_fidelity,
    gamma,
    gamma_bounds,
    steps,
    open_chain,
    always_on,
    mid_n,
    end_n,
    n_factor,
    n_diff,
    gamma_rescale=False,
):
    gamma_bounds[0] = 1 / gamma_bounds[0] if not gamma_rescale else gamma
    gamma_bounds[1] = 1 / gamma_bounds[1] if not gamma_rescale else gamma
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
    if not gamma_rescale:
        evolution_times = [t * gamma for t in evolution_times]

    bounds = [
        {
            "name": "gamma",
            "type": "continuous",
            "domain": (gamma_bounds[0], gamma_bounds[1]),
        }
    ]

    def cost(params):
        parameters = params.tolist()
        gamma_trial = parameters[0]
        _fidelity, _ = fidelity_time(
            spins,
            mu,
            omega,
            gamma_trial,
            evolution_times[0],
            open_chain,
            always_on,
            mid_n,
            end_n,
            n_factor,
            n_diff,
            gamma_rescale,
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

    max_iter = steps
    max_time = 300
    optimisation.run_optimization(max_iter, max_time, verbosity=True)

    optimum_gamma = optimisation.x_opt[0]

    fidelity, time = fidelity_time(
        spins,
        mu,
        omega,
        optimum_gamma,
        evolution_times[0],
        open_chain,
        always_on,
        mid_n,
        end_n,
        n_factor,
        n_diff,
        gamma_rescale,
    )
    return (
        optimum_gamma,
        fidelity,
        time,
        evolution_times[0],
        gamma_bounds[0],
        gamma_bounds[1],
    )


def approximate_gamma(Js):
    eigenvectors, eigenvalues = hamiltonians.Hamiltonian.spectrum(Js)
    s_1 = hamiltonians.Hamiltonian.s_parameter(
        eigenvalues, 1, open_chain=True, eigenvectors=eigenvectors
    )
    s_2 = hamiltonians.Hamiltonian.s_parameter(
        eigenvalues, 2, open_chain=True, eigenvectors=eigenvectors
    )
    return s_1, ((s_1 ** 4) / (s_2 ** 2))


def plot_states(times, states):
    fig, ax = plt.subplots()
    ax.plot(times, states)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


if __name__ == "__main__":
    target_alpha = 0.4
    protocol = "always_on_fast"
    chain = "open"
    mid_n = True
    end_n = False
    n_factor = 0  # 0 turns off n_factor (not 1)
    n_diff = 0
    save_tag = "optimum_gammas"

    save_tag += "_mid_n" if mid_n else ""
    save_tag += "_end_n" if end_n else ""
    save_tag += f"_n_over_{n_factor}" if n_factor else ""
    save_tag += f"_n_minus_{n_diff}" if n_diff else ""
    # save_tag += f"_mu_stability=1kHz"
    save_tag += f"_mu_min"
    always_on = True if protocol == "always_on_fast" else False
    open_chain = True if chain == "open" else False

    # spins_list = [x for x in range(4, 124, 4)]
    # spins_list = [x for x in range(12, 40, 4)]
    spins_list = [x for x in range(4, 52, 2)]

    data = main(
        spins_list,
        target_alpha,
        gamma_range_factor=1.4,
        gamma_steps=60,
        open_chain=open_chain,
        always_on=always_on,
        mid_n=mid_n,
        end_n=end_n,
        n_factor=n_factor,
        n_diff=n_diff,
        gamma_rescale=False,
        # mu_lower_bound=np.ones(1) * 2 * np.pi * 6.001e6,\
        use_existing_mu="optimum_gammas_end_n_mu_min",
    )

    protocol = "experimental/" + protocol
    data_open = data_handling.update_data(
        protocol=protocol,
        chain=chain,
        alpha=target_alpha,
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
