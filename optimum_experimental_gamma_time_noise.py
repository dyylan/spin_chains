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
    orig_data,
    alpha,
    t2,
    samples,
    open_chain,
    always_on,
    mid_n=False,
    end_n=False,
    n_factor=0,
    n_diff=0,
    gamma_rescale=False,
    broken_spin=None,
    broken_spin_factor=1,
):
    data = {
        "spins": orig_data["spins"],
        "mu": orig_data["mu"],
        "alpha": orig_data["alpha"],
        "z_trap_frequency": orig_data["z_trap_frequency"],
        "analytical_gamma": orig_data["analytical_gamma"],
        "optimum_gamma": orig_data["optimum_gamma"],
        "fidelity": [],
        "fidelity_lower_error": [],
        "fidelity_upper_error": [],
        "noiseless_fidelity": orig_data["fidelity"],
        "naive_fidelity": orig_data["naive_fidelity"],
        "time": [],
        "noiseless_time": orig_data["time"],
        "switch_time": orig_data["switch_time"],
    }

    for i, spins in enumerate(orig_data["spins"]):
        fidelity, fidelity_lower_error, fidelity_upper_error, time = optimise_gamma_BO(
            spins,
            np.ones(1) * orig_data["mu"][i],
            np.array(
                [2 * np.pi * 6e6, 2 * np.pi * 5e6, orig_data["z_trap_frequency"][i]]
            ),
            orig_data["optimum_gamma"][i],
            orig_data["time"][i],
            t2,
            samples,
            open_chain,
            always_on,
            mid_n,
            end_n,
            n_factor,
            n_diff,
            gamma_rescale,
            broken_spin,
            broken_spin_factor,
        )
        data["fidelity"].append(fidelity)
        data["fidelity_lower_error"].append(fidelity_lower_error)
        data["fidelity_upper_error"].append(fidelity_upper_error)
        data["time"].append(time)
        print(
            f"---- Computed\n\tfidelity = {fidelity} (noiseless fidelity = {orig_data['fidelity'][i]})"
            f"\n\tand time = {time} with t2 = {t2}"
            f"\n\tfor {spins} dimensions (up to {spins_list[-1]})"
        )
    return data


def quantum_communication(
    spins,
    marked_strength,
    switch_time,
    mu=1,
    omega=2 * np.pi * np.array([6e6, 5e6, 1e6]),
    t2=0,
    samples=1,
    open_chain=False,
    start_site=1,
    final_site=15,
    always_on=False,
    dt=0.01,
    gamma_rescale=False,
    broken_spin=None,
    broken_spin_factor=1,
):
    final_state = states.SingleState(spins, final_site, single_subspace=True)
    init_state = states.SingleState(spins, start_site, single_subspace=True)

    chain = chains.Chain1dSubspaceLongRangeExp(
        spins=spins, dt=dt, mu=mu, omega=omega, t2=t2, samples=samples
    )

    chain.add_marked_site(start_site, marked_strength, gamma_rescale=gamma_rescale)
    chain.initialise(init_state, noisy_evolution=True)

    if broken_spin == "n_over_2":
        chain.add_broken_spin(spins // 2, t2=t2 / broken_spin_factor)
    update_strength = 1 if gamma_rescale else marked_strength

    if always_on:
        if open_chain:
            switch_time = 1.5 * switch_time
        else:
            switch_time = 1.5 * switch_time
        chain.update_marked_site(final_site, update_strength)
    else:
        chain.noisy_time_evolution(time=switch_time)
        chain.update_marked_site(start_site, -update_strength)
        chain.update_marked_site(final_site, update_strength)

    times, psi_states = chain.noisy_time_evolution(time=switch_time, reset_state=False)

    return times, psi_states, final_state, chain


def fidelity_time(
    spins,
    mu,
    omega,
    gamma,
    evolution_time,
    t2,
    samples,
    open_chain,
    always_on,
    mid_n,
    end_n,
    n_factor,
    n_diff,
    gamma_rescale,
    broken_spin,
    broken_spin_factor,
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

    dt = 0.1 if gamma_rescale else 1e-6
    times, psi_states, final_state, chain = quantum_communication(
        spins,
        gamma,
        evolution_time,
        mu=mu,
        omega=omega,
        t2=t2,
        samples=samples,
        open_chain=open_chain,
        start_site=1,
        final_site=final_site,
        always_on=always_on,
        dt=dt,
        gamma_rescale=gamma_rescale,
        broken_spin=broken_spin,
        broken_spin_factor=broken_spin_factor,
    )
    qst_fidelity, qst_error = chain.overlaps_noisy_evolution(
        final_state.subspace_ket, psi_states, norm=True, std_dev=True
    )
    peaks, _ = find_peaks(qst_fidelity, height=(0.2, 1.05))

    qst_fidelity_upper_error = qst_fidelity + qst_error
    qst_fidelity_lower_error = qst_fidelity - qst_error
    if spins in []:
        plot_states(
            times, qst_fidelity, qst_fidelity_lower_error, qst_fidelity_upper_error
        )
    try:
        time = times[peaks[0]]
        fidelity = qst_fidelity[peaks[0]]
        fidelity_lower_error = qst_fidelity_lower_error[peaks[0]]
        fidelity_upper_error = qst_fidelity_upper_error[peaks[0]]
    except IndexError:
        print(f"Fidelity is too low, no peak found for {spins} spins")
        fidelity = 0
        time = 0

    return fidelity, fidelity_lower_error, fidelity_upper_error, time


def optimise_gamma_BO(
    spins,
    mu,
    omega,
    gamma,
    evolution_time,
    t2,
    samples,
    open_chain,
    always_on,
    mid_n,
    end_n,
    n_factor,
    n_diff,
    gamma_rescale=False,
    broken_spin=None,
    broken_spin_factor=1,
):
    fidelity, fidelity_lower_error, fidelity_upper_error, time = fidelity_time(
        spins,
        mu,
        omega,
        gamma,
        evolution_time,
        t2,
        samples,
        open_chain,
        always_on,
        mid_n,
        end_n,
        n_factor,
        n_diff,
        gamma_rescale,
        broken_spin,
        broken_spin_factor,
    )
    return fidelity, fidelity_lower_error, fidelity_upper_error, time


def plot_states(times, states, lower_error, upper_error):
    fig, ax = plt.subplots()
    ax.plot(times, states)
    ax.fill_between(times, lower_error, upper_error, facecolor="green", alpha=0.5)
    ax.set(xlabel="Time~$(s/\hbar)$")
    ax.grid()
    plt.show()


if __name__ == "__main__":
    target_alpha = 0.2
    protocol = "always_on_fast"
    chain = "open"
    t2 = 1e-2
    samples = 5000
    mid_n = False
    end_n = True
    n_factor = 0  # 0 turns off n_factor (not 1)
    n_diff = 0
    broken_spin = "n_over_2"
    broken_spin_factor = 30
    save_tag = "optimum_gammas"

    save_tag += "_mid_n" if mid_n else ""
    save_tag += "_end_n" if end_n else ""
    save_tag += f"_n_over_{n_factor}" if n_factor else ""
    save_tag += f"_n_minus_{n_diff}" if n_diff else ""
    always_on = True if protocol == "always_on_fast" else False
    open_chain = True if chain == "open" else False
    protocol = "experimental/" + protocol
    orig_data = data_handling.read_data(protocol, chain, target_alpha, save_tag)
    save_tag += f"_t2={t2}" if t2 else ""
    save_tag += f"_samples={samples}" if t2 else ""
    save_tag += f"_broken_spin={broken_spin}" if broken_spin else ""
    save_tag += f"_broken_spin_factor={broken_spin_factor}" if broken_spin else ""

    spins_list = [x for x in range(12, 124, 4)]
    # spins_list = [x for x in range(12, 40, 4)]
    spins_list = [x for x in range(18, 54, 2)]

    data = main(
        orig_data,
        target_alpha,
        open_chain=open_chain,
        always_on=always_on,
        t2=t2,
        samples=samples,
        mid_n=mid_n,
        end_n=end_n,
        n_factor=n_factor,
        n_diff=n_diff,
        gamma_rescale=False,
        broken_spin=broken_spin,
        broken_spin_factor=broken_spin_factor,
    )

    data_open = data_handling.update_data(
        protocol=protocol,
        chain=chain,
        alpha=target_alpha,
        save_tag=save_tag,
        data_dict=data,
        replace=True,
    )

    plots.plot_noisy_fidelity(
        data["spins"],
        data["fidelity"],
        data["fidelity_lower_error"],
        data["fidelity_upper_error"],
    )

    plots.plot_time(data["spins"], data["time"], data["time"])
