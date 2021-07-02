from spin_chains.quantum.chains import (
    Chain1d,
    Chain1dSubspace,
    Chain1dSubspaceLongRange,
)
from spin_chains.quantum.hamiltonians import Hamiltonian
from spin_chains.quantum.states import (
    PeriodicState,
    FourierState,
    SubspaceFourierState,
    SingleExcitationState,
    SuperpositionState,
    SingleState,
)
from spin_chains.functions.numerics import (
    eigenvalue_numba,
    s_parameter,
    s_k_parameter,
    delta_n,
    delta_k,
    first_order_fidelity_correction,
)
from spin_chains.plots.plots import (
    plot_deltas,
    plot_always_on_fidelity,
    plot_always_on_time,
    plot_fidelity,
    plot_time,
    plot_time_comparisons,
    plot_always_on_time_comparison,
)
from spin_chains.functions.fits import power_fit
from spin_chains.data_analysis.data_handling import update_data, read_data
from numba import jit
import numpy as np

np.set_printoptions(threshold=np.inf)
import peakutils
import scipy.special
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def quantum_communication(
    spins,
    marked_strength,
    switch_time,
    alpha=1,
    open_chain=False,
    start_site=1,
    final_site=15,
    always_on=False,
    long_time=False,
    dt=0.01,
    gamma_ratio=1,
):
    final_state = SingleState(spins, final_site, single_subspace=True)
    init_state = SingleState(spins, start_site, single_subspace=True)

    chain = Chain1dSubspaceLongRange(
        spins=spins,
        dt=dt,
        alpha=alpha,
        open_chain=open_chain,
    )

    if always_on and long_time:
        chain.add_marked_site(start_site, marked_strength, gamma_rescale=True)
        if alpha == 0.8:
            switch_time = (spins ** 0.6) * switch_time * ((gamma_ratio / 2))
        elif alpha == 0.5:
            switch_time = (spins ** 0.5) * switch_time * ((gamma_ratio / 2))
        elif alpha == 0.3:
            switch_time = (spins ** 0.4) * switch_time * ((gamma_ratio / 2))
        else:
            switch_time = (spins ** 0.7) * switch_time * ((gamma_ratio / 2) ** 2)
    else:
        chain.add_marked_site(start_site, marked_strength, gamma_rescale=True)
    chain.initialise(init_state)

    if always_on:
        switch_time = 3.5 * switch_time
        chain.update_marked_site(final_site, 1)
    else:
        chain.time_evolution(time=switch_time)
        chain.update_marked_site(start_site, -1)
        chain.update_marked_site(final_site, 1)

    times, states = chain.time_evolution(time=1.5 * switch_time, reset_state=False)

    return times, states, final_state, chain


def fidelity_time(
    spins,
    alpha,
    open_chain=False,
    always_on=False,
    long_time=False,
    time_error=False,
    correction=False,
    dt=0.01,
    calculate_analytical=True,
    gamma_ratio=1,
    find_switch_time=False,
    plot=False,
):
    optimum_gammas = []
    gammas = []
    qst_times = []
    switch_times = []
    analytical_times = []
    fidelities = []
    naive_fidelities = []
    analytical_fidelities = []
    asymptotic_fidelities = []
    for spin in spins:
        start_site = 1
        final_site = spin // 2
        if open_chain:
            start_site = 1
            final_site = spin
            open_chain_for_s = Chain1dSubspaceLongRange(
                spins=spin, dt=dt, alpha=alpha, open_chain=True
            )
            eigenvectors, eigenvalues = Hamiltonian.spectrum(
                open_chain_for_s.hamiltonian.H_subspace
            )
            s_1 = Hamiltonian.s_parameter(
                eigenvalues, 1, open_chain=True, eigenvectors=eigenvectors
            )
            s_2 = Hamiltonian.s_parameter(
                eigenvalues, 2, open_chain=True, eigenvectors=eigenvectors
            )
        else:
            s_1 = s_parameter(1, alpha, spin)
            s_2 = s_parameter(2, alpha, spin)
        optimum_gamma = s_1
        optimum_gammas.append(optimum_gamma)
        delta_n = s_1 / np.sqrt(spin * s_2)
        naive_fidelity = s_1 ** 2 / s_2
        analytical_time = (np.pi / 2) * np.sqrt(spin / naive_fidelity)
        analytical_times.append(2 * analytical_time)
        time_noise = analytical_time / 5
        analytical_time = analytical_time * (1 + (0.05 * time_error))
        first_order_correction = (
            0
            if not correction
            else first_order_fidelity_correction(analytical_time, s_1, alpha, spin)
        )
        naive_fidelities.append(naive_fidelity ** 2 + first_order_correction)

        if calculate_analytical:
            init_state = SingleState(spin, start_site, single_subspace=True)
            marked_chain = Chain1dSubspaceLongRange(
                spins=spin, dt=dt, alpha=alpha, open_chain=open_chain
            )
            marked_chain.add_marked_site(start_site, s_1, gamma_rescale=True)
            marked_chain.initialise(init_state)
            analytical_fidelity, asymptotic_fidelity = compute_fidelity(
                marked_chain.hamiltonian,
                spin,
                analytical_time,
                analytical_time,
                delta_n,
                start_site,
                final_site,
            )
            analytical_fidelities.append(analytical_fidelity)
            asymptotic_fidelities.append(asymptotic_fidelity)

        if always_on and long_time:
            gamma = optimum_gamma / gamma_ratio
        else:
            # gamma = (1 - (spin * 0.00008)) * optimum_gamma * gamma_ratio
            gamma = optimum_gamma * gamma_ratio
        gammas.append(gamma)

        if find_switch_time:
            switch_time = find_optimum_time(
                spin,
                gamma,
                analytical_time * 3,
                alpha=alpha,
                open_chain=open_chain,
            )
            times, states, final_state, chain = quantum_communication(
                spin,
                gamma,
                switch_time=switch_time,
                alpha=alpha,
                open_chain=open_chain,
                start_site=start_site,
                final_site=final_site,
                always_on=always_on,
                long_time=long_time,
                dt=dt,
            )
            switch_times.append(switch_time)
        else:
            switch_time = analytical_time
            switch_times.append(analytical_time)
            times, states, final_state, chain = quantum_communication(
                spin,
                gamma,
                switch_time=switch_time,
                alpha=alpha,
                open_chain=open_chain,
                start_site=start_site,
                final_site=final_site,
                always_on=always_on,
                long_time=long_time,
                dt=dt,
                gamma_ratio=gamma_ratio,
            )
        qst_fidelity = chain.overlaps_evolution(final_state.subspace_ket, states)
        if plot:
            fig, ax = plt.subplots()
            ax.plot(times, qst_fidelity)
            ax.legend(
                [
                    f"n={spin} fidelity",
                ],
                loc=4,
            )
            ax.set(xlabel="Time")
            ax.grid()
            plt.show()
        # peaks = peakutils.indexes(qst_fidelity, thres=0.5)
        peaks, _ = find_peaks(qst_fidelity, height=(0.5, 1.05))
        if always_on and long_time:
            peak_xy = max([(qst_fidelity[x], x) for x in peaks])
            peak = peak_xy[1]
        else:
            peak = peaks[0]
            # peak_xy = max([(qst_fidelity[x], x) for x in peaks])
            # peak = peak_xy[1]
        time = times[peak]
        qst_times.append(time)
        fidelity = qst_fidelity[peak]
        fidelities.append(fidelity)
        print(
            f"Computed max probability of {fidelity} "
            f"(analytical fidelity = {'NA' if not calculate_analytical else analytical_fidelity}) at time {time} (switch time = {switch_time}, analytical time = {analytical_time}) "
            f"for {spin} dimensions (up to {spins[-1]}) "
            f"with gamma = {s_1}"
        )
    return (
        optimum_gammas,
        gammas,
        naive_fidelities,
        analytical_fidelities,
        asymptotic_fidelities,
        fidelities,
        qst_times,
        analytical_times,
        switch_times,
    )


def find_optimum_time(
    spins,
    marked_strength,
    time,
    alpha=1,
    open_chain=False,
    dt=0.01,
    plot=False,
):
    final_state = SingleState(spins, spins, single_subspace=True)
    init_state = SuperpositionState(
        spins=spins, subspace=1, period=1, offset=0, single_subspace=True
    )

    chain = Chain1dSubspaceLongRange(
        spins=spins,
        dt=dt,
        alpha=alpha,
        open_chain=open_chain,
    )

    chain.add_marked_site(spins, marked_strength, gamma_rescale=True)
    chain.initialise(init_state, subspace_evolution=True)

    times, states = chain.time_evolution(time=time)

    search_fidelity = chain.overlaps_evolution(final_state.subspace_ket, states)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(times, search_fidelity)
        ax.legend(
            [f"n={spins} fidelity"],
            loc=4,
        )
        ax.set(xlabel="Time")
        ax.grid()
        plt.show()
    peaks, _ = find_peaks(search_fidelity, height=(0.5, 1.05))
    peak = peaks[0]
    switch_time = times[peak]
    return switch_time


def compute_overlaps(hamiltonian, n, w=1, f=15):
    evectors, evalues = Hamiltonian.spectrum(hamiltonian.H_subspace)
    psi_n = evectors[:, 0]
    psi_n_minus_1 = evectors[:, 1]
    f_psi_n = psi_n[n - f]
    f_psi_n_minus_1 = psi_n_minus_1[n - f]
    w_psi_n = psi_n[n - w]
    w_psi_n_minus_1 = psi_n_minus_1[n - w]
    overlap1 = f_psi_n * np.conj(w_psi_n)
    overlap2 = f_psi_n_minus_1 * np.conj(w_psi_n_minus_1)
    return overlap1, overlap2


def compute_fidelity(hamiltonian, n, t1, t2, delta_n, w, f):
    overlap1, overlap2 = compute_overlaps(hamiltonian, n, w, f)
    term1 = (np.exp(-1j * delta_n * (t1 + t2)) * overlap1) + (
        np.exp(1j * delta_n * (t1 + t2)) * overlap2
    )
    term2 = -(
        n
        * delta_n
        * np.sin(delta_n * t1)
        * np.sin(delta_n * t2)
        * (overlap1 - overlap2)
    )
    amplitude = term1 + term2
    asymptotic_amplitude = term2
    fidelity = np.abs(amplitude) ** 2
    asymptotic_fidelity = np.abs(asymptotic_amplitude) ** 2
    return fidelity, asymptotic_fidelity


if __name__ == "__main__":
    alpha = 0.1

    alpha1 = 0.5
    alpha2 = 0.8
    spins = [40 * x for x in range(8, 26)]
    # spins = [400, 600, 800]

    (
        ao_slow_optimum_gammas,
        ao_slow_gammas,
        ao_slow_naive_fidelities,
        ao_slow_analytical_fidelities,
        ao_slow_asymptotic_fidelities,
        ao_slow_fidelities,
        ao_slow_times1,
        ao_slow_analytical_times,
        ao_slow_switch_times,
    ) = fidelity_time(
        spins,
        alpha1,
        open_chain=True,
        always_on=True,
        long_time=True,
        time_error=False,
        correction=False,
        dt=10,
        calculate_analytical=False,
        gamma_ratio=2,
        find_switch_time=False,
        plot=False,
    )

    (
        ao_slow_optimum_gammas,
        ao_slow_gammas,
        ao_slow_naive_fidelities,
        ao_slow_analytical_fidelities,
        ao_slow_asymptotic_fidelities,
        ao_slow_fidelities,
        ao_slow_times2,
        ao_slow_analytical_times,
        ao_slow_switch_times,
    ) = fidelity_time(
        spins,
        alpha2,
        open_chain=True,
        always_on=True,
        long_time=True,
        time_error=False,
        correction=False,
        dt=10,
        calculate_analytical=False,
        gamma_ratio=2,
        find_switch_time=False,
        plot=False,
    )

    # (
    #     ao_slow_3_optimum_gammas,
    #     ao_slow_3_gammas,
    #     ao_slow_3_naive_fidelities,
    #     ao_slow_3_analytical_fidelities,
    #     ao_slow_3_asymptotic_fidelities,
    #     ao_slow_3_fidelities,
    #     ao_slow_3_times,
    #     ao_slow_3_analytical_times,
    #     ao_slow_3_switch_times,
    # ) = fidelity_time(
    #     spins,
    #     alpha,
    #     open_chain=True,
    #     always_on=True,
    #     long_time=True,
    #     time_error=False,
    #     correction=False,
    #     dt=10,
    #     calculate_analytical=False,
    #     gamma_ratio=3,
    #     find_switch_time=False,
    #     plot=False,
    # )

    (
        ao_fast_optimum_gammas,
        ao_fast_gammas,
        ao_fast_naive_fidelities,
        ao_fast_analytical_fidelities,
        ao_fast_asymptotic_fidelities,
        ao_fast_fidelities,
        ao_fast_times1,
        ao_fast_analytical_times,
        ao_fast_switch_times,
    ) = fidelity_time(
        spins,
        alpha1,
        open_chain=True,
        always_on=True,
        long_time=False,
        time_error=False,
        correction=False,
        dt=0.1,
        calculate_analytical=False,
        gamma_ratio=1,
        find_switch_time=False,
        plot=False,
    )
    (
        ao_fast_optimum_gammas,
        ao_fast_gammas,
        ao_fast_naive_fidelities,
        ao_fast_analytical_fidelities,
        ao_fast_asymptotic_fidelities,
        ao_fast_fidelities,
        ao_fast_times2,
        ao_fast_analytical_times,
        ao_fast_switch_times,
    ) = fidelity_time(
        spins,
        alpha2,
        open_chain=True,
        always_on=True,
        long_time=False,
        time_error=False,
        correction=False,
        dt=0.1,
        calculate_analytical=False,
        gamma_ratio=1,
        find_switch_time=False,
        plot=False,
    )

    # ao_data_dict = {
    #     "spins": spins,
    #     "time": ao_times,
    #     "fidelity": ao_fidelities,
    #     "gamma": ao_gammas,
    #     "optimum_gamma": ao_optimum_gammas,
    # }
    # ao_data_open = update_data(
    #     protocol="always_on_slow",
    #     chain="open",
    #     alpha=alpha,
    #     save_tag="20210405",
    #     data_dict=ao_data_dict,
    # )

    # (
    #     rs_optimum_gammas,
    #     rs_gammas,
    #     rs_naive_fidelities,
    #     rs_analytical_fidelities,
    #     rs_asymptotic_fidelities,
    #     rs_fidelities,
    #     rs_times,
    #     rs_analytical_times,
    #     switch_times,
    # ) = fidelity_time(
    #     spins,
    #     alpha,
    #     open_chain=False,
    #     always_on=False,
    #     long_time=False,
    #     time_error=False,
    #     correction=False,
    #     dt=0.1,
    #     calculate_analytical=False,
    #     gamma_ratio=1,
    #     find_switch_time=True,
    # )
    # rs_data_dict = {
    #     "spins": spins,
    #     "time": rs_times,
    #     "fidelity": rs_fidelities,
    #     "gamma": rs_gammas,
    #     "optimum_gamma": rs_optimum_gammas,
    # }
    # rs_data_open = update_data(
    #     protocol="reverse_search",
    #     chain="open",
    #     alpha=alpha,
    #     save_tag="20210405",
    #     data_dict=rs_data_dict,
    # )

    # plot_fidelity(
    #     spins,
    #     ao_naive_fidelities,
    #     ao_fidelities,
    # )
    # plot_fidelity(
    #     spins,
    #     rs_naive_fidelities,
    #     rs_fidelities,
    # )

    # plot_time(spins, ao_analytical_times, ao_times)
    # plot_time(spins, rs_analytical_times, rs_times)

    # plot_always_on_time(alpha, spins, ao_fast_times, ao_slow_times, ao_slow_3_times)
    plot_always_on_time_comparison(
        alpha1,
        alpha2,
        spins,
        ao_fast_times1,
        ao_slow_times1,
        ao_fast_times2,
        ao_slow_times2,
    )
