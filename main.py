from spin_chains.quantum.chains import (
    Chain1d,
    Chain1dSubspace,
    Chain1dSubspace2particles,
    Chain1dSubspaceNparticles,
    Chain1dSubspace2LongRange,
    Chain1dSubspaceLongRange,
    Chain1dLongRange,
    Chain1dSubspaceLongRangeExp,
)
from spin_chains.quantum.states import (
    PeriodicState,
    FourierState,
    SubspaceFourierState,
    SingleExcitationState,
    SuperpositionState,
    SingleState,
    SpecifiedState,
    TwoParticleHubbbardState,
    NParticleHubbbardState,
)
from spin_chains.quantum.hamiltonians import Hamiltonian

# from spin_chains.functions.couplings_calculations import mu_for_specific_alpha_BO

import numpy as np
import itertools

np.set_printoptions(threshold=np.inf)
import scipy.special
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


fontsize_axis = 18
fontsize_legend = 14
fontsize_ticks = 14


# plt.rc("text", usetex=True)
# font = {"family": "serif", "size": 16, "serif": ["computer modern roman"]}
# plt.rc("font", **font)


def state_hist_with_time(states, times):
    pass


def state_hist(state):
    states_x = np.arange(len(state.ket))
    states_y = state.ket
    fig, ax = plt.subplots()
    ax.bar(states_y, states_x)
    ax.set(xlabel="$$")
    ax.grid()
    # plt.savefig()
    plt.show()


def state_overlaps(spins, state, time):
    init_state = SingleState(spins, state)
    overlap_state = SingleState(spins, state)
    init_state.state_barplot()
    overlap_state.state_barplot()
    chain = Chain1d(spins=spins, dt=0.01, jx=1, jy=1, jz=0, h=0)
    chain.initialise(init_state)
    times, states = chain.time_evolution(time=time)
    # print(chain.hamiltonian.H)
    overlaps = chain.overlaps_evolution(overlap_state.ket, states)
    fig, ax = plt.subplots()
    for y in [overlaps]:
        ax.plot(times, y)
    # ax.legend([f'k = {k}' for k in range(2**spins)])
    ax.set(xlabel="$time~(s/\hbar)$")
    ax.grid()
    plt.show()


def fourier_state_overlaps_bm(spins):
    # init_state = SingleExcitationState(spins, 1)
    n = 2 ** spins
    init_state = SingleState(spins, 4)
    # init_state = PeriodicState(spins=spins,period=period)
    # init_state = FourierState(spins, 1)
    fourier_states = [FourierState(spins, k) for k in range(2 ** spins)]

    t = 1
    states = []
    times = []
    unitary = (1 / np.sqrt(n)) * np.array(
        [[np.exp(1j * 2 * np.pi * x * k / n) for x in range(n)] for k in range(n)]
    )
    ket = init_state
    state_ket = np.matmul(unitary, ket)
    states.append(state_ket)
    times.append(t)
    ket = state_ket

    for k, fourier_state in enumerate(fourier_states):
        overlaps.append(chain.overlaps_evolution(fourier_state.ket, states))
        print(f"Computed overlaps for fourier state k = {k}")
    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f'k = {k}' for k in range(2**spins)])
    ax.set(xlabel="$time~(s/\hbar)$")
    ax.grid()
    plt.show()


def fourier_state_overlaps(spins, period, time, jx, jy, jz, hz):
    # init_state = SingleExcitationState(spins, 1)
    init_state = SingleState(spins, 4)
    # init_state = PeriodicState(spins=spins,period=period)
    # init_state = FourierState(spins, 1)
    fourier_states = [FourierState(spins, k) for k in range(2 ** spins)]

    chain = Chain1d(spins=spins, dt=0.001, jx=jx, jy=jy, jz=jz)
    for spin, h in enumerate(hz):
        chain.add_marked_site(spin + 1, h)
    chain.initialise(init_state, subspace_evolution=False)
    times, states = chain.time_evolution(time=time)
    # print(chain.hamiltonian.H)
    overlaps = []
    for k, fourier_state in enumerate(fourier_states):
        overlaps.append(chain.overlaps_evolution(fourier_state.ket, states))
        print(f"Computed overlaps for fourier state k = {k}")
    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f'k = {k}' for k in range(2**spins)])
    ax.set(xlabel="$time~(s/\hbar)$")
    ax.grid()
    plt.show()


def subspace_fourier_state_overlaps(spins, subspace, period, time):
    init_state = SuperpositionState(spins=spins, subspace=subspace, period=period)
    # init_state = PeriodicState(spins=spins, period=period)
    # init_state = SubspaceFourierState(spins, 0, subspace)
    # init_state = SingleExcitationState(spins, excited_spin=1)
    subspace_size = int(scipy.special.comb(spins, subspace))
    fourier_states = [
        SubspaceFourierState(spins, k, subspace) for k in range(subspace_size)
    ]

    chain = Chain1d(spins=spins, dt=0.001, jx=1, jy=1, jz=0, hz=0)
    # for spin in range(1,spins+1):
    #     chain.add_marked_site(spin, spin)
    chain.add_marked_site(1, 0.5)
    chain.add_marked_site(3, 1)
    chain.add_marked_site(5, 1.5)
    chain.initialise(init_state)
    times, states = chain.time_evolution(time=time)
    print(chain.hamiltonian.H_subspace)
    # print(chain.hamiltonian.U_subspace)
    overlaps = []
    for k, fourier_state in enumerate(fourier_states):
        overlaps.append(chain.overlaps_evolution(fourier_state.subspace_ket, states))
        print(f"Computed overlaps for subspace fourier state k = {k}")
    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"k = {k}" for k in range(subspace_size)])
    ax.set(xlabel="$time~(s/\hbar)$")
    ax.grid()
    plt.show()


def test(spins, subspace, period, time):
    # init_state = SuperpositionState(spins=spins, subspace=subspace, period=period)
    # init_state = PeriodicState(spins=spins, period=period)
    # init_state = SubspaceFourierState(spins, 0, subspace)
    init_state = SingleExcitationState(spins, excited_spin=1)
    subspace_size = int(scipy.special.comb(spins, subspace))
    fourier_states = [
        SubspaceFourierState(spins, k, subspace) for k in range(subspace_size)
    ]
    exitation_states = [
        SingleExcitationState(spins, k + 1) for k in range(subspace_size)
    ]
    # jx = [1,1,1,1.1]
    jx = [1, 2, 3, 4]
    # jy = [1,1,1,1.1]
    jy = [1, 1, 1, 1]
    chain = Chain1d(spins=spins, dt=0.001, jx=jx, jy=jy, jz=0, h=0)
    # chain.add_marked_site(4, 1) # + np.random.normal(0,0.5))
    # chain.add_marked_site(2, 1)
    # chain.add_marked_site(1, 1)
    # chain.add_marked_site(1, -2)
    chain.initialise(init_state)
    times, states = chain.time_evolution(time=time)
    # print(states)
    print(chain.hamiltonian.H_subspace)
    overlaps = []
    for k, state in enumerate(fourier_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for subspace fourier state k = {k}")
    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"k = {k}" for k in range(subspace_size)])
    ax.set(xlabel="$time~(s/\hbar)$")
    ax.grid()
    plt.show()


def test2(spins, subspace, time):
    init_state = SingleExcitationState(spins=spins, excited_spin=1)
    # init_state = PeriodicState(spins=spins, period=period)
    # init_state = SubspaceFourierState(spins, 0, subspace)
    # init_state = SingleExcitationState(spins, excited_spin=1)
    final_state = SingleExcitationState(spins=spins, excited_spin=2)
    # subspace_size = int(scipy.special.comb(spins, subspace))
    # fourier_states = [SubspaceFourierState(spins, k, subspace) for k in range(subspace_size)]
    # exitation_states = [SingleExcitationState(spins, k+1) for k in range(subspace_size)]

    chain = Chain1d(spins=spins, dt=0.01, jx=0.5, jy=0.5, jz=0, hz=0)

    print(chain.hamiltonian.H)
    chain.initialise(init_state)
    print(chain.hamiltonian.H_subspace)
    times, states = chain.time_evolution(time=time)
    # print(states)
    # print(chain.state.subspace)
    # print(chain.subspace_evolution)
    overlaps = []
    overlaps.append(chain.overlaps_evolution(final_state.subspace_ket, states))

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend(["final state"])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def test3(spins, subspace, time):
    init_state = SingleExcitationState(spins=spins, excited_spin=1)
    # init_state = PeriodicState(spins=spins, period=period)
    # init_state = SubspaceFourierState(spins, 0, subspace)
    # init_state = SingleExcitationState(spins, excited_spin=1)
    final_state = SingleExcitationState(spins=spins, excited_spin=2)
    # subspace_size = int(scipy.special.comb(spins, subspace))
    # fourier_states = [SubspaceFourierState(spins, k, subspace) for k in range(subspace_size)]
    # exitation_states = [SingleExcitationState(spins, k+1) for k in range(subspace_size)]

    chain = Chain1dLongRange(spins=spins, dt=0.01, alpha=1, open_chain=True)

    print(chain.hamiltonian.H)
    chain.initialise(init_state)
    print(chain.hamiltonian.H_subspace)
    times, states = chain.time_evolution(time=time)
    # print(states)
    # print(chain.state.subspace)
    # print(chain.subspace_evolution)
    overlaps = []
    overlaps.append(chain.overlaps_evolution(final_state.subspace_ket, states))

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend(["final state"])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def test4(spins, subspace, time):
    init_state = SingleExcitationState(spins=spins, excited_spin=1)
    # init_state = PeriodicState(spins=spins, period=period)
    # init_state = SubspaceFourierState(spins, 0, subspace)
    # init_state = SingleExcitationState(spins, excited_spin=1)
    final_state = SingleExcitationState(spins=spins, excited_spin=2)
    # subspace_size = int(scipy.special.comb(spins, subspace))
    # fourier_states = [SubspaceFourierState(spins, k, subspace) for k in range(subspace_size)]
    # exitation_states = [SingleExcitationState(spins, k+1) for k in range(subspace_size)]

    chain = Chain1dSubspaceLongRange(spins, dt=0.01, alpha=1, open_chain=True)

    chain.add_marked_site(1, -1, gamma_rescale=False)
    chain.add_marked_site(2, -1, gamma_rescale=False)

    print(chain.hamiltonian.H)
    chain.initialise(init_state)
    print(chain.hamiltonian.H_subspace)
    times, states = chain.time_evolution(time=time)
    # print(states)
    # print(chain.state.subspace)
    # print(chain.subspace_evolution)
    overlaps = []
    overlaps.append(chain.overlaps_evolution(final_state.subspace_ket, states))

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend(["final state"])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def testXYchain(spins, subspace, period, time):
    init_state = SuperpositionState(
        spins=spins, subspace=subspace, period=period, offset=0
    )
    # init_state = PeriodicState(spins=spins, period=period)
    # init_state = SubspaceFourierState(spins, 0, subspace)
    # init_state = SingleExcitationState(spins, excited_spin=1)
    # final_state = SuperpositionState(spins=spins, subspace=subspace, period=period)
    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]
    # subspace_size = int(scipy.special.comb(spins, subspace))
    # fourier_states = [SubspaceFourierState(spins, k, subspace) for k in range(subspace_size)]
    # exitation_states = [SingleExcitationState(spins, k+1) for k in range(subspace_size)]
    # print(init_state.subspace_ket)

    chain = Chain1dSubspace(spins=spins, dt=0.01, js=1)

    chain.initialise(init_state)
    times, states = chain.time_evolution(time=time)

    # print(chain.state.subspace)
    # print(chain.subspace_evolution)
    # print(final_state.subspace_ket)
    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots(figsize=[8, 6])
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"j = {j+1}" for j in range(spins)])
    line1 = mlines.Line2D(
        [],
        [],
        color="tab:green",
        linestyle="solid",
        label="$|1\\rangle, |5\\rangle, |9\\rangle, |13\\rangle$",
    )
    line2 = mlines.Line2D(
        [],
        [],
        color="tab:purple",
        linestyle="solid",
        label="$|3\\rangle, |7\\rangle, |11\\rangle, |15\\rangle$",
    )
    line3 = mlines.Line2D(
        [],
        [],
        color="tab:brown",
        linestyle="solid",
        label="$|2\\rangle, |4\\rangle, |6\\rangle, |8\\rangle,$"
        "\n$|10\\rangle, |12\\rangle, |14\\rangle, |16\\rangle$",
    )
    legend1 = plt.legend(
        handles=[line1, line2, line3],
        bbox_to_anchor=(1, 1),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend1)
    ax.set_xlabel("Time $(s/\hbar)$", fontsize=fontsize_axis)
    plt.tight_layout()
    ax.grid()
    fig.savefig("fig", bbox_extra_artists=(legend1,), bbox_inches="tight")
    plt.show()


def longrangeXYchain_spatial_search(
    spins, marked_strength, period, time, dt=0.1, alpha=1, open_chain=False
):
    init_state = SuperpositionState(
        spins=spins, subspace=1, period=period, offset=0, single_subspace=True
    )
    # init_state = SingleState(spins, 1, single_subspace=True)

    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    chain = Chain1dSubspaceLongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )

    chain.add_marked_site(1, marked_strength, gamma_rescale=True)

    chain.initialise(init_state, subspace_evolution=True)
    times, states = chain.time_evolution(time=time)

    # print(chain.hamiltonian.H_subspace)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"j = {j+1}" for j in range(spins)])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def longrangeXYchain_spatial_search_noisy(
    spins,
    marked_strength,
    period,
    time,
    dt=0.1,
    alpha=1,
    open_chain=False,
    noise=0,
    samples=1,
):
    init_state = SuperpositionState(
        spins=spins, subspace=1, period=period, offset=0, single_subspace=True
    )
    # init_state = SingleState(spins, 1, single_subspace=True)

    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    chain = Chain1dSubspaceLongRange(
        spins=spins,
        dt=dt,
        alpha=alpha,
        open_chain=open_chain,
        noise=noise,
        samples=samples,
    )

    chain.add_marked_site(1, marked_strength, gamma_rescale=True)

    chain.initialise(init_state, subspace_evolution=True, noisy_evolution=True)
    times, states = chain.noisy_time_evolution(time=time)
    # times, states = chain.time_evolution(time=time)

    print(chain.hamiltonian.H_subspace)
    print(chain.hamiltonian.H_noise_subspace)
    print(chain.hamiltonian.noise_arrays)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_noisy_evolution(state.subspace_ket, states))
        # overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"j = {j+1}" for j in range(spins)])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def longrangeXYchain_spatial_search_noise_comparisons(
    spins,
    marked_strength,
    period,
    time,
    dt=0.1,
    alpha=1,
    open_chain=True,
    samples=10,
):
    noises = [0, 0.01, 0.02, 0.05, 0.1]

    overlaps = []

    for noise in noises:
        init_state = SuperpositionState(
            spins=spins, subspace=1, period=period, offset=0, single_subspace=True
        )
        # init_state = SingleState(spins, 1, single_subspace=True)

        final_state = SingleState(spins, spins, single_subspace=True)

        chain = Chain1dSubspaceLongRange(
            spins=spins,
            dt=dt,
            alpha=alpha,
            open_chain=open_chain,
            noise=noise,
            samples=samples,
        )

        chain.add_marked_site(spins, marked_strength, gamma_rescale=True)

        chain.initialise(init_state, subspace_evolution=True, noisy_evolution=True)
        times, states = chain.noisy_time_evolution(time=time)

        overlaps.append(
            chain.overlaps_noisy_evolution(final_state.subspace_ket, states)
        )
        print(f"Computed overlaps for noise={noise}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"noise = {noise}" for noise in noises])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def longrangeXYchain_spatial_search_multi_sites(
    spins, marked_strength, period, time, dt=0.1, alpha=1, open_chain=False
):
    init_state = SuperpositionState(
        spins=spins, subspace=1, period=period, offset=0, single_subspace=True
    )
    # init_state = SingleState(spins, 1, single_subspace=True)

    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    chain = Chain1dSubspaceLongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )

    chain.add_marked_site(1, marked_strength, gamma_rescale=True)
    chain.add_marked_site(spins // 2, 1, gamma_rescale=False)

    chain.initialise(init_state, subspace_evolution=True)
    times, states = chain.time_evolution(time=time)

    # print(chain.hamiltonian.H_subspace)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"j = {j+1}" for j in range(spins)])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def longrangeXYchain_spatial_search_subspace_2(
    spins, marked_strength, period, time, dt=0.1, alpha=1, open_chain=False, subspace=2
):
    subspace_size = int(scipy.special.comb(spins, subspace))
    # init_state = SuperpositionState(
    #     spins=spins, subspace=subspace, period=period, offset=0
    # )
    init_state = SingleState(spins, subspace_size - 7, excitations=2)
    # init_state = SubspaceFourierState(spins, 0, subspace)
    # init_state = SingleExcitationState(spins, excited_spin=1)
    # final_state = SuperpositionState(spins=spins, subspace=subspace, period=period)

    single_states = [
        SingleState(subspace_size, subspace_size - j, single_subspace=True)
        for j in range(subspace_size)
    ]
    # subspace_size = int(scipy.special.comb(spins, subspace))
    # fourier_states = [SubspaceFourierState(spins, k, subspace) for k in range(subspace_size)]
    # exitation_states = [SingleExcitationState(spins, k+1) for k in range(subspace_size)]

    chain = Chain1dLongRange(spins=spins, dt=dt, alpha=alpha, open_chain=open_chain)

    chain.add_marked_site(1, marked_strength, gamma_rescale=True)
    chain.add_marked_site(spins, 1, gamma_rescale=False)

    # chain.add_marked_site(7, 1, gamma_rescale=False)
    # chain.add_marked_site(8, 1, gamma_rescale=False)
    # chain.add_marked_site(11, 8)
    # chain.add_marked_site(16, 8)
    chain.initialise(init_state, subspace_evolution=True)

    print(chain.hamiltonian.H_subspace)

    times, states = chain.time_evolution(time=time)

    overlaps = []
    state_labels = chain.state.state_labels_subspace()
    print(state_labels)
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1} ({state_labels[j]})")

    # chain.state.state_barplot()
    # print(chain.hamiltonian.H_subspace)

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"{sta}" for sta in chain.state.state_labels_subspace()])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def longrangeXYchain(spins, subspace, period, time):
    init_state = SuperpositionState(
        spins=spins, subspace=subspace, period=period, offset=0
    )
    # init_state = PeriodicState(spins=spins, period=period)
    # init_state = SubspaceFourierState(spins, 0, subspace)
    # init_state = SingleExcitationState(spins, excited_spin=1)
    # final_state = SuperpositionState(spins=spins, subspace=subspace, period=period)
    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]
    # subspace_size = int(scipy.special.comb(spins, subspace))
    # fourier_states = [SubspaceFourierState(spins, k, subspace) for k in range(subspace_size)]
    # exitation_states = [SingleExcitationState(spins, k+1) for k in range(subspace_size)]
    print(init_state.subspace_ket)

    chain = Chain1dSubspaceLongRange(spins=spins, dt=0.01)

    chain.initialise(init_state)
    times, states = chain.time_evolution(time=time)
    print(states[-1])
    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots(figsize=[8, 6])
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"j = {j+1}" for j in range(spins)])
    line1 = mlines.Line2D(
        [],
        [],
        color="tab:green",
        linestyle="solid",
        label="$|1\\rangle, |5\\rangle, |9\\rangle, |13\\rangle$",
    )
    line2 = mlines.Line2D(
        [],
        [],
        color="tab:purple",
        linestyle="solid",
        label="$|3\\rangle, |7\\rangle, |11\\rangle, |15\\rangle$",
    )
    line3 = mlines.Line2D(
        [],
        [],
        color="tab:brown",
        linestyle="solid",
        label="$|2\\rangle, |4\\rangle, |6\\rangle, |8\\rangle,$"
        "\n$|10\\rangle, |12\\rangle, |14\\rangle, |16\\rangle$",
    )
    legend1 = plt.legend(
        handles=[line1, line2, line3],
        bbox_to_anchor=(1, 1),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend1)
    ax.set_xlabel("Time $(s/\hbar)$", fontsize=fontsize_axis)
    plt.tight_layout()
    ax.grid()
    fig.savefig("fig", bbox_extra_artists=(legend1,), bbox_inches="tight")
    plt.show()


def quantum_communication(
    spins,
    marked_strength,
    switch_time,
    alpha=1,
    start_site=1,
    final_site=15,
    open_chain=False,
    dt=0.01,
):
    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    init_state = SingleState(spins, start_site, single_subspace=True)

    chain = Chain1dSubspaceLongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )

    # print(chain.hamiltonian.H)
    chain.add_marked_site(start_site, marked_strength, gamma_rescale=True)
    # print(chain.hamiltonian.H_subspace)
    chain.initialise(init_state)

    chain.time_evolution(time=switch_time)

    chain.update_marked_site(start_site, -1)
    chain.update_marked_site(final_site, 1)

    times, states = chain.time_evolution(time=3 * switch_time, reset_state=False)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"j = {j+1}" for j in range(spins)], loc=2)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_communication_fast_protocol(
    m=3, alpha=1, open_chain=True, dt=0.01, scaling=0.25
):
    def calculate_time(group_sizeA, group_sizeB):
        t = (np.pi * ((group_sizeA + group_sizeB - 1) ** alpha)) / (
            2 * np.sqrt(group_sizeA * group_sizeB) * scaling
        )
        return t

    def next_group(b, direction="up"):
        if direction == "up":
            g = [b[-1] + i for i in range(1, 2 * len(b) + 1, 1)]
        elif direction == "down":
            g = [b[-1] + i for i in range(1, len(b) // 2 + 1, 1)]
        return g

    def increase_step(schedule):
        groupA = schedule["groups"][schedule["steps"]]
        schedule["steps"] += 1
        group_sizeA = len(groupA)
        groupB = next_group(groupA, direction="up")
        group_sizeB = len(groupB)
        time = calculate_time(group_sizeA, group_sizeB)
        schedule["spins"] += group_sizeB

        schedule["groups"].append(groupB)
        schedule["group_sizes"].append(group_sizeB)
        schedule["times"].append(time)
        schedule["total_time"] += time
        schedule["interactions"].append([(i, j) for i in groupA for j in groupB])
        schedule["distances"].append(group_sizeA + group_sizeB - 1)

        if schedule["steps"] == m:
            return groupB
        else:
            return increase_step(schedule)

    def decrease_step(schedule):
        groupA = schedule["groups"][schedule["steps"]]
        schedule["steps"] += 1
        group_sizeA = len(groupA)
        groupB = next_group(groupA, direction="down")
        group_sizeB = len(groupB)
        time = calculate_time(group_sizeA, group_sizeB)
        schedule["spins"] += group_sizeB

        schedule["groups"].append(groupB)
        schedule["group_sizes"].append(group_sizeB)
        schedule["times"].append(time)
        schedule["total_time"] += time
        schedule["interactions"].append([(i, j) for i in groupA for j in groupB])
        schedule["distances"].append(group_sizeA + group_sizeB - 1)

        if group_sizeB == 1:
            return groupB
        else:
            return decrease_step(schedule)

    schedule = {
        "steps": 0,
        "spins": 1,
        "times": [],
        "total_time": 0,
        "groups": [[1]],
        "group_sizes": [1],
        "interactions": [],
        "distances": [],
    }

    groupM = increase_step(schedule)
    groupN = decrease_step(schedule)

    single_states = [
        SingleState(schedule["spins"], j + 1, single_subspace=True)
        for j in range(schedule["spins"])
    ]

    init_state = SingleState(schedule["spins"], 1, single_subspace=True)

    chain = Chain1dSubspaceLongRange(
        spins=schedule["spins"],
        dt=0.01,
        alpha=alpha,
        open_chain=open_chain,
        hamiltonian_scaling=scaling,
        interaction_map=0,
        interaction_distance_bound=2,
        noise=0,
        samples=1,
    )
    # chain.initialise(init_state)
    chain.initialise(init_state, noisy_evolution=True)

    # print(chain.hamiltonian.H_subspace)

    for step in range(schedule["steps"] - 1):
        chain.update_interaction_map(
            schedule["interactions"][step], schedule["distances"][step]
        )
        # chain.time_evolution(time=schedule["times"][step], reset_state=False)
        chain.noisy_time_evolution(time=schedule["times"][step], reset_state=False)

    chain.update_interaction_map(
        schedule["interactions"][schedule["steps"] - 1],
        schedule["distances"][schedule["steps"] - 1],
    )
    times, states = chain.noisy_time_evolution(
        time=schedule["times"][schedule["steps"] - 1], reset_state=False
    )

    print(schedule["total_time"])

    overlaps = []
    for j, state in enumerate(single_states):
        # overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        overlaps.append(chain.overlaps_noisy_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"j = {j+1}" for j in range(schedule["spins"])], loc=2)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_communication_fast_protocol_B(
    m=2, alpha=1, open_chain=True, dt=0.01, scaling=0.25, noise=0, samples=1
):
    def calculate_time(group_sizeA, group_sizeB):
        t = (
            np.arctan(np.sqrt(group_sizeB / group_sizeA))
            * (((group_sizeA + group_sizeB - 1)) ** alpha)
        ) / (np.sqrt(group_sizeA * group_sizeB) * scaling)
        return t

    def calculate_reverse_time(group_sizeA, group_sizeB):
        t = (np.pi * ((group_sizeA + group_sizeB - 1) ** alpha)) / (
            2 * np.sqrt(group_sizeA * group_sizeB) * scaling
        ) - calculate_time(group_sizeA, group_sizeB)
        return t

    def next_group(b):
        g = [b[-1] + i for i in range(1, 2 * len(b) + 1, 1)]
        return g

    def generate_reverse_groups(groups, spins):
        return [[spins + 1 - site for site in group[::-1]] for group in groups[::-1]]

    def break_groups(b):
        g1 = [b[i] for i in range(0, (len(b) + 1) // 2, 1)]
        g2 = [b[i] for i in range((len(b) + 1) // 2, len(b), 1)]
        return g1, g2

    def increase_step(schedule):
        groupA = schedule["groups"][schedule["steps"]]
        superposition_groupA = [site for group in schedule["groups"] for site in group]
        schedule["superposition_groups"].append(superposition_groupA)
        schedule["steps"] += 1
        group_sizeA = len(superposition_groupA)
        groupB = next_group(groupA)
        group_sizeB = len(groupB)
        time = calculate_time(group_sizeA, group_sizeB)
        schedule["spins"] += group_sizeB

        schedule["groups"].append(groupB)
        schedule["group_sizes"].append(group_sizeB)
        schedule["times"].append(time)
        schedule["total_time"] += time
        schedule["interactions"].append(
            [(i, j) for i in superposition_groupA for j in groupB]
        )
        schedule["distances"].append(group_sizeA + group_sizeB - 1)

        if schedule["steps"] == m:
            return groupB
        else:
            return increase_step(schedule)

    def decrease_step(schedule):
        groupA = schedule["reverse_groups"][schedule["reverse_steps"]]
        groupB = schedule["reverse_superposition_groups"][schedule["reverse_steps"]]
        schedule["reverse_steps"] += 1
        group_sizeA = len(groupA)
        group_sizeB = len(groupB)
        time = calculate_reverse_time(group_sizeA, group_sizeB)

        schedule["group_sizes"].append(group_sizeB)
        schedule["times"].append(time)
        schedule["total_time"] += time
        schedule["interactions"].append([(i, j) for i in groupA for j in groupB])
        schedule["distances"].append(group_sizeA + group_sizeB - 1)

        if group_sizeB == 1:
            return groupB
        else:
            return decrease_step(schedule)

    schedule = {
        "steps": 0,
        "reverse_steps": 0,
        "spins": 1,
        "times": [],
        "total_time": 0,
        "groups": [[1]],
        "superposition_groups": [],
        "reverse_superposition_groups": [],
        "reverse_groups": [],
        "group_sizes": [1],
        "interactions": [],
        "distances": [],
    }

    groupM = increase_step(schedule)

    schedule["reverse_groups"] = generate_reverse_groups(
        schedule["groups"], schedule["spins"]
    )

    schedule["reverse_superposition_groups"] = generate_reverse_groups(
        schedule["superposition_groups"], schedule["spins"]
    )

    groupN = decrease_step(schedule)

    init_state = SingleState(schedule["spins"], 1, single_subspace=True)

    single_states = [
        SingleState(schedule["spins"], j + 1, single_subspace=True)
        for j in range(schedule["spins"])
    ]

    chain = Chain1dSubspaceLongRange(
        spins=schedule["spins"],
        dt=dt,
        alpha=alpha,
        open_chain=open_chain,
        hamiltonian_scaling=scaling,
        interaction_map=0,
        interaction_distance_bound=2,
        noise=noise,
        samples=samples,
    )
    chain.initialise(init_state, noisy_evolution=True)

    num_superposition_groups = len(schedule["superposition_groups"])

    for step in range(schedule["steps"]):
        chain.update_interaction_map(
            schedule["interactions"][step], schedule["distances"][step]
        )
        chain.noisy_time_evolution(time=schedule["times"][step], reset_state=False)
        for site in schedule["groups"][step + 1]:
            print(f"Rotation applied to site {site}")
            chain.local_z_rotation(site, angle=np.pi / 2)
        print("---")

    for step in range(schedule["reverse_steps"] - 1):
        total_step = schedule["steps"] + step
        chain.update_interaction_map(
            schedule["interactions"][total_step], schedule["distances"][total_step]
        )
        for site in schedule["reverse_groups"][step]:
            print(f"Rotation applied to site {site}")
            chain.local_z_rotation(site, angle=np.pi / 2)
        chain.noisy_time_evolution(
            time=schedule["times"][total_step], reset_state=False
        )
        print("---")

    for site in schedule["reverse_groups"][schedule["reverse_steps"] - 1]:
        print(f"Rotation applied to site {site}")
        chain.local_z_rotation(site, angle=np.pi / 2)
    steps = schedule["steps"] + schedule["reverse_steps"]
    chain.update_interaction_map(
        schedule["interactions"][steps - 1],
        schedule["distances"][steps - 1],
    )

    times, states = chain.noisy_time_evolution(
        time=schedule["times"][steps - 1],
        reset_state=False,
    )

    print(schedule["total_time"])

    overlaps = []
    for j, state in enumerate(single_states):
        # overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        overlaps.append(chain.overlaps_noisy_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"j = {j+1}" for j in range(schedule["spins"])], loc=2)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_communication_all_to_one(
    spins, alpha=1, open_chain=True, dt=0.01, scaling=0.25, noise=0, samples=1
):
    def calculate_time(group_sizeA, group_sizeB):
        t = (np.pi * ((group_sizeA + group_sizeB - 1) ** alpha)) / (
            2 * np.sqrt(group_sizeA * group_sizeB) * scaling
        )
        return t

    start_site = 1
    final_site = spins

    start_group = [start_site]
    final_group = [final_site]
    intermediate_group = [
        i for i in range(1, spins + 1, 1) if i not in start_group + final_group
    ]

    interaction_map_1 = [(i, j) for i in start_group for j in intermediate_group]
    interaction_map_2 = [(i, j) for i in intermediate_group for j in final_group]

    distance_1 = max([abs(i - j) for i, j in interaction_map_1])
    distance_2 = max([abs(i - j) for i, j in interaction_map_2])

    time_1 = calculate_time(1, len(intermediate_group))
    time_2 = calculate_time(len(intermediate_group), 1)

    init_state = SingleState(spins, 1, single_subspace=True)

    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    chain = Chain1dSubspaceLongRange(
        spins=spins,
        dt=dt,
        alpha=alpha,
        open_chain=open_chain,
        hamiltonian_scaling=scaling,
        interaction_map=0,
        interaction_distance_bound=2,
        noise=noise,
        samples=samples,
    )
    chain.initialise(init_state, noisy_evolution=True)

    chain.update_interaction_map(interaction_map_1, distance_1)
    chain.noisy_time_evolution(time_1, reset_state=False)

    chain.update_interaction_map(interaction_map_2, distance_2)
    times, states = chain.noisy_time_evolution(time_2, reset_state=False)

    overlaps = []
    for j, state in enumerate(single_states):
        # overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        overlaps.append(chain.overlaps_noisy_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"j = {j+1}" for j in range(spins)], loc=2)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_communication_subspace_2(
    spins, marked_strength, period, time, dt=0.1, alpha=1, open_chain=False, subspace=2
):
    subspace_size = int(scipy.special.comb(spins, subspace))
    # init_state = SuperpositionState(
    #     spins=spins, subspace=subspace, period=period, offset=0
    # )
    init_state = SingleState(spins, subspace_size - 1, excitations=2)
    # init_state = SubspaceFourierState(spins, 0, subspace)
    # init_state = SingleExcitationState(spins, excited_spin=1)
    # final_state = SuperpositionState(spins=spins, subspace=subspace, period=period)

    single_states = [
        SingleState(subspace_size, subspace_size - j, single_subspace=True)
        for j in range(subspace_size)
    ]
    # subspace_size = int(scipy.special.comb(spins, subspace))
    # fourier_states = [SubspaceFourierState(spins, k, subspace) for k in range(subspace_size)]
    # exitation_states = [SingleExcitationState(spins, k+1) for k in range(subspace_size)]

    chain = Chain1dLongRange(spins=spins, dt=dt, alpha=alpha, open_chain=open_chain)

    chain.add_marked_site(1, marked_strength, gamma_rescale=True)
    chain.add_marked_site(2, 1, gamma_rescale=False)

    chain.add_marked_site(spins - 1, 1, gamma_rescale=False)
    chain.add_marked_site(spins, 1, gamma_rescale=False)
    # chain.add_marked_site(7, 1, gamma_rescale=False)
    # chain.add_marked_site(8, 1, gamma_rescale=False)
    # chain.add_marked_site(11, 8)
    # chain.add_marked_site(16, 8)
    chain.initialise(init_state, subspace_evolution=True)

    print(chain.hamiltonian.H_subspace)

    times, states = chain.time_evolution(time=time)

    overlaps = []
    state_labels = chain.state.state_labels_subspace()
    print(state_labels)
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1} ({state_labels[j]})")

    # chain.state.state_barplot()
    # print(chain.hamiltonian.H_subspace)

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"{sta}" for sta in chain.state.state_labels_subspace()])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def spatial_search_subspace_2(
    spins, marked_strength, period, time, dt=0.1, alpha=1, open_chain=False, subspace=2
):
    subspace_size = int(scipy.special.comb(spins, subspace))
    # init_state = SuperpositionState(
    #     spins=spins, subspace=subspace, period=period, offset=0
    # )
    # init_state = SingleState(spins, subspace_size - 1, excitations=2)
    # init_state = SubspaceFourierState(spins, 0, subspace)
    # init_state = SingleExcitationState(spins, excited_spin=1)
    init_state = SuperpositionState(spins=spins, subspace=subspace, period=period)

    single_states = [
        SingleState(subspace_size, subspace_size - j, single_subspace=True)
        for j in range(subspace_size)
    ]
    # subspace_size = int(scipy.special.comb(spins, subspace))
    # fourier_states = [SubspaceFourierState(spins, k, subspace) for k in range(subspace_size)]
    # exitation_states = [SingleExcitationState(spins, k+1) for k in range(subspace_size)]

    chain = Chain1dSubspace2LongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )

    chain.add_marked_site(1, marked_strength, gamma_rescale=True)
    chain.add_marked_site(spins, 1, gamma_rescale=False)

    # chain.add_marked_site(spins - 1, 1, gamma_rescale=False)
    # chain.add_marked_site(spins, 1, gamma_rescale=False)
    # chain.add_marked_site(7, 1, gamma_rescale=False)
    # chain.add_marked_site(8, 1, gamma_rescale=False)
    # chain.add_marked_site(11, 8)
    # chain.add_marked_site(16, 8)
    chain.initialise(init_state, subspace_evolution=True)

    # print(chain.hamiltonian.H_subspace)

    times, states = chain.time_evolution(time=time)

    overlaps = []
    state_labels = chain.state.state_labels_subspace()
    # print(state_labels)
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1} ({state_labels[j]})")

    # chain.state.state_barplot()
    # print(chain.hamiltonian.H_subspace)

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)

    # ax1.legend([f"{sta}" for sta in chain1.state.state_labels_subspace()])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()

    plt.show()


def spatial_search_subspace_2_test(
    spins, marked_strength, period, time, dt=0.1, alpha=1, open_chain=False, subspace=2
):
    subspace_size = int(scipy.special.comb(spins, subspace))
    # init_state = SuperpositionState(
    #     spins=spins, subspace=subspace, period=period, offset=0
    # )
    # init_state = SingleState(spins, subspace_size - 1, excitations=2)
    # init_state = SubspaceFourierState(spins, 0, subspace)
    # init_state = SingleExcitationState(spins, excited_spin=1)
    init_state1 = SuperpositionState(spins=spins, subspace=subspace, period=period)
    init_state2 = SuperpositionState(spins=spins, subspace=subspace, period=period)

    single_states = [
        SingleState(subspace_size, subspace_size - j, single_subspace=True)
        for j in range(subspace_size)
    ]
    # subspace_size = int(scipy.special.comb(spins, subspace))
    # fourier_states = [SubspaceFourierState(spins, k, subspace) for k in range(subspace_size)]
    # exitation_states = [SingleExcitationState(spins, k+1) for k in range(subspace_size)]

    chain1 = Chain1dLongRange(spins=spins, dt=dt, alpha=alpha, open_chain=open_chain)
    chain2 = Chain1dSubspace2LongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )

    chain1.add_marked_site(1, marked_strength, gamma_rescale=True)
    chain1.add_marked_site(spins, 1, gamma_rescale=False)

    chain2.add_marked_site(1, marked_strength, gamma_rescale=True)
    chain2.add_marked_site(spins, 1, gamma_rescale=False)

    # chain.add_marked_site(spins - 1, 1, gamma_rescale=False)
    # chain.add_marked_site(spins, 1, gamma_rescale=False)
    # chain.add_marked_site(7, 1, gamma_rescale=False)
    # chain.add_marked_site(8, 1, gamma_rescale=False)
    # chain.add_marked_site(11, 8)
    # chain.add_marked_site(16, 8)
    chain1.initialise(init_state1, subspace_evolution=True)
    chain2.initialise(init_state2, subspace_evolution=True)

    # print(chain.hamiltonian.H_subspace)

    times1, states1 = chain1.time_evolution(time=time)
    times2, states2 = chain2.time_evolution(time=time)

    overlaps1 = []
    overlaps2 = []
    state_labels1 = chain1.state.state_labels_subspace()
    state_labels2 = chain2.state.state_labels_subspace()
    # print(state_labels)
    for j, state in enumerate(single_states):
        overlaps1.append(chain1.overlaps_evolution(state.subspace_ket, states1))
        overlaps2.append(chain2.overlaps_evolution(state.subspace_ket, states2))
        # print(f"Computed overlaps for excitation state {j+1} ({state_labels[j]})")

    # chain.state.state_barplot()
    # print(chain.hamiltonian.H_subspace)

    fig, axs = plt.subplots(2, 2)
    for y in overlaps1:
        axs[0, 0].plot(times1, y)
    for y in overlaps2:
        axs[0, 1].plot(times2, y)
    # ax1.legend([f"{sta}" for sta in chain1.state.state_labels_subspace()])
    axs[0, 0].set(xlabel="$Time~(s/\hbar)$")
    axs[0, 0].grid()

    # ax2.legend([f"{sta}" for sta in chain2.state.state_labels_subspace()])
    axs[0, 1].set(xlabel="$Time~(s/\hbar)$")
    axs[0, 1].grid()
    axs[1, 0].imshow(np.real(chain1.hamiltonian.H_subspace))
    axs[1, 1].imshow(np.real(chain2.hamiltonian.H_subspace))

    plt.show()


def quantum_communication_dual_rail(
    spins,
    marked_strength,
    switch_time,
    alpha=1,
    start_site=1,
    final_site=15,
    open_chain=False,
    dt=0.01,
):
    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    init_state = SingleState(spins, start_site, single_subspace=True)

    chain = Chain1dSubspaceLongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )

    # print(chain.hamiltonian.H)
    chain.add_marked_site(start_site, marked_strength, gamma_rescale=True)

    # print(chain.hamiltonian.H_subspace)
    chain.initialise(init_state)
    chain.update_marked_site(start_site + 1, 1)

    chain.time_evolution(time=switch_time)

    chain.update_marked_site(start_site, -1)
    chain.update_marked_site(start_site + 1, -1)
    chain.update_marked_site(final_site - 1, 1)
    chain.update_marked_site(final_site, 1)

    times, states = chain.time_evolution(time=3 * switch_time, reset_state=False)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"j = {j+1}" for j in range(spins)], loc=2)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_communication_always_on(
    spins,
    marked_strength,
    switch_time,
    alpha=1,
    start_site=1,
    final_site=15,
    open_chain=False,
    dt=0.01,
):
    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    init_state = SingleState(spins, start_site, single_subspace=True)

    chain = Chain1dSubspaceLongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )

    # print(chain.hamiltonian.H)
    chain.add_marked_site(start_site, marked_strength, gamma_rescale=True)
    # print(chain.hamiltonian.H_subspace)
    chain.initialise(init_state)
    chain.update_marked_site(final_site, 1)

    times, states = chain.time_evolution(time=switch_time)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"j = {j+1}" for j in range(spins)], loc=2)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_communication_always_on_slow_comparison(
    spins,
    marked_strength,
    switch_time,
    alpha=1,
    start_site=1,
    final_site=15,
    open_chain=False,
    dt=0.01,
):
    pass


def quantum_communication_always_on_dual_rail(
    spins,
    marked_strength,
    switch_time,
    alpha=1,
    start_site=1,
    final_site=15,
    open_chain=False,
    dt=0.01,
):
    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    init_state = SingleState(spins, start_site, single_subspace=True)

    chain = Chain1dSubspaceLongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )

    # print(chain.hamiltonian.H)
    chain.add_marked_site(start_site, marked_strength, gamma_rescale=True)
    # print(chain.hamiltonian.H_subspace)
    chain.initialise(init_state)
    chain.update_marked_site(start_site + 1, 1)
    chain.update_marked_site(final_site - 1, 1)
    chain.update_marked_site(final_site, 1)

    times, states = chain.time_evolution(time=switch_time)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"j = {j+1}" for j in range(spins)], loc=2)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_communication_two_step_dual_rail(
    spins,
    marked_strength,
    switch_time,
    alpha=1,
    start_site=1,
    final_site=15,
    open_chain=False,
    dt=0.01,
):
    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    init_state = SingleState(spins, start_site + 1, single_subspace=True)

    chain = Chain1dSubspaceLongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )

    # print(chain.hamiltonian.H)
    chain.add_marked_site(start_site, marked_strength, gamma_rescale=True)
    # print(chain.hamiltonian.H_subspace)
    chain.initialise(init_state)
    chain.update_marked_site(final_site, 1)

    chain.time_evolution(time=switch_time)

    chain.update_marked_site(start_site, -1)
    chain.update_marked_site(final_site, -1)
    chain.update_marked_site(start_site + 1, 1)
    chain.update_marked_site(final_site - 1, 1)

    times, states = chain.time_evolution(time=3 * switch_time, reset_state=False)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"j = {j+1}" for j in range(spins)], loc=2)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_communication_always_on_superposition(
    spins,
    marked_strength,
    switch_time,
    alpha=1,
    start_site=1,
    final_site=15,
    open_chain=False,
    dt=0.01,
):
    superposition_state = FourierState(spins, 0, single_subspace=True)

    init_state = SingleState(spins, start_site, single_subspace=True)

    chain = Chain1dSubspaceLongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )

    # print(chain.hamiltonian.H)
    chain.add_marked_site(start_site, marked_strength, gamma_rescale=True)

    # print(chain.hamiltonian.H_subspace)
    chain.initialise(init_state)
    chain.update_marked_site(final_site, 1)

    times, states = chain.time_evolution(time=switch_time)

    overlaps = []
    overlap = chain.overlaps_evolution(superposition_state.subspace_ket, states)
    overlaps.append(overlap)
    print(f"Computed overlaps for superposition state")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"j = {j+1}" for j in range(spins)], loc=2)
    ax.set(ylabel="$|s\\rangle$ state amplitude")
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_communication_random_t1(
    spins, marked_strength, switch_time, alpha=1, start_site=1, final_site=15
):
    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    init_state = SingleState(spins, start_site, single_subspace=True)

    chain = Chain1dSubspaceLongRange(spins=spins, dt=0.01, alpha=alpha)

    # print(chain.hamiltonian.H)
    chain.add_marked_site(start_site, marked_strength, gamma_rescale=True)
    # print(chain.hamiltonian.H_subspace)
    chain.initialise(init_state)

    chain.time_evolution(time=switch_time)

    # chain.update_marked_site(start_site, -1)
    chain.update_marked_site(final_site, 1)

    times, states = chain.time_evolution(time=20 * switch_time, reset_state=False)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"j = {j+1}" for j in range(spins)], loc=2)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_communication_always_on_nearest_neighbour(
    spins, marked_strength, switch_time, start_site=1, final_site=15, gamma_rescale=True
):
    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    init_state = SingleState(spins, start_site, single_subspace=True)

    chain = Chain1dSubspace(spins=spins, dt=0.01)

    # print(chain.hamiltonian.H)
    chain.add_marked_site(start_site, marked_strength, gamma_rescale=gamma_rescale)
    # print(chain.hamiltonian.H_subspace)
    chain.initialise(init_state)
    chain.update_marked_site(final_site, 1)

    times, states = chain.time_evolution(time=20 * switch_time)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"j = {j+1}" for j in range(spins)], loc=2)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_communication_no_fields_nearest_neighbour(
    spins, switch_time, start_site=1, final_site=15, gamma_rescale=True
):
    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    init_state = SingleState(spins, start_site, single_subspace=True)

    chain = Chain1dSubspace(spins=spins, dt=0.01)

    # print(chain.hamiltonian.H)
    # chain.add_marked_site(start_site, marked_strength, gamma_rescale=gamma_rescale)
    # print(chain.hamiltonian.H_subspace)
    chain.initialise(init_state)
    # chain.update_marked_site(final_site, 1)

    times, states = chain.time_evolution(time=20 * switch_time)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"j = {j+1}" for j in range(spins)], loc=2)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def longrangeXYchain_spatial_search_nearest_neighbour(
    spins, marked_strength, period, time
):
    init_state = SuperpositionState(
        spins=spins, subspace=1, period=period, offset=0, single_subspace=True
    )
    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    chain = Chain1dSubspace(spins=spins, dt=0.05)

    chain.add_marked_site(1, marked_strength, gamma_rescale=True)

    print(chain.hamiltonian.H)
    chain.initialise(init_state)
    times, states = chain.time_evolution(time=time)
    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"j = {j+1}" for j in range(spins)])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_state_transfer_engineered_chain(j_noise=0):
    spins = 32
    time = 40

    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    init_state = SingleState(spins, 1, single_subspace=True)

    j_noises = np.random.normal(0, j_noise, spins)

    js_engineered = [
        (np.sqrt((spin + 1) * (spins - spin - 1)) / 16) + j_noises[spin]
        for spin in range(spins)
    ]

    print(js_engineered)
    chain = Chain1dSubspace(spins=spins, dt=0.01, js=js_engineered, open_chain=True)

    # print(chain.hamiltonian.H)

    chain.initialise(init_state)

    times, states = chain.time_evolution(time=time)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f"site {j+1}" for j in range(spins)], loc=2)
    ax.legend([f"site 1", f"site 32"], loc=1)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_state_transfer_engineered_chain_noise_averaging(samples=100):
    spins = 32
    time = 40

    j_noises = [i / 1000 for i in range(0, 70, 5)]

    fidelities_for_transfer = []
    times_for_transfer = []

    for j_noise in j_noises:
        fidelities_for_transfer_sample = []
        times_for_transfer_sample = []

        for _ in range(samples):
            noise = np.random.normal(0, j_noise, spins)
            js_engineered = [
                (np.sqrt((spin + 1) * (spins - spin - 1)) / 16) + noise[spin]
                for spin in range(spins)
            ]

            init_state = SingleState(spins, 1, single_subspace=True)
            final_state = SingleState(spins, spins, single_subspace=True)
            chain = Chain1dSubspace(
                spins=spins, dt=0.01, js=js_engineered, open_chain=True
            )

            chain.initialise(init_state)

            times, states = chain.time_evolution(time=time)

            overlaps = chain.overlaps_evolution(final_state.subspace_ket, states)
            peaks, _ = find_peaks(overlaps, height=(0.2, 1.01))
            fidelities_for_transfer_sample.append(overlaps[peaks[0]])
            times_for_transfer_sample.append(times[peaks[0]])
        print(f"Computed overlaps for noise = {j_noise} (out of {j_noises[-1]})")

        fidelities_for_transfer.append(np.mean(fidelities_for_transfer_sample))
        times_for_transfer.append(np.mean(times_for_transfer_sample))

    fig, ax = plt.subplots()
    ax.plot(j_noises, fidelities_for_transfer)
    ax.legend([f"Fidelity"], loc=0)
    ax.set(xlabel="Coupling strength variation standard deviation")
    ax.grid()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(j_noises, times_for_transfer)
    ax.legend([f"Time"], loc=0)
    ax.set(xlabel="Coupling strength variation standard deviation")
    ax.grid()
    plt.show()


def quantum_state_transfer_weakly_coupled():
    spins = 32
    time = 1000

    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    init_state = SingleState(spins, 1, single_subspace=True)

    epsilon = 0.05
    J = 1

    js = [1 for _ in range(spins)]
    js_engineered = [np.sqrt((spin + 1) * (spins - spin - 1)) for spin in range(spins)]
    js_weakly = [epsilon if spin in [0, spins - 2] else J for spin in range(spins)]

    chain = Chain1dSubspace(spins=spins, dt=0.01, js=js_weakly, open_chain=True)

    for spin in range(spins):
        chain.add_marked_site(spin + 1, 1)

    # print(chain.hamiltonian.H)

    chain.initialise(init_state)

    times, states = chain.time_evolution(time=time)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"site {j+1}" for j in range(spins)], loc=1)
    # ax.legend([f"site 1", f"site 32"], loc=1)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_state_transfer_swap_every_four():
    spins = 31
    switch_time = 53.4

    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    init_state = SingleState(spins, 1, single_subspace=True)

    js = [0] * 27 + [1] * 3 + [0]
    print(js)

    chain = Chain1dSubspace(spins=spins, dt=0.01, js=js, open_chain=True)

    chain.initialise(init_state)

    print(chain.hamiltonian.H)
    while js[0] == 0:
        chain.time_evolution(time=switch_time, reset_state=False)
        js = js[3:] + [0] * 3
        print(js)
        print(chain.hamiltonian.H)
        chain.update_js(js)
    times, states = chain.time_evolution(time=switch_time, reset_state=False)
    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")
    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"site {j+1}" for j in range(spins)], bbox_to_anchor=[1, 1.1])
    # ax.legend([f"site 1", f"site 32"], loc=1)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def quantum_state_transfer_swap_test():
    spins = 8
    switch_time = 53.4

    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    init_state = SingleState(spins, 1, single_subspace=True)

    js = [0] * 4 + [1] * 4
    print(js)

    chain = Chain1dSubspace(spins=spins, dt=0.01, js=js, open_chain=True)

    chain.initialise(init_state)

    print(chain.hamiltonian.H)

    chain.time_evolution(time=switch_time, reset_state=False)
    js = [0] + [1] * 3 + [0] * 4
    print(js)
    chain.update_js(js)
    print(chain.hamiltonian.H)

    times, states = chain.time_evolution(time=2 * switch_time, reset_state=False)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")
    print()
    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"site {j+1}" for j in range(spins)], loc=1)
    # ax.legend([f"site 1", f"site 32"], loc=1)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def basic_two_spins_hubbard_model(
    spins,
    time=100,
    dt=0.1,
    j=1,
    j_noise=0,
    u=1,
    v=1,
    e=1,
    open_chain=True,
    use_js_engineered=True,
    use_us_engineered=False,
    use_vs_engineered=False,
    use_es_engineered=False,
    only_target_state=False,
):
    j_noises = np.random.normal(0, j_noise, spins)

    js_engineered = (
        [
            j * (np.sqrt((spin + 1) * (spins - spin - 1)) / (spins / 2))
            + j_noises[spin]
            for spin in range(spins)
        ]
        if use_js_engineered
        else j
    )

    print(f"js_engineered = {js_engineered}")

    us_engineered = (
        [
            u * (np.sqrt((spin + 1) * (spins - spin)) / (spins / 2))
            for spin in range(spins)
        ]
        if use_us_engineered
        else u
    )

    print(f"us_engineered = {us_engineered}")

    vs_engineered = (
        [
            v * (np.sqrt((spin + 1) * (spins - spin)) / (spins / 2))
            for spin in range(spins)
        ]
        if use_vs_engineered
        else v
    )

    print(f"vs_engineered = {vs_engineered}")

    es_engineered = (
        [
            e * (np.sqrt((spin + 1) * (spins - spin)) / (spins / 2))
            for spin in range(spins)
        ]
        if use_es_engineered
        else e
    )
    # es_engineered = [10, 0, 10]
    es_engineered = [10, 5, 0, 5, 10]
    print(f"es_engineered = {es_engineered}")

    init_state = TwoParticleHubbbardState(spins, sites=[3, 3])
    # sites = [1, 2]
    # state = [
    #     1 / np.sqrt(2)
    #     if (
    #         (i + 1 == sites[0] and j + 1 == sites[1])
    #         or (j + 1 == sites[0] and i + 1 == sites[1])
    #     )
    #     else 0
    #     for i in range(spins)
    #     for j in range(spins)
    # ]
    # init_state = TwoParticleHubbbardState(spins, sites=[1, 2], state_array=state)

    if only_target_state:
        complete_states = [
            TwoParticleHubbbardState(spins, sites=[3, 3]),
            TwoParticleHubbbardState(spins, sites=[2, 4]),
            TwoParticleHubbbardState(spins, sites=[4, 2]),
        ]
    else:
        complete_states = [
            TwoParticleHubbbardState(spins, sites=[i + 1, j + 1])
            for i in range(spins)
            for j in range(spins)
        ]

    chain = Chain1dSubspace2particles(
        spins=spins,
        dt=dt,
        js=js_engineered,
        us=us_engineered,
        es=es_engineered,
        vs=vs_engineered,
        open_chain=open_chain,
    )
    print(chain.hamiltonian.H_subspace)

    chain.initialise(init_state, subspace_evolution=True)

    times, states = chain.time_evolution(time=time)

    # print(chain.state.state_labels_subspace())
    overlaps = []
    for j, state in enumerate(complete_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    # for y in overlaps:
    #     ax.plot(times, y)
    ax.plot(times, overlaps[0])
    ax.plot(times, overlaps[1] + overlaps[2])
    ax.plot(times, overlaps[0] + overlaps[1] + overlaps[2])
    # ax.legend([f"{sta}" for sta in chain.state.state_labels_subspace()[:10]])
    ax.legend(["020", "101", "020 + 101"], loc="upper right")
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def end_to_end_two_spins_hubbard_model(
    spins,
    time=100,
    dt=0.1,
    j=1,
    j_noise=0,
    u=1,
    v=1,
    e=1,
    open_chain=True,
    use_js_engineered=True,
    use_us_engineered=False,
    use_vs_engineered=False,
    use_es_engineered=False,
    only_target_state=False,
):
    j_noises = np.random.normal(0, j_noise, spins)

    js_engineered = (
        [
            j * (np.sqrt((spin + 1) * (spins - spin - 1)) / (spins / 2))
            + j_noises[spin]
            for spin in range(spins)
        ]
        if use_js_engineered
        else j
    )

    print(f"js_engineered = {js_engineered}")

    us_engineered = (
        [
            u * (np.sqrt((spin + 1) * (spins - spin)) / (spins / 2))
            for spin in range(spins)
        ]
        if use_us_engineered
        else u
    )

    print(f"us_engineered = {us_engineered}")

    vs_engineered = (
        [
            v * (np.sqrt((spin + 1) * (spins - spin)) / (spins / 2))
            for spin in range(spins)
        ]
        if use_vs_engineered
        else v
    )

    print(f"vs_engineered = {vs_engineered}")

    es_engineered = (
        [
            e * (np.sqrt((spin + 1) * (spins - spin)) / (spins / 2))
            for spin in range(spins)
        ]
        if use_es_engineered
        else e
    )
    # es_engineered = [10, 0, 10]
    # es_engineered = [0, 0]
    print(f"es_engineered = {es_engineered}")

    # init_state = TwoParticleHubbbardState(spins, sites=[1, 3])
    sites = [1, 1]

    # state = [
    #     1 / np.sqrt(2)
    #     if (
    #         (i + 1 == sites[0] and j + 1 == sites[1])
    #         or (j + 1 == sites[0] and i + 1 == sites[1])
    #     )
    #     else 0
    #     for i in range(spins)
    #     for j in range(spins)
    # ]
    # init_state = TwoParticleHubbbardState(spins, sites=sites, state_array=state)

    init_state = TwoParticleHubbbardState(spins, sites=sites)

    if only_target_state:
        complete_states = [
            TwoParticleHubbbardState(spins, sites=[1, 1]),
            TwoParticleHubbbardState(spins, sites=[spins, spins]),
            TwoParticleHubbbardState(spins, sites=[spins // 2, spins // 2]),
        ]
    else:
        complete_states = [
            TwoParticleHubbbardState(spins, sites=[i + 1, j + 1])
            for i in range(spins)
            for j in range(spins)
        ]

    chain = Chain1dSubspace2particles(
        spins=spins,
        dt=dt,
        js=js_engineered,
        us=us_engineered,
        es=es_engineered,
        vs=vs_engineered,
        open_chain=open_chain,
    )
    # print(chain.hamiltonian.H_subspace)

    chain.initialise(init_state, subspace_evolution=True)

    times, states = chain.time_evolution(time=time)

    # print(chain.state.state_labels_subspace())
    overlaps = []
    for j, state in enumerate(complete_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.plot(times, overlaps[0])
    # ax.plot(times, overlaps[1] + overlaps[2])
    # ax.plot(times, overlaps[0] + overlaps[1] + overlaps[2])
    # ax.legend([f"{sta}" for sta in chain.state.state_labels_subspace()[:10]])
    # ax.legend(["020", "101", "020 + 101"], loc="upper right")
    ax.legend(
        [
            "$| 1 \\rangle | n \\rangle$",
            "$| n \\rangle | 1 \\rangle$",
            "$| n/2 \\rangle | n/2 \\rangle$",
        ]
    )
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def two_sites_two_spins_hubbard_model(
    time=100,
    dt=0.1,
    js=1,
    j_noise=0,
    us=1,
    vs=1,
    es=1,
    open_chain=True,
    only_target_state=False,
):
    spins = 2
    j_noises = np.random.normal(0, j_noise, spins)

    init_state = TwoParticleHubbbardState(spins, sites=[1, 1])
    # sites = [1, 1]
    # state = [
    #     1 / np.sqrt(2)
    #     if (
    #         (i + 1 == sites[0] and j + 1 == sites[1])
    #         or (j + 1 == sites[0] and i + 1 == sites[1])
    #     )
    #     else 0
    #     for i in range(spins)
    #     for j in range(spins)
    # ]
    # init_state = TwoParticleHubbbardState(spins, sites=[1, 2], state_array=state)

    if only_target_state:
        complete_states = [
            TwoParticleHubbbardState(spins, sites=[2, 2]),
            TwoParticleHubbbardState(spins, sites=[1, 2]),
            TwoParticleHubbbardState(spins, sites=[2, 1]),
        ]
    else:
        complete_states = [
            TwoParticleHubbbardState(spins, sites=[i + 1, j + 1])
            for i in range(spins)
            for j in range(spins)
        ]

    chain = Chain1dSubspace2particles(
        spins=spins,
        dt=dt,
        js=js,
        us=us,
        es=es,
        vs=vs,
        open_chain=open_chain,
    )
    print(chain.hamiltonian.H_subspace)

    chain.initialise(init_state, subspace_evolution=True)

    times, states = chain.time_evolution(time=time)

    # print(chain.state.state_labels_subspace())
    overlaps = []
    for j, state in enumerate(complete_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.plot(times, overlaps[0])
    # ax.plot(times, overlaps[1] + overlaps[2])
    # ax.plot(times, overlaps[0] + overlaps[1] + overlaps[2])
    ax.legend([f"{sta}" for sta in chain.state.state_labels_subspace()])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def n_spins_hubbard_model(spins, excitations, edges, time, dt):
    init_state = NParticleHubbbardState(
        spins,
        excitations,
        state=[0, 2, 0],
    )

    states = [
        np.array(state)
        for state in itertools.product(range(excitations + 1), repeat=spins)
        if np.sum(state) == excitations
    ]
    complete_states = [
        NParticleHubbbardState(spins, excitations, state) for state in states
    ]
    for state in complete_states:
        print(state.subspace_ket)
    chain = Chain1dSubspaceNparticles(
        spins=spins,
        excitations=2,
        edges=edges,
        dt=dt,
        js=1,
        us=20,
        es=[10, 0, 10],
        vs=10,
    )
    print(chain.hamiltonian.H_subspace)

    chain.initialise(init_state, subspace_evolution=True)

    times, states = chain.time_evolution(time=time)

    # print(chain.state.state_labels_subspace())
    overlaps = []
    for j, state in enumerate(complete_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.plot(times, overlaps[0])
    # ax.plot(times, overlaps[1] + overlaps[2])
    # ax.plot(times, overlaps[0] + overlaps[1] + overlaps[2])
    ax.legend([f"{sta}" for sta in chain.state.state_labels_subspace()])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def non_superposition_spatial_search(
    spins, k, marked_strength, time, dt=0.1, alpha=1, open_chain=False
):
    init_state = FourierState(spins=spins, k=k, single_subspace=True)
    # init_state = SingleState(spins, 1, single_subspace=True)

    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    chain = Chain1dSubspaceLongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )

    chain.add_marked_site(1, marked_strength, gamma_rescale=True)

    chain.initialise(init_state, subspace_evolution=True)
    times, states = chain.time_evolution(time=time)

    # print(chain.hamiltonian.H_subspace)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"j = {j+1}" for j in range(spins)])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def non_superposition_spatial_search_k_states(
    spins, k, marked_strength, time, dt=0.1, alpha=1, open_chain=False
):
    init_state = FourierState(spins=spins, k=k, single_subspace=True)
    # init_state = SingleState(spins, 1, single_subspace=True)

    single_states = [FourierState(spins, k, single_subspace=True) for k in range(spins)]

    chain = Chain1dSubspaceLongRange(
        spins=spins, dt=dt, alpha=alpha, open_chain=open_chain
    )

    chain.add_marked_site(1, marked_strength, gamma_rescale=True)

    chain.initialise(init_state, subspace_evolution=True)
    times, states = chain.time_evolution(time=time)

    # print(chain.hamiltonian.H_subspace)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"k = {j}" for j in range(spins)])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def spatial_search_experimental_parameters(
    spins, marked_strength, time, dt=0.000001, mu=2 * np.pi * 6.01e6, gamma_rescale=True
):
    init_state = SuperpositionState(
        spins=spins, subspace=1, period=1, offset=0, single_subspace=True
    )
    # init_state = SingleState(spins, 1, single_subspace=True)

    single_states = [
        SingleState(spins, j + 1, single_subspace=True) for j in range(spins)
    ]

    chain = Chain1dSubspaceLongRangeExp(spins=spins, dt=dt, mu=mu)

    if not gamma_rescale:
        marked_strength = 1 / marked_strength
    chain.add_marked_site(1, marked_strength, gamma_rescale=gamma_rescale)

    chain.initialise(init_state, subspace_evolution=True)
    times, states = chain.time_evolution(time=time)

    # print(chain.hamiltonian.H_subspace)

    overlaps = []
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1}")

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"j = {j+1}" for j in range(spins)])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def compute_approximate_gamma_and_mu(target_alpha, spins):
    Omega = np.ones((spins, 1)) * 2 * np.pi * 1e6  # rabi frequency
    delta_k = 2 * 2 * np.pi / 355e-9 * np.array([1, 0, 0])  # wave vector difference
    mu_bounds = [np.ones(1) * 2 * np.pi * 6.005e6, np.ones(1) * 2 * np.pi * 6.08e6]
    steps = 10
    mu, alpha, Js = mu_for_specific_alpha_BO(
        target_alpha, mu_bounds, steps, spins, Omega, delta_k
    )
    print(f"mu start = {mu_bounds[0]}, mu end = {mu_bounds[1]}")

    # print(f"Js = {Js}")
    eigenvectors, eigenvalues = Hamiltonian.spectrum(Js)
    s_1 = Hamiltonian.s_parameter(
        eigenvalues, 1, open_chain=True, eigenvectors=eigenvectors
    )
    s_2 = Hamiltonian.s_parameter(
        eigenvalues, 2, open_chain=True, eigenvectors=eigenvectors
    )
    gamma = s_1
    fidelity = s_1 ** 2 / s_2
    time = (np.pi / 2) * np.sqrt(spins / fidelity)
    print(f"approx gamma = {s_1}")
    print(f"approx fidelity = {fidelity}")
    print(f"approx time = {time}")
    print(f"approx mu = {mu} for alpha = {alpha}")

    return gamma, time, mu


def neel_state_intitialisation(spins=8, time=100, dt=0.1, open_chain=True):
    subspace_size = int(scipy.special.comb(spins, spins / 2))

    state_list = [0] * 20 + [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)] + [0] * 46

    init_state = SpecifiedState(spins, state_list, excitations=4)

    single_states = [
        SingleState(spins, subspace_size - j, excitations=4, single_subspace=False)
        for j in range(subspace_size)
    ]

    chain = Chain1d(
        spins=spins,
        dt=dt,
        jx=0,
        jy=0,
        jz=1,
        h_field=1,
        h_field_axis="x",
        open_chain=open_chain,
    )

    # chain.initialise(init_state, subspace_evolution=True)

    # state_labels = chain.state.state_labels_subspace()
    # print(state_labels)
    # print(state_labels[20])
    # print(state_labels[23])

    chain.initialise(init_state, subspace_evolution=True)

    times, states = chain.time_evolution(time=time)

    overlaps = []
    state_labels = chain.state.state_labels_subspace()
    print(state_labels)
    for j, state in enumerate(single_states):
        overlaps.append(chain.overlaps_evolution(state.subspace_ket, states))
        print(f"Computed overlaps for excitation state {j+1} ({state_labels[j]})")

    # chain.state.state_barplot()
    # print(chain.hamiltonian.H_subspace)

    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    ax.legend([f"{sta}" for sta in chain.state.state_labels_subspace()])
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


if __name__ == "__main__":

    spins = 8
    subspace = 1
    period = 1
    time = 10

    # test2(2, 1, np.pi)
    # test3(2, 1, np.pi)
    # test4(2, 1, np.pi)
    # u_list = [0.5, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    # basic_two_spins_hubbard_model(
    #     spins=5,
    #     j=1,
    #     u=20,
    #     e=0,
    #     v=10,
    #     use_js_engineered=False,
    #     use_es_engineered=True,
    #     use_us_engineered=False,
    #     use_vs_engineered=False,
    #     dt=0.01,
    #     time=100,
    #     only_target_state=True,
    #     open_chain=True,
    # )

    # end_to_end_two_spins_hubbard_model(
    #     spins=20,
    #     j=1,
    #     u=20,
    #     e=0,
    #     v=0,
    #     use_js_engineered=False,
    #     use_es_engineered=False,
    #     use_us_engineered=False,
    #     use_vs_engineered=False,
    #     dt=0.01,
    #     time=500,
    #     only_target_state=True,
    #     open_chain=False,
    # )

    # two_sites_two_spins_hubbard_model(
    #     js=1,
    #     us=[20, 20],
    #     es=[0, 0],
    #     vs=[0, 0],
    #     dt=0.01,
    #     time=20,
    #     only_target_state=False,
    #     open_chain=True,
    # )
    # js = [
    #     0.7453559924999299,
    #     0.9428090415820635,
    #     1.0,
    #     0.9428090415820635,
    #     0.7453559924999299,
    #     0.0,
    # ]
    # chain = Chain1dSubspace2particles(spins=6, dt=1, js=js, us=1, open_chain=True)
    # print(chain.hamiltonian.H_subspace)
    # longrangeXYchain_spatial_search(64, 6.78 / 64, 1, 28)
    # longrangeXYchain_spatial_search(
    #     spins=8,
    #     marked_strength=0.265,
    #     period=1,
    #     time=20,
    #     dt=0.1,
    #     alpha=1,
    #     open_chain=True,
    # )
    # longrangeXYchain_spatial_search_subspace_2(
    #     spins=8,
    #     marked_strength=0.265,
    #     period=1,
    #     time=100,
    #     dt=0.1,
    #     alpha=1,
    #     open_chain=True,
    #     subspace=2,
    # )

    # longrangeXYchain_spatial_search(
    #     spins=8,
    #     marked_strength=0.265,
    #     period=1,
    #     time=20,
    #     dt=0.1,
    #     alpha=1,
    #     open_chain=True,
    # )
    # longrangeXYchain_spatial_search_subspace_2(
    #     spins=8,
    #     marked_strength=0.265,
    #     period=1,
    #     time=100,
    #     dt=0.1,
    #     alpha=1,
    #     open_chain=True,
    #     subspace=2,
    # )

    # longrangeXYchain_spatial_search(
    #     spins=8,
    #     marked_strength=0.265,
    #     period=1,
    #     time=100,
    #     dt=0.1,
    #     alpha=1,
    #     open_chain=True,
    # )
    # non_superposition_spatial_search(
    #     spins=8,
    #     k=1,
    #     marked_strength=0.5,
    #     time=100,
    #     dt=0.01,
    #     alpha=1,
    #     open_chain=False,
    # )
    # non_superposition_spatial_search_k_states(
    #     spins=8,
    #     k=1,
    #     marked_strength=0.0625,
    #     time=100,
    #     dt=0.01,
    #     alpha=0.0005,
    #     open_chain=False,
    # )
    # longrangeXYchain_spatial_search_subspace_2(
    #     spins=8,
    #     marked_strength=0.265,
    #     period=1,
    #     time=100,
    #     dt=1,
    #     alpha=1,
    #     open_chain=True,-
    #     subspace=2,
    # )

    # longrangeXYchain_spatial_search(
    #     spins=256,
    #     marked_strength=0.10000836736049502,
    #     period=1,
    #     time=112,
    #     dt=0.1,
    #     alpha=1,
    #     open_chain=True,
    # )

    # quantum_communication(
    #     256,
    #     (1 - 0.02) * 0.10000836736049502,
    #     switch_time=58.60999999999691,
    #     alpha=1,
    #     start_site=1,
    #     final_site=256,
    #     open_chain=True,
    # )
    # quantum_communication(
    #     16,
    #     3.25 / 16,
    #     switch_time=10.2,
    #     alpha=1,
    #     start_site=1,
    #     final_site=16,
    #     open_chain=True,
    # )
    # quantum_communication_fast_protocol()
    # quantum_communication_fast_protocol_B()

    # spins = 68

    # longrangeXYchain_spatial_search(
    #     spins=spins,
    #     marked_strength=0.13560507724014415,
    #     period=1,
    #     time=40,
    #     dt=0.01,
    #     alpha=1,
    #     open_chain=True,
    # )
    # quantum_communication(
    #     spins=spins,
    #     marked_strength=0.138,
    #     switch_time=26.7,
    #     alpha=1,
    #     start_site=1,
    #     final_site=spins,
    #     open_chain=True,
    # )
    # quantum_communication_always_on(
    #     spins=spins,
    #     marked_strength=0.139,
    #     switch_time=40,
    #     alpha=1,
    #     start_site=1,
    #     final_site=spins,
    #     open_chain=True,
    # )

    # longrangeXYchain_spatial_search_noisy(
    #     spins=10,
    #     marked_strength=0.2307835686727283,
    #     period=1,
    #     time=20,
    #     dt=0.01,
    #     alpha=1,
    #     open_chain=True,
    #     noise=0,
    #     samples=10,
    # )
    # longrangeXYchain_spatial_search_noise_comparisons(
    #     spins=256,
    #     marked_strength=21.272124918913900 / 256,
    #     period=1,
    #     time=50,
    #     dt=0.01,
    #     alpha=1,
    #     open_chain=False,
    #     samples=100,
    # )
    # quantum_communication_random_t1(
    #     64, 6.78 / 64, 13, alpha=1, start_site=1, final_site=24
    # )

    # quantum_communication(64, 6.78 / 64, 13, alpha=1, start_site=1, final_site=24)
    # quantum_communication(64, 6.78 / 64, 10, alpha=1, start_site=1, final_site=24)
    # quantum_communication(64, 9.76 / 64, 13, alpha=1.2, start_site=1, final_site=24)
    # quantum_communication(64, 15.94 / 64, 13, alpha=1.5, start_site=1, final_site=24)

    # longrangeXYchain_spatial_search(96, 9.37 / 96, 1, 32)
    # quantum_communication(96, 9.37 / 96, 15.76, alpha=1, start_site=1, final_site=24)
    # quantum_communication(96, 9.37 / 96, 20, alpha=1, start_site=1, final_site=24)

    # longrangeXYchain_spatial_search_multi_sites(
    #     64, 0.13560507724014415, 1, 150, dt=0.1, alpha=1, open_chain=True
    # )

    # longrangeXYchain_spatial_search_multi_sites(
    #     4, 0.25, 1, 15, dt=0.05, alpha=0, open_chain=True
    # )
    # longrangeXYchain_spatial_search(4, 0.25, 1, 15, dt=0.05, alpha=0, open_chain=True)

    # quantum_communication_always_on(
    #     64,
    #     # 6.78 / 64,
    #     0.13560507724014415 / 3,
    #     3000,
    #     alpha=1,
    #     start_site=1,
    #     final_site=64,
    #     open_chain=True,
    #     dt=0.1,
    # )
    # quantum_communication_always_on(
    #     64,
    #     0.13827174390681074 / 8,
    #     8000000000,
    #     alpha=5,
    #     start_site=1,
    #     final_site=64,
    #     open_chain=True,
    #     dt=10000,
    # )
    # quantum_communication_always_on(
    #     224,
    #     0.10089519781285404,
    #     1000,
    #     alpha=1,
    #     start_site=1 + 2,
    #     final_site=224 - 2,
    #     open_chain=True,
    #     dt=1,
    # )

    # quantum_communication_always_on_superposition(
    #     224,
    #     0.08517756597287594,
    #     500,
    #     alpha=1,
    #     start_site=1,
    #     final_site=112,
    #     open_chain=False,
    #     dt=1,
    # )

    # quantum_communication_always_on(
    #     224,
    #     0.08517756597287594,
    #     1000,
    #     alpha=1,
    #     start_site=1,
    #     final_site=114,
    #     open_chain=False,
    #     dt=1,
    # )

    # quantum_communication_always_on(
    #     224,
    #     0.10089519781285405,
    #     1000,
    #     alpha=1,
    #     start_site=1,
    #     final_site=224,
    #     open_chain=True,
    #     dt=2,
    # )

    # longrangeXYchain_spatial_search(224, 0.1, 1, 500, open_chain=True)
    # quantum_communication_always_on(
    #     224,
    #     0.1 / 3,
    #     400,
    #     alpha=1,
    #     start_site=1,
    #     final_site=224,
    #     open_chain=True,
    #     dt=0.1,
    # )
    # quantum_communication_always_on(
    #     224,
    #     0.1,
    #     400,
    #     alpha=1,
    #     start_site=1,
    #     final_site=224,
    #     open_chain=True,
    # )

    # quantum_communication_always_on(
    #     224,
    #     0.10089519781285404,
    #     6000,
    #     alpha=1,
    #     start_site=1,
    #     final_site=224,
    #     open_chain=True,
    #     dt=1,
    # )

    # quantum_communication(
    #     224, 0.1, 520, alpha=1, start_site=1, final_site=224, open_chain=True, dt=1
    # )

    # quantum_communication(
    #     36,
    #     0.15864382828858214,
    #     88,
    #     alpha=1,
    #     start_site=1,
    #     final_site=36,
    #     open_chain=True,
    #     dt=1,
    # )

    # quantum_communication_always_on(
    #     224, 19.01 / 224, 24.06, alpha=1, start_site=1, final_site=224, open_chain=True
    # )
    # quantum_communication_always_on(
    #     224, 26 / 224, 24.06, alpha=1, start_site=1, final_site=224, open_chain=True
    # )
    # quantum_communication(
    #     224, 19.01 / 224, 100, alpha=1, start_site=1, final_site=224, open_chain=True
    # )

    # longrangeXYchain_spatial_search(512, 38.27 / 512, 1, 80)
    # quantum_communication(512, 38.27 / 512, 36.63, alpha=1, start_site=1, final_site=24)

    # longrangeXYchain_spatial_search(8, 0.17, 1, 8)
    # quantum_communication(8, 0.17, 4.45, alpha=1, start_site=1, final_site=4)
    # quantum_communication_always_on(8, 0.17, 4.45, alpha=1, start_site=1, final_site=4)

    # quantum_communication_always_on_nearest_neighbour(
    #     8, 0.17, 28, start_site=1, final_site=4
    # )

    # quantum_communication_no_fields_nearest_neighbour(6, 3, start_site=1, final_site=4)
    # quantum_communication_always_on(64, 6.78 / 64, 50, start_site=1, final_site=24)

    # quantum_communication_always_on_nearest_neighbour(
    #     64, 6.78 / 64, 50, start_site=1, final_site=24
    # )

    # quantum_communication(
    #     64,
    #     0.13560507724014415,
    #     24,
    #     alpha=1,
    #     start_site=1,
    #     final_site=64,
    #     open_chain=True,
    #     dt=0.01,
    # )
    # quantum_communication_dual_rail(
    #     64,
    #     0.13560507724014415,
    #     11.5,
    #     alpha=1,
    #     start_site=1,
    #     final_site=64,
    #     open_chain=True,
    #     dt=0.01,
    # )

    # quantum_communication_always_on(
    #     64,
    #     0.13560507724014415,
    #     100,
    #     alpha=1,
    #     start_site=1,
    #     final_site=64,
    #     open_chain=True,
    #     dt=0.01,
    # )

    # quantum_communication_always_on_dual_rail(
    #     64,
    #     0.13560507724014415,
    #     100,
    #     alpha=1,
    #     start_site=1,
    #     final_site=64,
    #     open_chain=True,
    #     dt=0.01,
    # )

    # quantum_communication_two_step_dual_rail(
    #     64,
    #     0.13560507724014415,
    #     34.5,
    #     alpha=1,
    #     start_site=1,
    #     final_site=64,
    #     open_chain=True,
    #     dt=0.01,
    # )

    # longrangeXYchain_spatial_search_subspace_2(
    #     spins=8,
    #     # marked_strength=0.265,
    #     marked_strength=0.36,
    #     period=1,
    #     time=30,
    #     dt=0.01,
    #     alpha=1,
    #     open_chain=True,
    #     subspace=2,
    # )

    # quantum_communication_subspace_2(
    #     spins=8,
    #     # marked_strength=0.265,
    #     marked_strength=0.2,
    #     period=1,
    #     time=100,
    #     dt=0.01,
    #     alpha=1,
    #     open_chain=True,
    #     subspace=2,
    # )

    # neel_state_intitialisation()

    # longrangeXYchain_spatial_search(
    #     spins=100,
    #     marked_strength=4.123399747319335 / 100,
    #     period=1,
    #     time=30,
    #     alpha=0.5,
    #     open_chain=True,
    #     dt=0.01,
    # )

    # quantum_communication(
    #     spins=8,
    #     marked_strength=1.3202764465995 / 8,
    #     switch_time=5.4,
    #     alpha=0.5,
    #     start_site=1,
    #     final_site=8,
    #     open_chain=True,
    #     dt=0.01,
    # )

    # quantum_communication_all_to_one(
    #     spins, alpha=0.5, open_chain=True, dt=0.01, scaling=0.25, noise=0, samples=1
    # )

    # quantum_communication_always_on(
    #     116,
    #     0.11811262856477409,
    #     3.5 * 1.5 * np.sqrt(116),
    #     alpha=1,
    #     start_site=1,
    #     final_site=116,
    #     open_chain=True,
    #     dt=0.01,
    # )

    # quantum_communication_always_on(
    #     60,
    #     0.1460755072612601,
    #     120,
    #     alpha=1,
    #     start_site=1,
    #     final_site=60,
    #     open_chain=False,
    #     dt=0.01,
    # )

    # spins = 31
    # approx_gamma, approx_time, mu = compute_approximate_gamma_and_mu(
    #     target_alpha=0.5, spins=spins
    # )

    # spatial_search_experimental_parameters(
    #     spins=spins,
    #     marked_strength=approx_gamma,
    #     mu=mu,
    #     time=approx_time * 2,
    #     dt=0.01,
    # )

    # spatial_search_experimental_parameters(
    #     spins=spins,
    #     marked_strength=approx_gamma,
    #     mu=mu,
    #     time=approx_time * 2 * approx_gamma,
    #     dt=1e-8,
    #     gamma_rescale=False,
    # )

    # spatial_search_subspace_2(
    #     spins=20,
    #     marked_strength=0.1074997587355275,
    #     period=1,
    #     time=15,
    #     dt=0.1,
    #     alpha=0.5,
    #     open_chain=True,
    # )

    n_spins_hubbard_model(
        spins=3, excitations=2, edges=[[0, 1], [1, 2]], time=5, dt=0.01
    )
