from chains import Chain1dSubspaceLongRange
from hamiltonians import Hamiltonian
from states import SingleState
from plots import plot_time_comparisons, plot_fidelity_comparisons
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
import scipy.special
from scipy.signal import find_peaks


def fidelity_time_peaks(spins, chain, states, times, fast_protocol=False, plot=False):
    final_state = SingleState(spins, spins, single_subspace=True)
    qst_fidelity = chain.overlaps_noisy_evolution(final_state.subspace_ket, states)

    if plot:
        fig, ax = plt.subplots()
        ax.plot(times, qst_fidelity)
        ax.legend([f"Fidelity"])
        ax.set(xlabel="$Time~(s/\hbar)$")
        ax.grid()
        plt.show()

    if fast_protocol:
        time = times[-1]
        fidelity = qst_fidelity[-1]
    else:
        peaks, _ = find_peaks(qst_fidelity, height=(0.3, 1.05))
        peak = peaks[0]
        time = times[peak]
        fidelity = qst_fidelity[peak]
    return fidelity, time


def quantum_communication_reverse_search(
    spins,
    marked_strength,
    switch_time,
    alpha=1,
    open_chain=True,
    dt=0.01,
    noise=0,
    samples=0,
):
    init_state = SingleState(spins, 1, single_subspace=True)

    chain = Chain1dSubspaceLongRange(
        spins=spins,
        dt=dt,
        alpha=alpha,
        open_chain=open_chain,
        noise=noise,
        samples=samples,
    )

    chain.add_marked_site(1, marked_strength, gamma_rescale=True)
    chain.initialise(init_state, noisy_evolution=True)

    chain.noisy_time_evolution(time=switch_time)
    chain.update_marked_site(1, -1)
    chain.update_marked_site(spins, 1)

    times, states = chain.noisy_time_evolution(
        time=1.5 * switch_time, reset_state=False
    )

    fidelity, time = fidelity_time_peaks(spins, chain, states, times)
    print(
        f"RS protocol for {spins} spins: computed fidelity of {fidelity} | "
        f"time {time} (switch time = {switch_time}) | "
        f"gamma = {marked_strength} | "
        f"noise = {noise} with samples = {samples}"
    )
    return fidelity, time


def quantum_communication_always_on(
    spins,
    marked_strength,
    switch_time,
    alpha=1,
    open_chain=True,
    dt=0.01,
    noise=0,
    samples=0,
):

    init_state = SingleState(spins, 1, single_subspace=True)

    chain = Chain1dSubspaceLongRange(
        spins=spins,
        dt=dt,
        alpha=alpha,
        open_chain=open_chain,
        noise=noise,
        samples=samples,
    )

    chain.add_marked_site(1, marked_strength, gamma_rescale=True)
    chain.initialise(init_state, noisy_evolution=True)

    chain.update_marked_site(spins, 1)

    times, states = chain.noisy_time_evolution(time=3 * switch_time, reset_state=False)

    fidelity, time = fidelity_time_peaks(spins, chain, states, times)
    print(
        f"AO protocol for {spins} spins: computed fidelity of {fidelity} | "
        f"time {time} (switch time = {switch_time}) | "
        f"gamma = {marked_strength} | "
        f"noise = {noise} with samples = {samples}"
    )
    return fidelity, time


def quantum_communication_fast_protocol_A(
    m=2,
    alpha=1,
    open_chain=True,
    dt=0.01,
    scaling=0.25,
    noise=0,
    samples=0,
    time_noise=0,
):
    def calculate_time(group_sizeA, group_sizeB):
        t = (np.pi * (((group_sizeA + group_sizeB - 1)) ** alpha)) / (
            2 * np.sqrt(group_sizeA * group_sizeB) * scaling
        ) + np.random.normal(0, time_noise)
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

    init_state = SingleState(schedule["spins"], 1, single_subspace=True)

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

    for step in range(schedule["steps"] - 1):
        chain.update_interaction_map(
            schedule["interactions"][step], schedule["distances"][step]
        )
        chain.noisy_time_evolution(time=schedule["times"][step], reset_state=False)

    chain.update_interaction_map(
        schedule["interactions"][schedule["steps"] - 1],
        schedule["distances"][schedule["steps"] - 1],
    )

    times, states = chain.noisy_time_evolution(
        time=schedule["times"][schedule["steps"] - 1], reset_state=False
    )

    fidelity, time = fidelity_time_peaks(
        schedule["spins"], chain, states, times, fast_protocol=True, plot=False
    )
    print(
        f"Fast protocol for {schedule['spins']} spins: computed fidelity of {fidelity} | "
        f"time {time} | "
        f"scaling = {scaling}"
    )
    return schedule["spins"], fidelity, time, schedule["total_time"]


def quantum_communication_fast_protocol_B(
    m=2,
    alpha=1,
    open_chain=True,
    dt=0.01,
    scaling=0.25,
    noise=0,
    samples=0,
    time_noise=0,
):
    def calculate_time(group_sizeA, group_sizeB):
        t = (
            np.arctan(np.sqrt(group_sizeB / group_sizeA))
            * (((group_sizeA + group_sizeB - 1)) ** alpha)
        ) / (np.sqrt(group_sizeA * group_sizeB) * scaling) + np.random.normal(
            0, time_noise
        )
        return t

    def calculate_reverse_time(group_sizeA, group_sizeB):
        t = (
            (np.pi * ((group_sizeA + group_sizeB - 1) ** alpha))
            / (2 * np.sqrt(group_sizeA * group_sizeB) * scaling)
            - calculate_time(group_sizeA, group_sizeB)
            + np.random.normal(0, time_noise)
        )
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
            chain.local_z_rotation(site, angle=np.pi / 2)

    for step in range(schedule["reverse_steps"] - 1):
        total_step = schedule["steps"] + step
        chain.update_interaction_map(
            schedule["interactions"][total_step], schedule["distances"][total_step]
        )
        for site in schedule["reverse_groups"][step]:
            chain.local_z_rotation(site, angle=np.pi / 2)
        chain.noisy_time_evolution(
            time=schedule["times"][total_step], reset_state=False
        )

    for site in schedule["reverse_groups"][schedule["reverse_steps"] - 1]:
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

    fidelity, time = fidelity_time_peaks(
        schedule["spins"], chain, states, times, fast_protocol=True, plot=False
    )
    print(
        f"Fast protocol for {schedule['spins']} spins: computed fidelity of {fidelity} | "
        f"time {time} | "
        f"scaling = {scaling}"
    )
    return schedule["spins"], fidelity, time, schedule["total_time"]


def quantum_communication_fast_strong(
    spin, alpha=1, open_chain=True, dt=0.01, scaling=0.25, noise=0, samples=1
):
    def calculate_time(group_sizeA, group_sizeB):
        t = (np.pi * ((group_sizeA + group_sizeB - 1) ** alpha)) / (
            2 * np.sqrt(group_sizeA * group_sizeB) * scaling
        )
        return t

    start_site = 1
    final_site = spin

    start_group = [start_site]
    final_group = [final_site]
    intermediate_group = [
        i for i in range(1, spin + 1, 1) if i not in start_group + final_group
    ]

    interaction_map_1 = [(i, j) for i in start_group for j in intermediate_group]
    interaction_map_2 = [(i, j) for i in intermediate_group for j in final_group]

    distance_1 = max([abs(i - j) for i, j in interaction_map_1])
    distance_2 = max([abs(i - j) for i, j in interaction_map_2])

    time_1 = calculate_time(1, len(intermediate_group))
    time_2 = calculate_time(len(intermediate_group), 1)

    init_state = SingleState(spin, 1, single_subspace=True)

    single_states = [
        SingleState(spin, j + 1, single_subspace=True) for j in range(spin)
    ]

    chain = Chain1dSubspaceLongRange(
        spins=spin,
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

    fidelity, time = fidelity_time_peaks(
        spin, chain, states, times, fast_protocol=True, plot=False
    )
    print(
        f"Fast (strongly long-range) protocol for {spin} spins: computed fidelity of {fidelity} | "
        f"time {time} | "
        f"scaling = {scaling}"
    )
    return fidelity, time


def ao_fidelity_time_protocols(
    spins,
    ao_gammas,
    ao_switch_times,
    alpha=1,
    open_chain=True,
    dt=0.01,
    noise=0,
    samples=0,
):
    ao_fidelities = []
    ao_times = []
    for i, spin in enumerate(spins):
        ao_fidelity, ao_time = quantum_communication_always_on(
            spin,
            ao_gammas[i],
            ao_switch_times[i],
            alpha=alpha,
            open_chain=open_chain,
            dt=dt,
            noise=noise,
            samples=samples,
        )
        ao_fidelities.append(ao_fidelity)
        ao_times.append(ao_time)
    return ao_fidelities, ao_times


def rs_fidelity_time_protocols(
    spins,
    rs_gammas,
    rs_switch_times,
    alpha=1,
    open_chain=True,
    dt=0.01,
    noise=0,
    samples=0,
):
    rs_fidelities = []
    rs_times = []
    for i, spin in enumerate(spins):
        rs_fidelity, rs_time = quantum_communication_reverse_search(
            spin,
            rs_gammas[i],
            rs_switch_times[i],
            alpha=alpha,
            open_chain=open_chain,
            dt=dt,
            noise=noise,
            samples=samples,
        )
        rs_fidelities.append(rs_fidelity)
        rs_times.append(rs_time)
    return rs_fidelities, rs_times


def fidelity_fast_strong(
    spins, alpha=1, open_chain=True, dt=0.01, rescalings=1, noise=0, samples=0
):
    fidelities = []
    times = []
    for i, spin in enumerate(spins):
        fidelity, time = quantum_communication_fast_strong(
            spin, alpha, open_chain, dt, rescalings, noise, samples
        )
        fidelities.append(fidelity)
        times.append(time)
    return fidelities, times


def fidelity_time_fast(
    protocol="A",
    steps=5,
    alpha=1,
    open_chain=True,
    dt=0.01,
    rescalings=1,
    noise=0,
    samples=0,
    time_noise=0,
):
    fast_spins = []
    fast_fidelities = []
    fast_times = []
    for m in range(1, steps + 1, 1):
        if protocol == "A":
            (
                fast_spin,
                fast_fidelity,
                fast_time,
                fast_total_time,
            ) = quantum_communication_fast_protocol_A(
                m,
                alpha=alpha,
                open_chain=open_chain,
                scaling=rescalings,
                noise=noise,
                samples=samples,
                time_noise=time_noise,
            )
        else:
            (
                fast_spin,
                fast_fidelity,
                fast_time,
                fast_total_time,
            ) = quantum_communication_fast_protocol_B(
                m,
                alpha=alpha,
                open_chain=open_chain,
                scaling=rescalings,
                noise=noise,
                samples=samples,
                time_noise=time_noise,
            )
        fast_spins.append(fast_spin)
        fast_fidelities.append(fast_fidelity)
        fast_times.append(fast_time)
    return fast_spins, fast_fidelities, fast_times


if __name__ == "__main__":
    alpha = 1
    noise = 0
    samples = 1

    spins = [int(4 * x) for x in range(2, 32, 1)]
    print(spins)
    gammaNs = [
        1.3202764465995,
        1.5979568403176392,
        1.8198737557925577,
        2.009995174710549,
        2.1789783751687013,
        2.3326207342825436,
        2.4744795755141875,
        2.6069206240875507,
        2.731610536900808,
        2.849775512827979,
        2.9623487131225663,
        3.0700597949497914,
        3.173492098973373,
        3.27312070841783,
        3.369338657548557,
        3.462475507482325,
        3.55281084114758,
        3.640584278818152,
        3.72600305147005,
        3.809247822555538,
        3.8904772292658985,
        3.969831471592342,
        4.047435182430634,
        4.123399747319335,
        4.197825197581354,
        4.270801769028302,
        4.342411195741055,
        4.4127277919814984,
        4.4818193631714465,
        4.54974797784146,
    ]
    gammas = [gammaN / spins[i] for i, gammaN in enumerate(gammaNs)]
    print(gammas)
    rs_switch_times = [
        5.4,
        6.8,
        7.94,
        8.89,
        9.74,
        10.53,
        11.31,
        12.1,
        12.89,
        13.66,
        14.38,
        15.05,
        15.66,
        16.23,
        16.78,
        17.29,
        17.79,
        18.26,
        18.73,
        19.17,
        19.61,
        20.04,
        20.46,
        20.88,
        21.3,
        21.72,
        22.150000000000002,
        22.59,
        23.05,
        23.52,
    ]
    ao_switch_times = [np.sqrt(2) * t for t in rs_switch_times]
    ao_gammas = gammas
    rs_gammas = gammas

    rs_fidelities, rs_times = rs_fidelity_time_protocols(
        spins,
        rs_gammas,
        rs_switch_times,
        alpha=alpha,
        open_chain=True,
        dt=0.01,
        noise=noise,
        samples=samples,
    )

    ao_fidelities, ao_times = ao_fidelity_time_protocols(
        spins,
        ao_gammas,
        ao_switch_times,
        alpha=alpha,
        open_chain=True,
        dt=0.01,
        noise=noise,
        samples=samples,
    )

    fast_strong_fidelities, fast_strong_times = fidelity_fast_strong(
        spins,
        alpha=0.6,
        open_chain=True,
        dt=0.01,
        rescalings=0.25,
        noise=noise,
        samples=samples,
    )

    fast_B_spins, fast_B_fidelities, fast_B_times = fidelity_time_fast(
        protocol="B",
        steps=6,
        alpha=1,
        open_chain=True,
        dt=0.01,
        rescalings=0.25,
        noise=noise,
        samples=samples,
    )

    plot_time_comparisons(spins, ao_times, rs_times, fast_B_spins, fast_B_times, False)
    plot_fidelity_comparisons(
        spins, ao_fidelities, rs_fidelities, fast_B_spins, fast_B_fidelities
    )

    # Fast algorithm for strongly long-range interacting systems
    plot_time_comparisons(spins, ao_times, rs_times, spins, fast_strong_times, False)
    plot_fidelity_comparisons(
        spins, ao_fidelities, rs_fidelities, spins, fast_strong_fidelities
    )
    # spins = [
    #     8,
    #     12,
    #     16,
    #     20,
    #     24,
    #     28,
    #     32,
    #     36,
    #     40,
    #     44,
    #     48,
    #     52,
    #     56,
    #     60,
    #     64,
    #     68,
    #     72,
    #     76,
    #     80,
    #     84,
    #     88,
    #     92,
    # ]
    # ao_gammas = [
    #     0.35,
    #     0.27,
    #     0.23,
    #     0.22,
    #     0.205,
    #     0.188,
    #     0.180,
    #     0.162,
    #     0.158,
    #     0.154,
    #     0.149,
    #     0.146,
    #     0.143,
    #     0.142,
    #     0.139,
    #     0.136,
    #     0.134,
    #     0.132,
    #     0.130,
    #     0.128,
    #     0.126,
    #     0.125,
    # ]
    # rs_gammas = [
    #     0.275,
    #     0.222,
    #     0.199,
    #     0.192,
    #     0.180,
    #     0.172,
    #     0.165,
    #     0.162,
    #     0.157,
    #     0.151,
    #     0.148,
    #     0.145,
    #     0.143,
    #     0.140,
    #     0.139,
    #     0.136,
    #     0.135,
    #     0.133,
    #     0.131,
    #     0.128,
    #     0.126,
    #     0.125,
    # ]
    # ao_switch_times = [
    #     12.5,
    #     16,
    #     20,
    #     20,
    #     24,
    #     28,
    #     30,
    #     32,
    #     36,
    #     40,
    #     42,
    #     44,
    #     44,
    #     46,
    #     48,
    #     50,
    #     52,
    #     54,
    #     56,
    #     58,
    #     60,
    #     62,
    # ]
    # rs_switch_times = [
    #     7.45,
    #     8.4,
    #     10,
    #     11.6,
    #     12.9,
    #     14.3,
    #     15.3,
    #     16.5,
    #     17.6,
    #     18.6,
    #     19.8,
    #     21,
    #     22.9,
    #     24.7,
    #     26.4,
    #     27.6,
    #     28.5,
    #     29.4,
    #     30.2,
    #     30.6,
    #     31.7,
    #     32.3,
    # ]
