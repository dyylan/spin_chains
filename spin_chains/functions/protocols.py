from spin_chains.quantum import chains
from spin_chains.quantum import states

import numpy as np


def quantum_communication_exp(
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
            switch_time = 2.5 * switch_time
        else:
            switch_time = 2.5 * switch_time
        chain.update_marked_site(final_site, update_strength)
    else:
        chain.time_evolution(time=switch_time)
        chain.update_marked_site(start_site, -update_strength)
        chain.update_marked_site(final_site, update_strength)

    times, psi_states = chain.time_evolution(time=switch_time, reset_state=False)

    return times, psi_states, final_state, chain


def quantum_communication_exp_noise(
    spins,
    marked_strength,
    switch_time,
    mu=1,
    omega=2 * np.pi * np.array([6e6, 5e6, 1e6]),
    t2=1e-3,
    samples=100,
    open_chain=False,
    start_site=1,
    final_site=15,
    always_on=False,
    dt=0.01,
    gamma_rescale=False,
):
    final_state = states.SingleState(spins, final_site, single_subspace=True)
    init_state = states.SingleState(spins, start_site, single_subspace=True)

    chain = chains.Chain1dSubspaceLongRangeExp(
        spins=spins, dt=dt, mu=mu, omega=omega, t2=t2, samples=samples
    )

    chain.add_marked_site(start_site, marked_strength, gamma_rescale=gamma_rescale)
    chain.initialise(init_state, noisy_evolution=True)

    update_strength = 1 if gamma_rescale else marked_strength

    if always_on:
        if open_chain:
            switch_time = 2.5 * switch_time
        else:
            switch_time = 2.5 * switch_time
        chain.update_marked_site(final_site, update_strength)
    else:
        chain.noisy_time_evolution(time=switch_time)
        chain.update_marked_site(start_site, -update_strength)
        chain.update_marked_site(final_site, update_strength)

    times, psi_states = chain.noisy_time_evolution(time=switch_time, reset_state=False)

    return times, psi_states, final_state, chain
