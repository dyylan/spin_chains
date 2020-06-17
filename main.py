from chains import Chain1d
from states import PeriodicState, FourierState, SingleExcitationState, SuperpositionState, SingleState
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt


def state_hist_with_time(states, times):
    pass

def state_hist(state):
    states_x = np.arange(len(state.ket))
    states_y = state.ket
    fig, ax = plt.subplots()
    ax.bar(states_y, states_x) 
    ax.set(xlabel='$$')
    ax.grid()
    # plt.savefig()
    plt.show()


def state_overlaps(spins, state, time):
    init_state = SingleState(spins, state)
    overlap_state = SingleState(spins, state)
    init_state.state_barplot()
    overlap_state.state_barplot()
    chain = Chain1d(spins=spins, approx=False, dt=0.01, jx=1, jy=1, jz=0, h=0)
    chain.initialise(init_state)
    times, states = chain.time_evolution(time=time)
    # print(chain.hamiltonian.H)
    overlaps = chain.overlaps_evolution(overlap_state.ket, states)
    fig, ax = plt.subplots()
    for y in [overlaps]:
        ax.plot(times, y)
    # ax.legend([f'k = {k}' for k in range(2**spins)])
    ax.set(xlabel='$time~(s/\hbar)$')
    ax.grid()
    plt.show()


def fourier_state_overlaps(spins, period, time):
    init_state = SuperpositionState(spins=spins,period=period)
    init_state = PeriodicState(spins=spins,period=period)
    # init_state = FourierState(spins, 1)
    fourier_states = [FourierState(spins, k) for k in range(2**spins)]

    chain = Chain1d(spins=spins, approx=False, dt=0.01, jx=1, jy=1, jz=0, h=0)
    chain.initialise(init_state)
    times, states = chain.time_evolution(time=time)
    # print(chain.hamiltonian.H)
    overlaps = []
    for k, fourier_state in enumerate(fourier_states):
        overlaps.append(chain.overlaps_evolution(fourier_state.ket, states))
        print(f'Computed overlaps for fourier state k = {k}')
    fig, ax = plt.subplots()
    for y in overlaps:
        ax.plot(times, y)
    # ax.legend([f'k = {k}' for k in range(2**spins)])
    ax.set(xlabel='$time~(s/\hbar)$')
    ax.grid()
    plt.show()


def quantum_communication(spins, start_state, end_state):
    init_state = SingleExcitationState(spins, start_state)
    final_state = SingleExcitationState(spins, end_state)
    print(init_state.ket)
    print(final_state.ket)
    chain = Chain1d(spins=spins, approx=False)
    chain.initialise(init_state)
    times, states = chain.time_evolution(time=time)
    overlaps = chain.overlaps_evolution(final_state.ket, states)
    fig, ax = plt.subplots()
    ax.plot(times, overlaps)
    ax.legend([f'final state overlap'])
    ax.set(xlabel='$time~(s/\hbar)$')
    ax.grid()
    plt.show()


if __name__ == "__main__":
    # init_state = PeriodicState(spins=4,period=2)
    # fourier_states = FourierState(4,0)
    # fourier_state_1 = FourierState(4,1)
    # fourier_state_8 = FourierState(4,8)
    # chain = Chain1d(spins=4)
    # chain.initialise(init_state) 
    # times, states = chain.time_evolution(time=5)
    # overlaps = chain.overlaps_evolution(fourier_state_8.ket, states)
    spins = 8
    period = 2
    time = 10

    # fourier_state_overlaps(spins, period, time)
    # quantum_communication(spins, 1, 1)
    init_state = PeriodicState(spins=4,period=2)
    chain = Chain1d(spins=4)
    chain.initialise(init_state)
    times, states = chain.time_evolution(time=0.1)
    
    