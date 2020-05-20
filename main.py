from chains import Chain1d
from states import PeriodicState, FourierState
import numpy as np
import matplotlib.pyplot as plt


def state_hist_with_time(states, times):
    pass

def state_hist(state):
    states_x = np.arange(len(state.ket))
    states_y = state.ket
    fig, ax = plt.subplots()
    ax.bar( states_y) 
    ax.set(xlabel='$$')
    ax.grid()
    # plt.savefig()
    plt.show()


def fourier_state_overlaps(spins, period, time):
    init_state = PeriodicState(spins=spins,period=period)
    fourier_states = [FourierState(spins, k) for k in range(2**spins)]

    chain = Chain1d(spins=spins)
    chain.initialise(init_state)
    times, states = chain.time_evolution(time=time)
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

if __name__ == "__main__":
    # init_state = PeriodicState(spins=4,period=2)
    # fourier_states = FourierState(4,0)
    # fourier_state_1 = FourierState(4,1)
    # fourier_state_8 = FourierState(4,8)
    # chain = Chain1d(spins=4)
    # chain.initialise(init_state) 
    # times, states = chain.time_evolution(time=5)
    # overlaps = chain.overlaps_evolution(fourier_state_8.ket, states)
    spins = 10
    period = 5
    time = 8

    fourier_state_overlaps(spins, period, time)
    
    
    
    