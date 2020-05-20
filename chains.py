import numpy as np
from hamiltonians import HeisenbergHamiltonian
from states import FourierState


class Chain:

    def __init__(self, spins):
        self.n = spins


class Chain1d(Chain):

    def __init__(self, spins, dt=0.1, jx=1, jy=1, jz=1):
        super().__init__(spins)
        self.hamiltonian = HeisenbergHamiltonian(spins, dt, jx, jy, jz)

    def initialise(self, state):
        self.initial_state = state
        self.state = state

    def time_evolution(self, time, reset=True):
        initial_ket = np.copy(self.state.ket)
        t = 0
        states = []
        times = []
        while t < time:
            state_ket = np.matmul(self.hamiltonian.U, self.state.ket)
            states.append(state_ket)
            times.append(t)
            self.state.update(state_ket) 
            t += self.hamiltonian.dt
        if reset: 
            self.state.update(initial_ket)
        return times, states

    @staticmethod
    def overlaps_evolution(bra, states, norm=True):
        overlaps = [np.vdot(bra, state) for state in states]
        if norm:
            overlaps = np.abs(np.multiply(np.conj(overlaps), overlaps))
        return overlaps