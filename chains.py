import numpy as np
from hamiltonians import HeisenbergHamiltonian
from states import FourierState


class Chain:

    def __init__(self, spins):
        self.n = spins


class Chain1d(Chain):

    def __init__(self, spins, dt=0.1, jx=1, jy=1, jz=0, h=0, approx=False):
        super().__init__(spins)
        self.hamiltonian = HeisenbergHamiltonian(spins, dt, jx, jy, jz, h, approx)

    def initialise(self, state, subspace_evolution=True):
        if self.hamiltonian.spins != state.spins:
            raise ValueError(f'Hamiltonian number of spins, {self.hamiltonian.spins}, '
                             f'must equal state spins, {state.spins}')
        self.initial_state = state
        self.state = state
        self.subspace_evolution = subspace_evolution
        if subspace_evolution:
            self.hamiltonian.initialise_subspace(state.subspace)

    def time_evolution(self, time, reset_state=True):
        initial_ket = np.copy(self.state.ket)
        t = 0
        states = []
        times = []
        if self.subspace_evolution and (self.state.subspace >= 0):
            unitary = self.hamiltonian.U_subspace
            ket = self.state.subspace_ket
        else:
            unitary = self.hamiltonian.U
            ket = self.state.ket
        while t < time:
            state_ket = np.matmul(unitary, ket)
            states.append(state_ket)
            times.append(t)
            self.state.update(state_ket) 
            t += self.hamiltonian.dt
        if reset_state:
            self.state.update(initial_ket)
        return times, states

    @staticmethod
    def overlaps_evolution(bra, states, norm=True):
        overlaps = [np.vdot(bra, state) for state in states]
        if norm:
            overlaps = np.abs(np.multiply(np.conj(overlaps), overlaps))
        return overlaps