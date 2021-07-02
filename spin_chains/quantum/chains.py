from numba import jit
import numpy as np
import scipy.constants
import copy
from .hamiltonians import (
    HeisenbergHamiltonian,
    LongRangeXYHamiltonian,
    XYHamiltonian1dSubspace,
    HubbardHamiltonian2particles,
    LongRangeXYHamiltonian1d,
    LongRangeXYHamiltonian1dExp,
)
from .states import FourierState


class Chain:
    def __init__(self, spins):
        self.initialised = False
        self.spins = spins
        self.marked_sites = []
        self.states = []
        self.times = []
        self.time = 0

    def add_marked_site(self, spin, h=1, field="z", gamma_rescale=False):
        if self.initialised:
            raise ValueError(f"Must add marked state before initilising the Chain.")
        self.marked_sites.append((spin, h, field))
        self.hamiltonian.add_marked_site(spin, h, field, gamma_rescale)

    def update_marked_site(self, spin, h=1, field="z"):
        raise NotImplementedError(f"Not implemented for this Chain.")

    def time_evolution(self, time, reset_state=True):
        time = time / scipy.constants.hbar if self.hamiltonian.hbar else time
        if not self.initialised:
            raise ValueError(f"Must initilise the Chain before time evolution.")
        self.hamiltonian.initialise_evolution(self.subspace_evolution)
        t = 0
        unitary = self.hamiltonian.U

        if self.subspace_evolution and (self.state.subspace >= 0):
            initial_ket = self.initial_state.subspace_ket
        else:
            initial_ket = self.initial_state.ket

        if reset_state:
            self.state.update(initial_ket)
            self.states = []
            self.times = []
            self.time = 0

        if self.subspace_evolution and (self.state.subspace >= 0):
            ket = self.state.subspace_ket
        else:
            ket = self.state.ket

        while t < time:
            state_ket = np.matmul(unitary, ket)
            self.states.append(state_ket)
            self.times.append(self.time + t)
            self.state.update(state_ket)
            t += self.hamiltonian.dt
            ket = state_ket

        self.time += np.copy(t)
        return self.times, self.states

    def noisy_time_evolution(self, time, reset_state=True):
        if not self.initialised:
            raise ValueError(f"Must initilise the Chain before time evolution.")
        if not self.noisy_evolution:
            raise ValueError(
                f"Must initialise a noisy evolution for a noisy time evolution!"
            )
        self.hamiltonian.initialise_evolution(
            self.subspace_evolution, samples=self.hamiltonian.samples
        )
        if reset_state:
            self.noisy_states = [[] for _ in range(self.hamiltonian.samples)]
            self.times = []
            self.time = 0
        for i, state in enumerate(self.state_samples):
            t = 0
            states = []
            times = []
            unitary = self.hamiltonian.U[i]
            if self.subspace_evolution and (state.subspace >= 0):
                initial_ket = self.initial_state.subspace_ket
            else:
                initial_ket = self.initial_state.ket

            if reset_state:
                state.update(initial_ket)

            if self.subspace_evolution and (state.subspace >= 0):
                ket = state.subspace_ket
            else:
                ket = state.ket

            while t < time:
                state_ket = np.matmul(unitary, ket)
                self.noisy_states[i].append(state_ket)
                times.append(t + self.time)
                t += self.hamiltonian.dt
                ket = state_ket
            state.update(state_ket)

        self.time += np.copy(t)
        self.times.extend(times)
        return self.times, self.noisy_states

    def _check_state(self, state):
        if self.hamiltonian.spins != state.spins:
            raise ValueError(
                f"Hamiltonian number of spins, {self.hamiltonian.spins}, "
                f"must equal state spins, {state.spins}"
            )

    @staticmethod
    def overlaps_evolution(bra, states, norm=True):
        overlaps = np.array(
            [Chain._compute_overlap(bra, state) for state in states],
            dtype=np.complex128,
        )
        if norm:
            # overlaps = np.abs(np.multiply(np.conj(overlaps), overlaps))
            overlaps = Chain._square_norm(overlaps)
        return overlaps

    @staticmethod
    def overlaps_noisy_evolution(bra, noisy_states, norm=True):
        overlaps = [
            Chain.overlaps_evolution(bra, sample, norm=False) for sample in noisy_states
        ]
        mean_overlaps = np.mean(overlaps, axis=0)
        if norm:
            mean_overlaps = Chain._square_norm(mean_overlaps)
        return mean_overlaps

    @staticmethod
    @jit(nopython=True)
    def _compute_overlap(bra, ket):
        return np.vdot(bra, ket)

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _square_norm(overlaps):
        return np.abs(np.multiply(np.conj(overlaps), overlaps))


class Chain1d(Chain):
    def __init__(
        self,
        spins,
        dt=0.1,
        jx=1,
        jy=1,
        jz=0,
        h_field=0,
        h_field_axis="z",
        open_chain=False,
    ):
        super().__init__(spins)
        self.hamiltonian = HeisenbergHamiltonian(
            spins, dt, jx, jy, jz, h_field, h_field_axis, open_chain
        )

    def initialise(self, state, subspace_evolution=True):
        self._check_state(state)
        self.initial_state = state
        self.initialised = True
        self.state = state
        self.subspace_evolution = subspace_evolution
        if subspace_evolution:
            self.hamiltonian.initialise_subspace(state.subspace)


class Chain1dLongRange(Chain):
    def __init__(self, spins, dt=0.1, alpha=1, open_chain=False):
        super().__init__(spins)
        self.hamiltonian = LongRangeXYHamiltonian(spins, dt, alpha, open_chain)

    def initialise(self, state, subspace_evolution=True):
        self._check_state(state)
        self.initial_state = state
        self.initialised = True
        self.state = state
        self.subspace_evolution = subspace_evolution
        if subspace_evolution:
            self.hamiltonian.initialise_subspace(state.subspace)


class Chain1dSubspace(Chain):
    def __init__(self, spins, dt=0.1, js=1, open_chain=False):
        super().__init__(spins)
        self.hamiltonian = XYHamiltonian1dSubspace(spins, dt, js, open_chain)

    def add_marked_site(self, spin, h=1, gamma_rescale=False):
        if self.initialised:
            raise ValueError(
                f"Must add marked state before initilising the Chain. Try 'update_marked_state()' instead"
            )
        self.marked_sites.append((spin, h))
        self.hamiltonian.add_marked_site(spin, h, gamma_rescale=gamma_rescale)

    def update_marked_site(self, spin, h=1):
        self.marked_sites.append((spin, h))
        self.hamiltonian.update_marked_site(spin, h)

    def update_js(self, js):
        self.hamiltonian.update_js(js)

    def initialise(self, state, subspace_evolution=True, noisy_evolution=False):
        self._check_state(state)
        self.initial_state = state
        self.initialised = True
        self.state = state
        self.subspace_evolution = subspace_evolution
        self.noisy_evolution = noisy_evolution
        if self.noisy_evolution:
            if not self.hamiltonian.samples:
                raise ValueError(
                    f"Must create a chain with non-zero samples for noisy evolution!"
                )
            self.state_samples = [
                copy.copy(state) for _ in range(self.hamiltonian.samples)
            ]
            self.state = self.state_samples
            self.noisy_states = [[] for _ in range(self.hamiltonian.samples)]

    def local_z_rotation(self, site, angle=np.pi / 2):
        if not self.subspace_evolution:
            raise ValueError(f"Must be using subspace evolution for this rotation!")
        if self.noisy_evolution:
            for i in range(self.hamiltonian.samples):
                self.state_samples[i].local_z_rotation(site, angle)
        else:
            self.state.local_z_rotation(site, angle)


class Chain1dSubspace2particles(Chain1dSubspace):
    def __init__(self, spins, dt=0.1, js=1, us=1, es=0, vs=0, open_chain=False):
        super().__init__(spins)
        self.hamiltonian = HubbardHamiltonian2particles(
            spins, dt, js, us, es, vs, open_chain
        )

    def add_marked_site(self, spin, h=1, gamma_rescale=False):
        raise NotImplementedError(f"Not implemented for this Chain.")

    def update_marked_site(self, spin, h=1):
        raise NotImplementedError(f"Not implemented for this Chain.")


class Chain1dSubspaceLongRange(Chain1dSubspace):
    def __init__(
        self,
        spins,
        dt=0.1,
        alpha=1,
        open_chain=False,
        sender_reciever_qubits=False,
        hamiltonian_scaling=1,
        interaction_map="all",
        interaction_distance_bound=0,
        noise=0,
        samples=0,
    ):
        super().__init__(spins)
        self.hamiltonian = LongRangeXYHamiltonian1d(
            spins,
            dt,
            alpha,
            open_chain,
            sender_reciever_qubits,
            hamiltonian_scaling,
            interaction_map,
            interaction_distance_bound,
            noise,
            samples,
        )

    def update_interaction_map(self, interaction_map, interaction_distance_bound=0):
        self.hamiltonian.update_interaction_map(
            interaction_map, interaction_distance_bound
        )


class Chain1dSubspaceLongRangeExp(Chain1dSubspace):
    def __init__(
        self,
        spins,
        dt=0.000001,
        mu=2 * np.pi * 6.01e6,
        Omega=2 * np.pi * 1e6,
        delta_k=2 * 2 * np.pi / 355e-9 * np.array([1, 0, 0]),
        omega=2 * np.pi * np.array([6e6, 5e6, 1e6]),
        use_optimal_omega=False,
        t2=10e-3,
        hbar=False,
    ):
        super().__init__(spins)
        self.hamiltonian = LongRangeXYHamiltonian1dExp(
            spins,
            dt=dt,
            mu=mu,
            Omega=Omega,
            delta_k=delta_k,
            omega=omega,
            use_optimal_omega=use_optimal_omega,
            t2=10e-3,
            hbar=hbar,
        )


class Chain1dSubspace2LongRange(Chain1dSubspace):
    def __init__(
        self, spins, dt=0.1, alpha=1, open_chain=False, sender_reciever_qubits=False
    ):
        super().__init__(spins)
        self.hamiltonian = LongRangeXYHamiltonian1d(
            spins, dt, alpha, open_chain, sender_reciever_qubits
        )
