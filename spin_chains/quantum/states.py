import itertools
import numpy as np
import scipy.special
import matplotlib.pyplot as plt


class Ket:
    def __init__(self, spins):
        self.subspace_only = False
        self.subspace = -1
        self.spins = spins
        self.dimensions = 2 ** spins

    def update(self, state_ket):
        if len(state_ket) == len(self.subspace_ket):
            if not self.subspace_only:
                self.ket[self._subspace_filter] = np.copy(state_ket)
            self.subspace_ket = np.copy(state_ket)
        elif len(state_ket) == self.dimensions:
            self.ket = np.copy(state_ket)
            if self.subspace >= 0:
                self.subspace_ket = self.ket[self._subspace_filter]
        else:
            raise ValueError(
                "State ket is the wrong length for full space or subspace."
            )

    def local_z_rotation(self, site, angle):
        rotation = np.exp(angle * 1.0j)
        if not self.subspace_only:
            raise ValueError(f"Must be using subspace states for this rotation.")
        self.subspace_ket[self.spins - site] = (
            self.subspace_ket[self.spins - site] * rotation
        )

    def generate_subspace_ket(self, excitations):
        self._generate_subspace(excitations)
        self.subspace_ket = self.ket[self._subspace_filter]
        return self.subspace_ket

    def state_labels_full(self):
        states = [
            "".join([str(i) for i in state])
            for state in itertools.product(range(2), repeat=self.spins)
        ]
        return states

    def state_labels_subspace(self):
        states = [
            "".join([str(i) for i in state]) if self._check_subspace(state) else None
            for state in itertools.product(range(2), repeat=self.spins)
        ]
        return list(filter(None, states))

    def state_barplot(self, subspace=True, save_plot=""):
        if subspace:
            states_x = np.arange(len(self.subspace_ket))
            states_y = self.subspace_ket
            states_y_labels = self.state_labels_subspace()
        else:
            states_x = np.arange(self.dimensions)
            states_y = self.subspace_ket
            states_y_labels = self.state_labels_full()
        fig, ax = plt.subplots()
        ax.bar(states_x, states_y, tick_label=states_y_labels)
        ax.set(xlabel="states")
        ax.grid()
        if save_plot:
            plt.savefig(save_plot)
        plt.show()

    def _generate_subspace(self, subspace):
        self.subspace = subspace
        self._subspace_filter = [
            self._check_subspace(state)
            for state in itertools.product(range(2), repeat=self.spins)
        ]

    # def _return_subspace(self, state):
    #     excites = [np.sum(s) for s in itertools.product(range(2), repeat=self.spins)]
    #     return excites[state]

    def _check_subspace(self, state):
        return np.sum(state) == self.subspace

    @staticmethod
    def single_state(N, k):
        return np.array([1 if i + 1 == k else 0 for i in range(N)], dtype=np.complex128)

    @staticmethod
    def specified_state(state_list):
        return np.array(state_list, dtype=np.complex128)

    @staticmethod
    def random_state(N):
        state = np.random.random(N) + np.random.random(N) * 1j
        return state / np.linalg.norm(state)

    @staticmethod
    def single_excitation_state(spins, excited_spin):
        excited_state = tuple(
            [0] * (excited_spin - 1) + [1] + [0] * (spins - excited_spin)
        )

        state = np.array(
            [
                1 if state == excited_state else 0
                for state in itertools.product(range(2), repeat=spins)
            ],
            dtype=np.complex128,
        )
        return state

    @staticmethod
    def superposition_state(N, spins, subspace, subspace_filter, period, offset):
        dim = int(scipy.special.comb(spins, subspace))
        subspace_ket = np.array(
            [1 if np.isclose(i % period, offset) else 0 for i in range(dim)],
            dtype=np.complex128,
        )
        subspace_ket = 1 / np.sqrt(np.sum(subspace_ket)) * subspace_ket
        ket = np.zeros(N, dtype=np.complex128)
        np.put(ket, subspace_filter, subspace_ket)
        return ket, subspace_ket

    @staticmethod
    def subspace_superposition_state(spins, period=1, offset=0):
        subspace_ket = np.array(
            [1 if np.isclose(i % period, offset) else 0 for i in range(spins)],
            dtype=np.complex128,
        )
        norm = 1 / np.sqrt(np.sum(subspace_ket))
        return norm * subspace_ket

    @staticmethod
    def fourier_state(N, k):
        return (1 / np.sqrt(N)) * np.array(
            [np.exp(1j * 2 * np.pi * k * i / N) for i in range(N)], dtype=np.complex128
        )

    @staticmethod
    def subspace_fourier_state(spins, k, subspace, subspace_filter):
        dim = int(scipy.special.comb(spins, subspace))
        subspace_ket = (1 / np.sqrt(dim)) * np.array(
            [np.exp(1j * 2 * np.pi * k * i / dim) for i in range(dim)],
            dtype=np.complex128,
        )
        ket = np.zeros(N, dtype=np.complex128)
        np.put(ket, subspace_filter, subspace_ket)
        return ket, subspace_ket

    @staticmethod
    def subspace_single_state(spins, k, subspace, subspace_filter):
        dim = int(scipy.special.comb(spins, subspace))
        subspace_ket = np.array(
            [1 if i == k else 0 for i in range(dim)],
            dtype=np.complex128,
        )
        ket = np.zeros(2 ** spins, dtype=np.complex128)
        np.put(ket, subspace_filter, subspace_ket)
        return ket, subspace_ket

    @staticmethod
    def subspace_specified_state(spins, state_list, subspace, subspace_filter):
        dim = int(scipy.special.comb(spins, subspace))
        subspace_ket = np.array(
            state_list,
            dtype=np.complex128,
        )
        ket = np.zeros(2 ** spins, dtype=np.complex128)
        np.put(ket, subspace_filter, subspace_ket)
        return ket, subspace_ket

    @staticmethod
    def periodic_state(spins, period, offset=0):
        periodic_state = tuple(
            [1 if np.isclose(i % period, offset) else 0 for i in range(spins)]
        )
        state = np.array(
            [
                1 if state == periodic_state else 0
                for state in itertools.product(range(2), repeat=spins)
            ],
            dtype=np.complex128,
        )
        return state

    @staticmethod
    def two_particle_hubbard_state(spins, sites):
        state = np.array(
            [
                1 if i + 1 == sites[0] and j + 1 == sites[1] else 0
                for i in range(spins)
                for j in range(spins)
            ],
            dtype=np.complex128,
        )
        return state

    @staticmethod
    def n_particle_hubbard_state(spins, excitations, sites=[], state=[]):
        if state:
            states = [
                np.array(state)
                for state in itertools.product(range(excitations + 1), repeat=spins)
                if np.sum(state) == excitations
            ]
            state = np.array(
                [1 if state == s else 0 for s in states],
                dtype=np.complex128,
            )
        else:
            state = np.zeros(spins)
            for site in sites:
                state[site - 1] += 1
        return state


class SuperpositionState(Ket):
    def __init__(self, spins, subspace=1, period=1, offset=0, single_subspace=False):
        super().__init__(spins)
        if single_subspace and subspace != 1:
            raise ValueError("In single_subspace is True, subspace must be 1.")
        self.period = period
        self.offset = offset
        self.excitations = subspace
        self.subspace_only = single_subspace
        if self.subspace_only:
            self.subspace = 1
            self.subspace_ket = Ket.subspace_superposition_state(
                self.spins, self.period, self.offset
            )
        else:
            self._generate_subspace(subspace)
            self.ket, self.subspace_ket = Ket.superposition_state(
                self.dimensions,
                self.spins,
                self.subspace,
                self._subspace_filter,
                self.period,
                self.offset,
            )


class FourierState(Ket):
    def __init__(self, spins, k, single_subspace=False):
        super().__init__(spins)
        self.k = k
        self.subspace_only = single_subspace
        if self.subspace_only:
            self.subspace = 1
            self.subspace_ket = Ket.fourier_state(self.spins, self.k)
        else:
            self.ket = Ket.fourier_state(self.dimensions, self.k)


class SubspaceFourierState(Ket):
    def __init__(self, spins, k, subspace):
        super().__init__(spins)
        self.k = k
        self.excitations = subspace
        self._generate_subspace(subspace)
        self.ket, self.subspace_ket = Ket.subspace_fourier_state(
            self.dimensions, self.spins, self.k, self.subspace, self._subspace_filter
        )


class PeriodicState(Ket):
    def __init__(self, spins, period, offset=0):
        super().__init__(spins)
        self.period = period
        self.offset = offset
        self.excitations = self.spins / self.period
        if not self.excitations.is_integer():
            raise ValueError(f"Period ({period}) must be a factor of spins ({spins})!")
        self.ket = Ket.periodic_state(self.spins, self.period, self.offset)
        self.generate_subspace_ket(self.excitations)


class SingleExcitationState(Ket):
    def __init__(self, spins, excited_spin):
        super().__init__(spins)
        self.excited_spin = excited_spin
        self.excitations = 1
        self.ket = Ket.single_excitation_state(self.spins, self.excited_spin)
        self.generate_subspace_ket(self.excitations)


class SingleState(Ket):
    def __init__(self, spins, state, excitations=1, single_subspace=False):
        super().__init__(spins)
        self.state = state
        self.subspace_only = single_subspace
        if self.subspace_only:
            self.subspace = 1
            self.subspace_ket = Ket.single_state(
                self.spins, self.spins - self.state + 1
            )
        else:
            self.excitations = excitations
            self._generate_subspace(self.excitations)
            self.ket, self.subspace_ket = Ket.subspace_single_state(
                self.spins, self.state, self.subspace, self._subspace_filter
            )


class SpecifiedState(Ket):
    def __init__(self, spins, state_list, excitations, single_subspace=False):
        super().__init__(spins)
        self.state = state_list
        self.subspace_only = single_subspace
        if self.subspace_only:
            self.subspace = 1
            self.subspace_ket = Ket.specified_state(self.spins, state_list)
        else:
            self.excitations = excitations
            self._generate_subspace(self.excitations)
            self.ket, self.subspace_ket = Ket.subspace_specified_state(
                self.spins, state_list, self.subspace, self._subspace_filter
            )


class RandomState(Ket):
    def __init__(self, spins, single_subspace=False):
        super().__init__(spins)
        if single_subspace:
            self.subspace_ket = Ket.random_state(self.spins)
        else:
            self.ket = Ket.random_state(self.dimensions)


class TwoParticleHubbbardState(Ket):
    def __init__(self, spins, sites=[1, 1], state_array=False):
        super().__init__(spins)
        self.subspace = 2
        self.subspace_only = True
        if state_array:
            self.subspace_ket = np.array(state_array, dtype=np.complex128)
        else:
            self.subspace_ket = Ket.two_particle_hubbard_state(self.spins, sites)

    def state_labels_subspace(self):
        spin_locations = [
            (spin1, spin2) for spin1 in range(self.spins) for spin2 in range(self.spins)
        ]
        states = []
        for locs in spin_locations:
            state = [0 for _ in range(self.spins)]
            state[locs[0]] += 1
            state[locs[1]] += 1
            states.append("".join([str(i) for i in state]))
        return states


class NParticleHubbbardState(Ket):
    def __init__(self, spins, excitations, state, state_array=False):
        super().__init__(spins)
        self.subspace = 2
        self.subspace_only = True
        self.excitations = excitations
        if state_array:
            self.subspace_ket = np.array(state_array, dtype=np.complex128)
        else:
            self.subspace_ket = Ket.n_particle_hubbard_state(
                self.spins, self.excitations, state
            )

    def state_labels_subspace(self):
        states = [
            "".join([str(i) for i in state])
            for state in itertools.product(
                range(self.excitations + 1), repeat=self.spins
            )
            if np.sum(state) == self.excitations
        ]
        return states
