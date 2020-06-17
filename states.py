import itertools
import numpy as np
import matplotlib.pyplot as plt


class Ket:

    def __init__(self, spins):
        self.subspace=-1
        self.spins = spins
        self.dimensions = 2**spins

    def update(self, state_ket):
        if len(state_ket) == len(self.ket):
            self.ket = np.copy(state_ket)
            if self.subspace >= 0:
                self.subspace_ket = self.ket[self.subspace_filter]
        elif len(state_ket) == len(self.subspace_ket):
            print(self.ket)
            print(self.subspace_filter)
            print(state_ket)
            self.ket[self.subspace_filter] = np.copy(state_ket)
            print(self.ket)
            self.subspace_ket = np.copy(state_ket)
        else:
            raise ValueError('State ket is the wrong length for full space or subspace.')

    def generate_subspace_ket(self, excitations):
        self.subspace = excitations
        self.subspace_filter = [self._check_subspace(state)
                for state in itertools.product(range(2), repeat=self.spins)]
        self.subspace_ket = self.ket[self.subspace_filter]
        return self.subspace_ket

    def state_labels_full(self):
        states = [''.join([str(i) for i in state]) 
                  for state in itertools.product(range(2), repeat=self.spins)]
        return states

    def state_labels_subspace(self):
        states = [''.join([str(i) for i in state]) if self._check_subspace(state) else None
                  for state in itertools.product(range(2), repeat=self.spins)]
        return list(filter(None, states))

    def state_barplot(self, save_plot=''):
        states_x = np.arange(self.dimensions)
        states_y = self.ket
        states_y_labels = self.state_labels()
        fig, ax = plt.subplots()
        ax.bar(states_x, states_y, tick_label=states_y_labels) 
        ax.set(xlabel='states')
        ax.grid()
        if save_plot:
            plt.savefig(save_plot)
        plt.show()

    def _check_subspace(self, state):
        return np.sum(state) == self.subspace 

    @staticmethod
    def single_state(N, k):
        return np.array([1 if i == k else 0 for i in range(N)], dtype=complex)

    @staticmethod
    def single_excitation_state(spins, excited_spin):
        excited_state = tuple([1]+[0 for _ in range(1,spins)])
        excited_states = [
            state for state in itertools.permutations(excited_state,spins)
        ]
        state = np.array([
            1 if state in excited_states and np.isclose((state.index(1)+1)/excited_spin, 1) else 0 
            for state in itertools.product(range(2), repeat=spins)
        ], dtype=complex)
        return state

    @staticmethod
    def superposition_state(spins, period):
        excited_state = tuple([1]+[0 for _ in range(1,spins)])
        excited_states = [
            state for state in itertools.permutations(excited_state,spins)
        ]
        state = (1/np.sqrt(spins/period))*np.array([
            1 if state in excited_states and np.isclose(state.index(1) % period, 0) else 0 
            for state in itertools.product(range(2), repeat=spins)
        ], dtype=complex)
        return state

    @staticmethod
    def fourier_state(N, k):
        return (1/np.sqrt(N))*np.array([np.exp(1j*2*np.pi*k*i/N) for i in range(N)], dtype=complex)

    @staticmethod
    def periodic_state(spins, period):
        periodic_state = tuple([1 if np.isclose(i % period, 0) else 0 for i in range(spins)])
        state = np.array([
            1 if state == periodic_state else 0 
            for state in itertools.product(range(2), repeat=spins)
        ], dtype=complex)
        return state


class SuperpositionState(Ket):

    def __init__(self, spins, period):
        super().__init__(spins)
        self.period = period
        self.excitations = 1
        self.ket = Ket.superposition_state(self.spins, self.period)
        self.generate_subspace_ket(self.excitations)


class FourierState(Ket):

    def __init__(self, spins, k):
        super().__init__(spins)
        self.k = k
        self.ket = Ket.fourier_state(self.dimensions, self.k)


class PeriodicState(Ket):

    def __init__(self, spins, period):
        super().__init__(spins)
        self.period = period
        self.excitations = self.spins/self.period
        if not self.excitations.is_integer():
            raise ValueError(f'Period ({period}) must be a factor of spins ({spins})!')
        self.ket = Ket.periodic_state(self.spins, self.period) 
        self.generate_subspace_ket(self.excitations)

 
class SingleExcitationState(Ket):

    def __init__(self, spins, excited_spin):
        super().__init__(spins)
        self.excited_spin = excited_spin
        self.excitations = 1
        self.ket = Ket.single_excitation_state(self.spins, self.excited_spin)
        self.generate_subspace_ket(self.excitations)


class SingleState(Ket):

    def __init__(self, spins, state):
        super().__init__(spins)
        self.state = state
        self.ket = Ket.single_state(self.dimensions, self.state)