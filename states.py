import itertools
import numpy as np
import matplotlib.pyplot as plt


class Ket:

    def __init__(self, spins):
        self.spins = spins
        self.dimensions = 2**spins

    def update(self, state_ket):
        self.ket = np.copy(state_ket)

    def state_labels(self):
        return [''.join([str(i) for i in state]) 
                for state in itertools.product(range(2), repeat=self.spins)]

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

    @staticmethod
    def single_excitation_state(spins, excited_spin):
        excited_state = tuple([1]+[0 for _ in range(1,spins)])
        excited_states = [
            state for state in itertools.permutations(excited_state,spins)
        ]
        state = np.array([
            1 if state in excited_states and np.isclose((state.index(1)+1)/excited_spin, 1) else 0 
            for state in itertools.product(range(2), repeat=spins)
        ])
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
        ])
        return state

    @staticmethod
    def fourier_state(N, k):
        return (1/np.sqrt(N))*np.array([np.exp(1j*2*np.pi*k*i/N) for i in range(N)])

    @staticmethod
    def periodic_state(spins, period):
        periodic_state = tuple([1 if np.isclose(i % period, 0) else 0 for i in range(spins)])
        state = np.array([
            1 if state == periodic_state else 0 
            for state in itertools.product(range(2), repeat=spins)
        ])
        return state


class SuperpositionState(Ket):

    def __init__(self, spins):
        super().__init__(spins)
        self.ket = Ket.superposition_state(self.dimensions)


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

 
class SingleExcitationState(Ket):

    def __init__(self, spins, excited_spin):
        super().__init__(spins)
        self.excited_spin = excited_spin
        self.ket = Ket.single_excitation_state(spins, excited_spin)