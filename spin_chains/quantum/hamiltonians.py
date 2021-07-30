from numba import jit
import numpy as np
import itertools
import numpy.linalg
import scipy.linalg
import scipy.special
import scipy.constants

from .iontrap import IonTrap


class Hamiltonian:
    def __init__(self, dimensions, dt, hbar=False):
        self.dimensions = dimensions
        self.hbar = hbar
        self.dt = dt / scipy.constants.hbar if hbar else dt

    def initialise_evolution(self, subspace_evolution=False, samples=0):
        if samples:
            if subspace_evolution:
                self.U = []
                for sample in range(samples):
                    self.U.append(
                        Hamiltonian.U_matrix(self.H_noise_subspace[sample], self.dt)
                    )
            else:
                self.U = []
                for sample in range(samples):
                    self.U.append(Hamiltonian.U_matrix(self.H_noise[sample], self.dt))
        else:
            if subspace_evolution:
                self.U = Hamiltonian.U_matrix(self.H_subspace, self.dt)
            else:
                self.U = Hamiltonian.U_matrix(self.H, self.dt)

    def add_marked_site(self, spin, h=1, field="z", gamma_rescale=False):
        site = self.spins - spin
        if gamma_rescale:
            self.H = h * self.H
            H_marked = (-1 / 2) * Hamiltonian.multikron(
                Hamiltonian.pauli_product(field, self.spins, site)
            )
        else:
            H_marked = (-h / 2) * Hamiltonian.multikron(
                Hamiltonian.pauli_product(field, self.spins, site)
            )
        self.H = self.H + H_marked

    def update_marked_site(self, spin, h=1, field="z"):
        raise NotImplementedError(f"Not implemented for this Hamiltonian.")

    def _set_couplings(self, j):
        if type(j) is not list:
            j_list = [j] * self.spins
        else:
            if len(j) != self.spins:
                raise ValueError(
                    f"List of couplings must be the same length as number of spins."
                )
            else:
                j_list = j
        return j_list

    @staticmethod
    def s_parameter(eigenvalues, q=1, open_chain=True, eigenvectors=[]):
        """Returns the s_q paramter of the Hamiltonian."""

        def summand(max_eigenvalue, eigenvalue, coef):
            return coef / (np.power((max_eigenvalue - eigenvalue), q))

        largest_eigenvalue = eigenvalues[0]
        n = len(eigenvalues)

        if open_chain:
            if not eigenvectors.any():
                raise ValueError(
                    f"If graph is not vertex transitive, the eigenvectors are required."
                )
            else:
                summation = np.array(
                    [
                        summand(
                            largest_eigenvalue,
                            eigenvalues[i],
                            np.abs(eigenvectors[0, i]) ** 2,
                        )
                        for i in range(1, n, 1)
                    ]
                )
                s_q = np.sum(summation)
        else:
            summation = np.array(
                [summand(largest_eigenvalue, eigenvalues[i], 1) for i in range(1, n, 1)]
            )
            s_q = np.sum(summation) / n
        return s_q

    @staticmethod
    def U_matrix(H, dt):
        return scipy.linalg.expm(-1j * H * dt)

    @staticmethod
    @jit(nopython=True, parallel=True)
    def spectrum(hamiltonian):
        """Returns sorted eigenvalues and eigenvectors."""
        eigenvalues, eigenvectors = numpy.linalg.eigh(hamiltonian)
        index = np.argsort(-eigenvalues)  # largest eigenvalue to smallest
        eigenvectors = eigenvectors[:, index]
        eigenvalues = eigenvalues[index]
        return eigenvectors, eigenvalues

    @staticmethod
    # @jit(nopython=True, parallel=True)
    def multikron(matrices):
        product = matrices[0]
        for mat in matrices[1:]:
            product = np.kron(product, mat)
        return product

    @staticmethod
    def pauli_product(axis, n, j):
        paulis = [
            Hamiltonian.pauli(axis) if i == j else Hamiltonian.pauli(0)
            for i in range(n)
        ]
        return paulis

    @staticmethod
    def pauli(axis):
        if axis in [0, "i"]:
            sigma = np.array([[1, 0], [0, 1]])
        elif axis in [1, "x"]:
            sigma = np.array([[0, 1], [1, 0]])
        elif axis in [2, "y"]:
            sigma = np.array([[0, -1j], [1j, 0]])
        elif axis in [3, "z"]:
            sigma = np.array([[1, 0], [0, -1]])
        return sigma


class HeisenbergHamiltonian(Hamiltonian):
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
        dimensions = 2 ** spins
        super().__init__(dimensions, dt)
        self.spins = spins
        self.jx = self._set_couplings(jx)
        self.jy = self._set_couplings(jy)
        self.jz = self._set_couplings(jz)
        self.js = {"x": self.jx, "y": self.jy, "z": self.jz}
        self.h_field = h_field
        self.h_field_axis = h_field_axis
        self.open_chain = open_chain
        self.H = self.H_matrix()

    def initialise_subspace(self, subspace):
        self.transition_map = self.generate_transition_map()
        self.excitation_map = self.generate_excitation_map()
        if subspace >= 0:
            self.H_subspace = self.subspace_H_matrix(subspace)

    def H_matrix(self):
        H = (1 / 2) * np.sum(
            [
                self.js[axis][j]
                * np.matmul(
                    Hamiltonian.multikron(
                        Hamiltonian.pauli_product(axis, self.spins, j)
                    ),
                    Hamiltonian.multikron(
                        Hamiltonian.pauli_product(
                            axis, self.spins, (j + 1) % self.spins
                        )
                    ),
                )
                for j in range(self.spins - self.open_chain)
                for axis in ["x", "y", "z"]
            ],
            axis=0,
        ) + np.sum(
            [
                self.h_field
                * Hamiltonian.multikron(
                    Hamiltonian.pauli_product(self.h_field_axis, self.spins, j)
                )
                for j in range(self.spins)
            ],
            axis=0,
        )
        return H

    def subspace_H_matrix(self, excitations):
        dim = range(len(self.excitation_map))
        subspace_H = np.array(
            [
                [
                    self.excitation_map[row][col] == (excitations, excitations)
                    for col in dim
                ]
                for row in dim
            ]
        )
        subspace_size = scipy.special.comb(self.spins, excitations, exact=True)
        subspace_H = np.reshape(self.H[subspace_H], (subspace_size, subspace_size))
        return subspace_H

    def generate_transition_map(self):
        states = [
            "".join([str(i) for i in state])
            for state in itertools.product(range(2), repeat=self.spins)
        ]
        t_map = [
            [(states[i], states[j]) for j in range(len(states))]
            for i in range(len(states))
        ]
        return t_map

    def generate_excitation_map(self):
        excites = [
            np.sum(state) for state in itertools.product(range(2), repeat=self.spins)
        ]
        e_map = [
            [(excites[i], excites[j]) for j in range(len(excites))]
            for i in range(len(excites))
        ]
        return e_map


class XYHamiltonian1dSubspace(Hamiltonian):
    def __init__(self, spins, dt=0.1, js=1, open_chain=False):
        super().__init__(spins, dt)
        self.spins = spins
        self.js = self._set_couplings(js)
        self.open_chain = open_chain
        self.H = self.H_matrix()
        self.H_subspace = self.H

    def H_matrix(self):
        if self.open_chain:

            def interaction(spinA, spinB):
                if spinA - spinB == 1:
                    return self.js[spinB - 1]
                if spinB - spinA == 1:
                    return self.js[spinA - 1]
                return 0

        else:

            def interaction(spinA, spinB):
                if spinA - (spinB % self.spins) == 1:
                    return self.js[spinB - 1]
                if spinB - (spinA % self.spins) == 1:
                    return self.js[spinA - 1]
                return 0

        H = np.array(
            [
                [interaction(spinA + 1, spinB + 1) for spinA in range(self.spins)]
                for spinB in range(self.spins)
            ]
        )
        return H

    def add_marked_site(self, spin, hz=1, gamma_rescale=False):
        site = self.spins - spin
        if gamma_rescale:
            self.H = hz * self.H
            self.H[(site, site)] = self.H[(site, site)] + 1
        else:
            self.H[(site, site)] = self.H[(site, site)] + hz
        self.H_subspace = self.H

    def update_marked_site(self, spin, hz=1):
        self.add_marked_site(spin, hz)

    def update_js(self, js):
        self.js = self._set_couplings(js)
        self.H = self.H_matrix()
        self.H_subspace = self.H

    def add_broken_spin(self, spin, t2):
        raise NotImplementedError(f"Not implemented for this Hamiltonian.")


class HubbardHamiltonian2particles(Hamiltonian):
    def __init__(self, spins, dt=0.1, js=1, us=1, es=0, vs=0, open_chain=False):
        dimensions = spins ** 2
        super().__init__(dimensions, dt)
        self.spins = spins
        self.js = self._set_couplings(js)
        self.us = self._set_interactions_list(us)
        self.es = self._set_interactions_list(es)
        self.vs = self._set_interactions_list(vs)
        self.open_chain = open_chain
        self.H = self.H_matrix()
        self.H_subspace = self.H

    def H_matrix(self):
        if self.open_chain:

            def hopping(i, j, a, b):
                if a == b:
                    if i - j == 1:
                        return self.js[j - 1]
                    if j - i == 1:
                        return self.js[i - 1]
                return 0

        else:

            def hopping(i, j, a, b):
                if a == b:
                    if i - (j % self.spins) == 1:
                        return self.js[j - 1]
                    if j - (i % self.spins) == 1:
                        return self.js[i - 1]
                return 0

        def u_interaction(i, j, a, b, strength):
            interaction = strength if i == j == a == b else 0
            return interaction

        def e_onsite_potential(i, j, a, b, strength):
            interaction = strength if (i == a) and (j == b) else 0
            return interaction

        def v_interaction(i, j, a, b, strength, l=1):
            interaction = strength if (i == a) and (j == b) and (abs(i - j) == l) else 0
            return interaction

        H = np.array(
            [
                [
                    hopping(spin1_row + 1, spin1_col + 1, spin2_row, spin2_col)
                    + hopping(spin2_row + 1, spin2_col + 1, spin1_row, spin1_col)
                    + u_interaction(
                        spin1_row, spin2_row, spin1_col, spin2_col, self.us[spin1_row]
                    )
                    + e_onsite_potential(
                        spin1_row,
                        spin2_row,
                        spin1_col,
                        spin2_col,
                        self.es[spin1_row] + self.es[spin2_row],
                    )
                    + v_interaction(
                        spin1_row,
                        spin2_row,
                        spin1_col,
                        spin2_col,
                        self.vs[spin1_row],
                    )
                    + v_interaction(spin1_row, spin2_row, spin1_col, spin2_col, 0, l=2)
                    # + v_interaction(spin1_row, spin2_row, spin1_col, spin2_col, l=3)
                    for spin1_row in range(self.spins)
                    for spin2_row in range(self.spins)
                ]
                for spin1_col in range(self.spins)
                for spin2_col in range(self.spins)
            ]
        )
        return H

    def update_js(self, js):
        self.js = self._set_couplings(js)
        self.H = self.H_matrix()
        self.H_subspace = self.H

    def _set_interactions_list(self, u):
        if type(u) is not list:
            u_list = [u] * self.spins
        else:
            if len(u) != self.spins:
                raise ValueError(
                    f"List of couplings must be the same length as number of spins."
                )
            else:
                u_list = u
        return u_list


class HubbardHamiltonianNparticles(Hamiltonian):
    def __init__(
        self,
        spins,
        excitations=2,
        edges=[[0, 1], [1, 2]],
        dt=0.1,
        js=1,
        us=1,
        es=0,
        vs=0,
    ):
        dimensions = spins ** 2
        super().__init__(dimensions, dt)
        self.spins = spins
        self.excitations = excitations
        self.edges = edges
        self.states = self._generate_states()
        self.js = self._set_couplings(js)
        self.us = self._set_interactions_list(us)
        self.es = self._set_interactions_list(es)
        self.vs = self._set_couplings(vs)
        self.adjacency_matrix = self.A_matrix()
        self.H = self.H_matrix()
        self.H_subspace = self.H

    def H_matrix(self):
        def u_interaction(state):
            bunches = state - 1
            bunches[bunches < 0] = 0
            energy = [bunch * self.us[i] for i, bunch in enumerate(bunches)]
            return np.sum(energy)

        def e_onsite_potential(state):
            energy = [occupation * self.es[i] for i, occupation in enumerate(state)]
            return np.sum(energy)

        def v_interaction(state):
            energy = [
                state[edge[0]] * state[edge[1]] * self.vs[e]
                for e, edge in enumerate(self.edges)
            ]
            return np.sum(energy)

        H = self.adjacency_matrix + np.diag(
            [
                u_interaction(state) + e_onsite_potential(state) + v_interaction(state)
                for state in self.states
            ]
        )
        return H

    def A_matrix(self):
        a = np.zeros((len(self.states), len(self.states)))
        for i, state_row in enumerate(self.states):
            for j, state_col in enumerate(self.states):
                transition = abs(state_row - state_col)
                ones = np.where(transition == 1)[0].tolist()
                for e, edge in self.edges:
                    if ones == edge:
                        a[i, j] = self.js[e]
        return a

    def update_js(self, js):
        self.js = self._set_couplings(js)
        self.H = self.H_matrix()
        self.H_subspace = self.H

    def _set_interactions_list(self, u):
        if type(u) is not list:
            u_list = [u] * self.spins
        else:
            if len(u) != self.spins:
                raise ValueError(
                    f"List of couplings must be the same length as number of spins."
                )
            else:
                u_list = u
        return u_list

    def _set_couplings(self, u):
        if type(u) is not list:
            u_list = [u] * len(self.edges)
        else:
            if len(u) != len(self.edges):
                raise ValueError(
                    f"List of couplings must be the same as the number of edges."
                )
            else:
                u_list = u
        return u_list

    def _generate_states(self):
        return [
            np.array(state)
            for state in itertools.product(
                range(self.excitations + 1), repeat=self.spins
            )
            if np.sum(state) == self.excitations
        ]


class LongRangeXYHamiltonian(HeisenbergHamiltonian):
    def __init__(self, spins, dt=0.1, alpha=1, open_chain=False):
        self.alpha = alpha
        dimensions = spins ** 2
        super().__init__(
            spins, dt=dt, jx=1, jy=1, jz=0, h_field=0, open_chain=open_chain
        )
        self.H = self.H_matrix()

    def H_matrix(self):
        if self.open_chain:

            def coef(row, col):
                return 1 / ((np.abs(col - row)) ** (self.alpha))

        else:

            def coef(row, col):
                return 1 / ((np.abs(col - row)) ** (self.alpha)) + 1 / (
                    (np.abs(self.spins - np.abs(col - row))) ** (self.alpha)
                )

        H = (1 / 4) * np.sum(
            [
                coef(i, j)
                * np.matmul(
                    Hamiltonian.multikron(
                        Hamiltonian.pauli_product(axis, self.spins, i)
                    ),
                    Hamiltonian.multikron(
                        Hamiltonian.pauli_product(axis, self.spins, j % self.spins)
                    ),
                )
                if i != j
                else 0
                for i in range(self.spins)
                for j in range(self.spins)
                for axis in ["x", "y"]
            ],
            axis=0,
        )
        return H


class LongRangeXYHamiltonian1d(Hamiltonian):
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
        hbar=False,
    ):
        super().__init__(spins, dt, hbar)
        self.open_chain = open_chain
        self.sender_reciever_qubits = sender_reciever_qubits
        self.spins = spins
        self.alpha = alpha
        self.hamiltonian_scaling = hamiltonian_scaling
        if interaction_map in ["all", 0]:
            self.is_interaction_map = 0
        else:
            self.is_interaction_map = 1
            self.interaction_map = interaction_map
            self.interaction_distance_bound = (
                interaction_distance_bound if interaction_distance_bound else 1
            )
            self.interaction_bound = self._generate_interaction_bound(
                self.interaction_distance_bound, self.hamiltonian_scaling
            )
        self.noise = noise
        self.samples = samples
        if self.samples:
            self.noise_arrays = []
            self.H_noise = []
            for sample in range(1, self.samples + 1, 1):
                self.noise_arrays.append(np.random.normal(0, self.noise, self.spins))
                self.H_noise.append(self.H_matrix(sample))
            self.H_noise_subspace = self.H_noise
        self.H = self.H_matrix()
        self.H_subspace = self.H

    def H_matrix(self, sample=0):
        if self.open_chain:
            if self.is_interaction_map:

                def coef(row, col):
                    if col != row:
                        if (
                            self.spins - col,
                            self.spins - row,
                        ) in self.interaction_map or (
                            self.spins - row,
                            self.spins - col,
                        ) in self.interaction_map:
                            c = 1 / ((np.abs(col - row)) ** (self.alpha))
                            c = (
                                c
                                if c <= self.interaction_bound
                                else self.interaction_bound
                            )
                        else:
                            c = 0
                    else:
                        c = 1
                    return c

            else:

                def coef(row, col):
                    if col != row:
                        c = 1 / ((np.abs(col - row)) ** (self.alpha))
                    else:
                        c = 1
                    return c

        else:
            if self.is_interaction_map:

                def coef(row, col):
                    if col != row:
                        if (
                            self.spins - col,
                            self.spins - row,
                        ) in self.interaction_map or (
                            self.spins - row,
                            self.spins - col,
                        ) in self.interaction_map:
                            c = 1 / ((np.abs(col - row)) ** (self.alpha)) + 1 / (
                                (np.abs(self.spins - np.abs(col - row))) ** (self.alpha)
                            )
                            c = (
                                c
                                if c <= self.interaction_bound
                                else self.interaction_bound
                            )
                        else:
                            c = 0
                    else:
                        c = 1
                    return c

            else:

                def coef(row, col):
                    if col != row:
                        c = 1 / ((np.abs(col - row)) ** (self.alpha)) + 1 / (
                            (np.abs(self.spins - np.abs(col - row))) ** (self.alpha)
                        )
                    else:
                        c = 1
                    return c

        H = np.array(
            [
                [coef(row, col) for col in range(self.spins)]
                for row in range(self.spins)
            ],
            dtype=np.complex128,
        )
        if sample:
            for i in range(self.spins):
                H[i, i] += self.noise_arrays[sample - 1][i]

        if self.open_chain and self.sender_reciever_qubits:
            zeros = [0 for _ in range(self.spins - 2)]
            H[0] = [1, 1] + zeros
            H[:, 0] = [1, 1] + zeros
            H[-1] = zeros + [1, 1]
            H[:, -1] = zeros + [1, 1]

        H = np.multiply(self.hamiltonian_scaling, H)
        return H

    def rescale_hamiltonian(self, scale):
        self.hamiltonian_scaling = scale
        if self.samples:
            for sample in range(self.samples):
                self.H_noise[sample] = scale * self.H_noise[sample]
            self.H_noise_subspace = self.H_noise
        else:
            self.H = scale * self.H
            self.H_subspace = self.H
        if self.is_interaction_map:
            self.interaction_bound = self._generate_interaction_bound(
                self.interaction_distance_bound
            )

    def add_marked_site(self, spin, hz=1, gamma_rescale=False, rescale_noise=True):
        site = self.spins - spin
        if self.samples:
            if gamma_rescale:
                for sample in range(self.samples):
                    self.H_noise[sample] = hz * self.H_noise[sample]
                    if rescale_noise:
                        for i in range(self.spins):
                            self.H_noise[sample][i, i] += (1 - hz) * self.noise_arrays[
                                sample
                            ][i]
                    self.H_noise[sample][(site, site)] = (
                        self.H_noise[sample][(site, site)] + 1
                    )
            else:
                for sample in range(self.samples):
                    self.H_noise[sample][(site, site)] = (
                        self.H_noise[sample][(site, site)] + hz
                    )
            self.H_noise_subspace = self.H_noise
        if gamma_rescale:
            self.H = hz * self.H
            self.H[(site, site)] = self.H[(site, site)] + 1
            # if self.is_interaction_map:
            #     self.interaction_bound = self._generate_interaction_bound(
            #         self.interaction_distance_bound, scaling=hz
            #     )
        else:
            self.H[(site, site)] = self.H[(site, site)] + hz
        self.H_subspace = self.H

    def update_marked_site(self, spin, hz=1, gamma_rescale=False, rescale_noise=True):
        self.add_marked_site(spin, hz, gamma_rescale, rescale_noise)

    def update_interaction_map(self, interaction_map, interaction_distance_bound=0):
        if interaction_map in ["all", 1]:
            self.is_interaction_map = 0
        else:
            self.is_interaction_map = 1
            self.interaction_map = interaction_map
        self.interaction_distance_bound = (
            interaction_distance_bound if interaction_distance_bound else 1
        )
        self.interaction_bound = self._generate_interaction_bound(
            self.interaction_distance_bound
        )
        if self.samples:
            self.H_noise = []
            for sample in range(1, self.samples + 1, 1):
                self.H_noise.append(self.H_matrix(sample))
            self.H_noise_subspace = self.H_noise
        self.H = self.H_matrix()
        self.H_subspace = self.H

    def _generate_interaction_bound(self, distance, scaling=1):
        if not self.open_chain:
            raise ValueError(
                f"Interaction bound in not currently implemented for closed spin chains."
            )
        else:
            return scaling / ((np.abs(distance)) ** (self.alpha))


class LongRangeXYHamiltonian1dExp(Hamiltonian):
    def __init__(
        self,
        spins,
        dt=0.000001,
        mu=2 * np.pi * 6.01e6,
        Omega=2 * np.pi * 1e6,
        delta_k=2 * 2 * np.pi / 355e-9 * np.array([1, 0, 0]),
        omega=2 * np.pi * np.array([6e6, 5e6, 0.5e6]),
        use_optimal_omega=False,
        t2=10e-3,
        samples=0,
        hbar=True,
    ):
        super().__init__(spins, dt, hbar)
        self.spins = spins
        self.mu = np.ones(1) * mu
        self.Omega = np.ones((spins, 1)) * Omega
        self.delta_k = delta_k
        self.ion_trap = IonTrap(
            spins,
            mu,
            omega,
            Omega,
            use_optimal_omega=use_optimal_omega,
            calculate_alpha=True,
        )
        self.alpha = self.ion_trap.alpha

        self.t2 = t2  # dephasing time
        self.dephasing_rate = 1 / (2 * np.pi * t2)
        self.samples = samples
        if self.samples:
            self.noise_arrays = []
            self.H_noise = []
            for sample in range(1, self.samples + 1, 1):
                self.noise_arrays.append(
                    np.random.normal(0, self.dephasing_rate, self.spins)
                )
                self.H_noise.append(self.H_matrix(sample))
            self.H_noise_subspace = self.H_noise
        self.H = self.H_matrix()
        self.H_subspace = self.H

    def H_matrix(self, sample=0):
        H = np.copy(self.ion_trap.Js)
        if sample:
            for i in range(self.spins):
                H[i, i] += self.noise_arrays[sample - 1][i]
        return H

    def update_H_matrix(self, sample, site, delta):
        H = np.copy(self.H_noise[sample - 1])
        H[site, site] += delta
        return H

    def rescale_hamiltonian(self, scale):
        self.hamiltonian_scaling = scale
        self.H = scale * self.H
        self.H_subspace = self.H

    def add_marked_site(self, spin, hz=1, gamma_rescale=False, rescale_noise=True):
        site = self.spins - spin
        if self.samples:
            if gamma_rescale:
                for sample in range(self.samples):
                    self.H_noise[sample] = hz * self.H_noise[sample]
                    if rescale_noise:
                        for i in range(self.spins):
                            self.H_noise[sample][i, i] += (1 - hz) * self.noise_arrays[
                                sample
                            ][i]
                    self.H_noise[sample][(site, site)] = (
                        self.H_noise[sample][(site, site)] + 1
                    )
            else:
                for sample in range(self.samples):
                    self.H_noise[sample][(site, site)] = (
                        self.H_noise[sample][(site, site)] + hz
                    )
            self.H_noise_subspace = self.H_noise
        if gamma_rescale:
            self.H = hz * self.H
            self.H[(site, site)] = self.H[(site, site)] + 1
            # if self.is_interaction_map:
            #     self.interaction_bound = self._generate_interaction_bound(
            #         self.interaction_distance_bound, scaling=hz
            #     )
        else:
            self.H[(site, site)] = self.H[(site, site)] + hz
        self.H_subspace = self.H

    def update_marked_site(self, spin, hz=1, gamma_rescale=False):
        self.add_marked_site(spin, hz, gamma_rescale)

    def add_broken_spin(self, spin, t2):
        self.broken_spin_noise_array = []
        dephasing_rate = 1 / t2
        H_broken_noise = []
        for sample in range(1, self.samples + 1, 1):
            spin_noise = np.random.normal(0, dephasing_rate)
            self.broken_spin_noise_array.append(spin_noise)
            H_broken_noise.append(self.update_H_matrix(sample, spin, delta=spin_noise))
        self.H_noise = H_broken_noise
        self.H_noise_subspace = self.H_noise
        self.H = self.H_matrix()


class LongRangeXYHamiltonian1dSubspace2(Hamiltonian):
    def __init__(self, spins, dt=0.1, alpha=1, open_chain=False):
        dimensions = spins ** 2
        super().__init__(dimensions, dt)
        self.open_chain = open_chain
        self.spins = spins
        self.alpha = alpha
        self.H = self.H_matrix()
        self.H_subspace = self.H

    def H_matrix(self):
        if self.open_chain:

            def coef(state_1, state_2):
                excitations_1 = [i for i, e in enumerate(state_1) if e == 1]
                excitations_2 = [i for i, e in enumerate(state_2) if e == 1]
                if excitations_1 != excitations_2:
                    distances = [
                        np.array(excitations_1) - np.array(excitations_2),
                        np.array(excitations_1) - np.array(excitations_2[::-1]),
                    ]
                    for distance in distances:
                        if 0 in distance:
                            for d in distance:
                                if d != 0:
                                    return 1 / ((np.abs(d)) ** (self.alpha))
                return 0

        else:

            def coef(state_1, state_2):
                excitations_1 = [i for i, e in enumerate(state_1) if e == 1]
                excitations_2 = [i for i, e in enumerate(state_2) if e == 1]
                spin1_row, spin2_row = excitations_1[0], excitations_1[1]
                spin1_col, spin2_col = excitations_2[0], excitations_2[1]
                if spin2_row == spin2_col:
                    if spin1_row != spin1_col:
                        c = 1 / (
                            (np.abs(spin1_col - spin1_row)) ** (self.alpha)
                        ) + 1 / (
                            (np.abs(self.spins - np.abs(spin1_col - spin1_row)))
                            ** (self.alpha)
                        )
                    else:
                        c = 0
                elif spin1_row == spin2_col:
                    if spin2_row != spin1_col:
                        c = 1 / (
                            (np.abs(spin2_row - spin1_col)) ** (self.alpha)
                        ) + 1 / (
                            (np.abs(self.spins - np.abs(spin2_row - spin1_col)))
                            ** (self.alpha)
                        )
                    else:
                        c = 0
                else:
                    c = 0
                return c

        states = list(itertools.product(range(2), repeat=self.spins))
        excitations = np.sum(states, axis=1)
        H = np.array(
            [
                [
                    coef(state_1, state_2)  # + coef(state_2, state_1)
                    for e, state_1 in enumerate(states)
                    if excitations[e] == 2
                ]
                for e, state_2 in enumerate(states)
                if excitations[e] == 2
            ]
        )
        return H

    def add_marked_site(self, spin, hz, gamma_rescale=False):
        site = self.spins - spin

        def coef(state_1, state_2):
            if state_1 == state_2:
                if state_1[site] == 1:
                    if gamma_rescale:
                        c = 1
                    else:
                        c = hz
                else:
                    c = 0
            else:
                c = 0
            return c

        if gamma_rescale:
            self.H = hz * self.H
        states = list(itertools.product(range(2), repeat=self.spins))
        excitations = np.sum(
            list(itertools.product(range(2), repeat=self.spins)), axis=1
        )
        H_marked = np.array(
            [
                [
                    coef(state_1, state_2)
                    for e, state_1 in enumerate(states)
                    if excitations[e] == 2
                ]
                for e, state_2 in enumerate(states)
                if excitations[e] == 2
            ]
        )
        self.H += H_marked
        self.H_subspace = self.H
