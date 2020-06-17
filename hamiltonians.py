import numpy as np
import itertools
import scipy.linalg
import scipy.special


class Hamiltonian:

    def __init__(self, dimensions, dt):
        self.dimensions = dimensions
        self.dt = dt

    def U_matrix(self, H):
        return scipy.linalg.expm(-1j*H*self.dt)

    def U_matrix_approx(self, H):
        return np.identity(self.dimensions) - 1j*self.dt*H

    @staticmethod
    def spectrum(hamiltonian):
        """Returns sorted eigenvalues and eigenvectors."""
        eigenvalues, eigenvectors = scipy.linalg.eig(hamiltonian)
        index = np.argsort(-eigenvalues)  # largest eigenvalue to smallest
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[:,index]
        return eigenvectors, eigenvalues

    @staticmethod
    def multikron(matrices):
        product = matrices[0]
        for mat in matrices[1:]:
            product = np.kron(product, mat)
        return product

    @staticmethod
    def pauli_product(axis, n, j):
        paulis = [
            Hamiltonian.pauli(axis) if i == j else Hamiltonian.pauli(0) 
                for i in range(n)]
        return paulis

    @staticmethod
    def pauli(axis):
        if axis in [0, 'i']:
            sigma = np.array([[1,0],[0,1]])
        elif axis in [1, 'x']:
            sigma = np.array([[0,1],[1,0]])
        elif axis in [2, 'y']:
            sigma = np.array([[0,-1j],[1j,0]])
        elif axis in [3, 'z']:
            sigma = np.array([[1,0],[0,-1]])
        return sigma


class HeisenbergHamiltonian(Hamiltonian):

    def __init__(self, spins, dt=0.1, jx=1, jy=1, jz=0, h=0, approx=False):
        dimensions = 2**spins
        super().__init__(dimensions, dt)
        self.spins = spins
        self.jx = jx
        self.jy = jy
        self.jz = jz
        self.js = {'x': jx, 'y': jy, 'z': jz}
        self.h = h
        self.H = self.H_matrix()
        self.transition_map = self.generate_transition_map()
        self.excitation_map = self.generate_excitation_map()
        self.approx = approx
        if approx:
            self.U = self.U_matrix_approx(self.H)
        else:
            self.U = self.U_matrix(self.H)

    def initialise_subspace(self, subspace):
        if subspace >= 0:
            self.H_subspace = self.subspace_H_matrix(subspace)
            if self.approx:
                self.U_subspace = self.U_matrix_approx(self.H_subspace)
            else:
                self.U_subspace = self.U_matrix(self.H_subspace)           

    def H_matrix(self):
        H = -(1/2)*np.sum([
            self.js[axis]*np.matmul(
                Hamiltonian.multikron(Hamiltonian.pauli_product(axis, self.spins, j)), 
                Hamiltonian.multikron(Hamiltonian.pauli_product(axis, self.spins, (j+1)%self.spins))) 
            for j in range(self.spins)
            for axis in ['x', 'y', 'z']
            ], axis=0) - np.sum([
            self.h*Hamiltonian.multikron(Hamiltonian.pauli_product('z', self.spins, j))
            for j in range(self.spins)], axis=0)
        return H

    def subspace_H_matrix(self, excitations):
        dim = range(len(self.excitation_map))
        subspace_H = np.array([[self.excitation_map[row][col] == (excitations, excitations) for col in dim] 
                            for row in dim])
        subspace_size = scipy.special.comb(self.spins, excitations, exact=True)
        subspace_H = np.reshape(self.H[subspace_H], (subspace_size, subspace_size))
        return subspace_H

    def generate_transition_map(self):
        states = [''.join([str(i) for i in state]) for state in itertools.product(range(2), repeat=self.spins)] 
        t_map = [
            [(states[i], states[j]) for j in range(len(states))] 
            for i in range(len(states))]
        return t_map

    def generate_excitation_map(self):
        excites = [np.sum(state) for state in itertools.product(range(2), repeat=self.spins)]
        e_map = [
            [(excites[i], excites[j]) for j in range(len(excites))] 
            for i in range(len(excites))]
        return e_map
