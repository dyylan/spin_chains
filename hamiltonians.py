import numpy as np
import scipy.linalg

class Hamiltonian:

    def __init__(self, dimensions, dt):
        self.dimensions = dimensions
        self.dt = dt

    def U_matrix(self):
        return scipy.linalg.expm(-1j*self.H*self.dt)

    def U_matrix_approx(self):
        return np.identity(self.dimensions) - 1j*self.dt*self.H

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

    def __init__(self, spins, dt=0.1, jx=1, jy=1, jz=1, approx=False):
        dimensions = 2**spins
        super().__init__(dimensions, dt)
        self.n = spins
        self.jx = jx
        self.jy = jy
        self.jz = jz
        self.js = {'x': jx, 'y': jy, 'z': jz}
        self.H = self.H_matrix()
        if approx:
            self.U = self.U_matrix_approx()
        else:
            self.U = self.U_matrix()
        

    def H_matrix(self):
        H = -(1/2)*np.sum([
            self.js[axis]*np.matmul(
                Hamiltonian.multikron(Hamiltonian.pauli_product(axis, self.n, j)), 
                Hamiltonian.multikron(Hamiltonian.pauli_product(axis, self.n, (j+1)%self.n)))
            for j in range(self.n)
            for axis in ['x', 'y', 'z']
        ], axis=0)
        return H