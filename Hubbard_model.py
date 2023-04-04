import quimb
import numpy as np
import matplotlib.pyplot as plt
from quimb.core import expectation


def hubbard_hamiltonian(n_ions, n_sites, js):
    dims = [2] * n_ions
    all_terms = []
    for i in range(n_sites):
        for j in range(n_sites):
            if i != j:
                Sx = quimb.spin_operator("x", sparse=True)
                Sy = quimb.spin_operator("y", sparse=True)
                all_terms.append(quimb.ikron([4 * js[i][j] * Sx, Sx], dims, [i, j]))
                all_terms.append(quimb.ikron([4 * js[i][j] * Sy, Sy], dims, [i, j]))
    H = sum(all_terms)
    return H
