from re import A
import quimb
import numpy as np
import matplotlib.pyplot as plt
from quimb.core import expectation, trace
from quimb.linalg.base_linalg import expm
from torch import spmm

from spin_chains.quantum import iontrap, chains, states
from spin_chains.data_analysis import data_handling, data_plots


class SpinBosonHamMS:
    def __init__(self, n_phonons, n_ions, omega_eff, gs, omega_single_mode, init_t=0):
        self.n_ions = n_ions
        self.n_phonons = n_phonons
        self.dims = [n_phonons] + ([2] * n_ions)
        self.omega_eff = omega_eff
        self.gs = gs
        self.omega_single_mode = omega_single_mode
        self.init_t = init_t
        self.h_interaction_term_1 = self.generate_interaction_term("create", "x")
        self.h_interaction_term_2 = self.generate_interaction_term("destroy", "x")
        self.h_interaction_term_3 = self.generate_interaction_term("create", "y")
        self.h_interaction_term_4 = self.generate_interaction_term("destroy", "y")

    def __call__(self, t):
        t += self.init_t
        e = np.exp(1j * self.omega_single_mode * t)
        e_d = np.exp(-1j * self.omega_single_mode * t)
        c = np.cos(self.omega_eff * t)
        s = np.sin(self.omega_eff * t)
        ham = (
            -(e * c * self.h_interaction_term_1)
            - (e_d * c * self.h_interaction_term_2)
            + (e * s * self.h_interaction_term_3)
            + (e_d * s * self.h_interaction_term_4)
        )
        return ham

    # @quimb.hamiltonian_builder
    def generate_interaction_term(self, boson_op="create", spin_op="x"):
        s = quimb.spin_operator(spin_op, sparse=True)
        if boson_op in ["create", "c"]:
            a = quimb.create(self.n_phonons)
        else:
            a = quimb.destroy(self.n_phonons)
        ham = sum(
            [
                quimb.ikron([a, self.gs[i] * s], self.dims, [0, i + 1])
                for i in range(self.n_ions)
            ]
        )
        return ham

    # @quimb.hamiltonian_builder
    def generate_marked_term(self, dims, gamma, mark_site=1):
        Sz = quimb.spin_operator("Z", sparse=True)
        return quimb.ikron(2 * gamma * Sz, dims, inds=[mark_site])


class SpinBosonHamMSDyson3:
    def __init__(self, n_phonons, n_ions, omega_eff, gs, omega_single_mode, init_t=0):
        self.n_ions = n_ions
        self.n_phonons = n_phonons
        self.dims = [n_phonons] + ([2] * n_ions)
        self.omega_eff = omega_eff
        self.g = gs
        self.js = (
            gs ** 2 * omega_single_mode / (2 * omega_eff ** 2 - omega_single_mode ** 2)
        )
        self.omega_single_mode = omega_single_mode
        self.init_t = init_t
        self.Delta = omega_eff - omega_single_mode
        self.Sum = omega_eff + omega_single_mode
        self.xy_ham = self.xy_hamiltonian()
        self.h_interaction_term_001100 = self.generate_interaction_term("001100")
        self.h_interaction_term_000011 = self.generate_interaction_term("000011")
        self.h_interaction_term_011001 = self.generate_interaction_term("011001")
        self.h_interaction_term_010110 = self.generate_interaction_term("010110")
        self.h_interaction_term_011100 = self.generate_interaction_term("011100")
        self.h_interaction_term_010011 = self.generate_interaction_term("010011")
        self.h_interaction_term_101001 = self.generate_interaction_term("101001")
        self.h_interaction_term_100110 = self.generate_interaction_term("100110")
        self.h_interaction_term_101100 = self.generate_interaction_term("101100")
        self.h_interaction_term_100011 = self.generate_interaction_term("100011")
        self.h_interaction_term_111001 = self.generate_interaction_term("111001")
        self.h_interaction_term_110110 = self.generate_interaction_term("110110")

    def __call__(self, t):
        t += self.init_t
        gamma_001100 = self.gamma_001100(t)
        gamma_000011 = self.gamma_000011(t)
        gamma_011001 = self.gamma_011001(t)
        gamma_010110 = self.gamma_010110(t)
        gamma_011100 = self.gamma_011100(t)
        gamma_010011 = self.gamma_010011(t)
        gamma_101001 = self.gamma_101001(t)
        gamma_100110 = self.gamma_100110(t)
        gamma_101100 = self.gamma_101100(t)
        gamma_100011 = self.gamma_100011(t)
        gamma_111001 = self.gamma_111001(t)
        gamma_110110 = self.gamma_110110(t)
        ham = self.xy_ham - (
            (self.g ** 3)
            * (
                # (gamma_001100 * self.h_interaction_term_001100)
                # + (gamma_000011 * self.h_interaction_term_000011)
                # + (gamma_011001 * self.h_interaction_term_011001)
                # + (gamma_010110 * self.h_interaction_term_010110)
                # + (gamma_011100 * self.h_interaction_term_011100)
                # + (gamma_010011 * self.h_interaction_term_010011)
                # + (gamma_101001 * self.h_interaction_term_101001)
                # + (gamma_100110 * self.h_interaction_term_100110)
                # +
                (gamma_101100 * self.h_interaction_term_101100)
                + (gamma_100011 * self.h_interaction_term_100011)
                # + (gamma_111001 * self.h_interaction_term_111001)
                # + (gamma_110110 * self.h_interaction_term_110110)
            )
        )

        return ham

    def xy_hamiltonian(self):
        all_terms = []
        for i in range(self.n_ions):
            for j in range(self.n_ions):
                if i != j:
                    Sx = quimb.spin_operator("x", sparse=True)
                    Sy = quimb.spin_operator("y", sparse=True)
                    all_terms.append(
                        quimb.ikron([4 * self.js * Sx, Sx], self.dims, [i + 1, j + 1])
                    )
                    all_terms.append(
                        quimb.ikron([4 * self.js * Sy, Sy], self.dims, [i + 1, j + 1])
                    )
        H = sum(all_terms)
        return H

    def generate_interaction_term(self, pattern):
        sx = quimb.spin_operator("x", sparse=True)
        sy = quimb.spin_operator("y", sparse=True)
        splus = sx + (1.0j * sy)
        sminus = sx - (1.0j * sy)
        spp = splus @ splus
        smp = sminus @ splus
        spm = splus @ sminus
        smm = sminus @ sminus
        sppm = splus @ splus @ sminus
        spmp = splus @ sminus @ splus
        smpm = sminus @ splus @ sminus
        smmp = sminus @ sminus @ splus
        create = quimb.create(self.n_phonons)
        destroy = quimb.destroy(self.n_phonons)
        ccd = create @ create @ destroy
        cdc = create @ destroy @ create
        cdd = create @ destroy @ destroy
        dcd = destroy @ create @ destroy
        ddc = destroy @ destroy @ create
        if pattern == "001100":
            operators_ijk = [dcd, sminus, splus, sminus]
            operators_iik = [dcd, smp, sminus]
            operators_ijj = [dcd, sminus, spm]
            operators_kik = [dcd, smm, splus]
            operators_iii = [dcd, smpm]
        elif pattern == "000011":
            operators_ijk = [ddc, sminus, sminus, splus]
            operators_iik = [ddc, smm, splus]
            operators_ijj = [ddc, sminus, smp]
            operators_kik = [ddc, smp, sminus]
            operators_iii = [ddc, smmp]
        elif pattern == "011001":
            operators_ijk = [dcd, splus, sminus, splus]
            operators_iik = [dcd, spm, splus]
            operators_ijj = [dcd, splus, smp]
            operators_kik = [dcd, spp, sminus]
            operators_iii = [dcd, spmp]
        elif pattern == "010110":
            operators_ijk = [ddc, splus, splus, sminus]
            operators_iik = [ddc, spp, sminus]
            operators_ijj = [ddc, splus, spm]
            operators_kik = [ddc, spm, splus]
            operators_iii = [ddc, sppm]
        elif pattern == "011100":
            operators_ijk = [dcd, splus, splus, sminus]
            operators_iik = [dcd, spp, sminus]
            operators_ijj = [dcd, splus, spm]
            operators_kik = [dcd, spm, splus]
            operators_iii = [dcd, sppm]
        elif pattern == "010011":
            operators_ijk = [ddc, splus, sminus, splus]
            operators_iik = [ddc, spm, splus]
            operators_ijj = [ddc, splus, smp]
            operators_kik = [ddc, spp, sminus]
            operators_iii = [ddc, spmp]
        elif pattern == "101001":
            operators_ijk = [ccd, sminus, sminus, splus]
            operators_iik = [ccd, smm, splus]
            operators_ijj = [ccd, sminus, smp]
            operators_kik = [ccd, smp, sminus]
            operators_iii = [ccd, smmp]
        elif pattern == "100110":
            operators_ijk = [cdc, sminus, splus, sminus]
            operators_iik = [cdc, smp, sminus]
            operators_ijj = [cdc, sminus, spm]
            operators_kik = [cdc, smm, splus]
            operators_iii = [cdc, smpm]
        elif pattern == "101100":
            operators_ijk = [ccd, splus, splus, sminus]
            operators_iik = [ccd, spp, sminus]
            operators_ijj = [ccd, splus, spm]
            operators_kik = [ccd, spm, splus]
            operators_iii = [ccd, sppm]
        elif pattern == "100011":
            operators_ijk = [cdc, sminus, splus, sminus]
            operators_iik = [cdc, smp, sminus]
            operators_ijj = [cdc, sminus, spm]
            operators_kik = [cdc, smm, splus]
            operators_iii = [cdc, smpm]
        elif pattern == "111001":
            operators_ijk = [cdd, splus, sminus, splus]
            operators_iik = [cdd, spm, splus]
            operators_ijj = [cdd, splus, smp]
            operators_kik = [cdd, spp, sminus]
            operators_iii = [cdd, spmp]
        elif pattern == "110110":
            operators_ijk = [cdc, splus, splus, sminus]
            operators_iik = [cdc, spp, sminus]
            operators_ijj = [cdc, splus, spm]
            operators_kik = [cdc, spm, splus]
            operators_iii = [cdc, sppm]
        ham = sum(
            [
                quimb.ikron(
                    operators_ijk,
                    self.dims,
                    [0, i + 1, j + 1, k + 1],
                )
                for i in range(self.n_ions)
                for j in range(self.n_ions)
                for k in range(self.n_ions)
                if (i != j) and (j != k) and (i != k)
            ]
            + [
                quimb.ikron(
                    operators_iik,
                    self.dims,
                    [0, i + 1, j + 1],
                )
                for i in range(self.n_ions)
                for j in range(self.n_ions)
                if i != j
            ]
            + [
                quimb.ikron(
                    operators_ijj,
                    self.dims,
                    [0, i + 1, j + 1],
                )
                for i in range(self.n_ions)
                for j in range(self.n_ions)
                if i != j
            ]
            + [
                quimb.ikron(
                    operators_kik,
                    self.dims,
                    [0, i + 1, j + 1],
                )
                for i in range(self.n_ions)
                for j in range(self.n_ions)
                if i != j
            ]
            + [
                quimb.ikron(
                    operators_iii,
                    self.dims,
                    [0, i + 1],
                )
                for i in range(self.n_ions)
            ]
        )
        return ham

    def gamma_001100(self, t):
        a = np.exp(2.0j * self.omega_eff * t) / (
            self.omega_eff ** 2 - self.omega_single_mode ** 2
        )
        return a

    def gamma_000011(self, t):
        a = np.exp(2.0j * self.omega_single_mode * t) / (
            self.omega_eff ** 2 - self.omega_single_mode ** 2
        )
        return a

    def gamma_011001(self, t):
        a = -1 * (1 + np.exp(1.0j * self.Delta * t)) / (self.Delta ** 2)
        return a

    def gamma_010110(self, t):
        a = np.exp(1.0j * self.Delta * t) / (self.Delta ** 2)
        return a

    def gamma_011100(self, t):
        a = np.exp(1.0j * self.Delta * t) / (
            self.omega_eff ** 2 - self.omega_single_mode ** 2
        )
        return a

    def gamma_010011(self, t):
        a = np.exp(1.0j * self.Sum * t) / (
            self.omega_eff ** 2 - self.omega_single_mode ** 2
        )
        return a

    def gamma_101001(self, t):
        a = np.exp(-1.0j * self.Delta * t) / (self.Delta ** 2)
        return a

    def gamma_100110(self, t):
        a = -1 * (np.exp(-1.0j * self.Delta * t) + 1) / (self.Delta ** 2)
        return a

    def gamma_101100(self, t):
        a = (
            -1
            * (np.exp(-1.0j * self.Delta * t))
            / (self.omega_eff ** 2 - self.omega_single_mode ** 2)
        )
        return a

    def gamma_100011(self, t):
        a = (np.exp(-1.0j * self.Delta * t)) / (
            self.omega_eff ** 2 - self.omega_single_mode ** 2
        )
        return a

    def gamma_111001(self, t):
        a = (
            -1
            * (np.exp(-1.0j * self.Sum * t))
            / (self.omega_eff ** 2 - self.omega_single_mode ** 2)
        )
        return a

    def gamma_110110(self, t):
        a = np.exp(1.0j * self.Sum * t) / (
            self.omega_eff ** 2 - self.omega_single_mode ** 2
        )
        return a


def xyh_hamiltonian(n_ions, js, hs, r=1):
    dims = [2] * n_ions
    all_terms = []
    Sx = quimb.spin_operator("x", sparse=True)
    Sy = quimb.spin_operator("y", sparse=True)
    Sz = quimb.spin_operator("z", sparse=True)
    for i in range(n_ions):
        for j in range(n_ions):
            if i != j:
                all_terms.append(quimb.ikron([r * 4 * js[i][j] * Sx, Sx], dims, [i, j]))
                all_terms.append(quimb.ikron([r * 4 * js[i][j] * Sy, Sy], dims, [i, j]))
        all_terms.append(quimb.ikron([r * 2 * hs[i] * Sz], dims, [i]))
    H = sum(all_terms)
    return H


if __name__ == "__main__":
    n_ions = 6
    n_phonons = 4
    time = 0.00001
    target_alpha = 0.2
    r = 1

    # Ion trap parameters
    data = data_handling.read_data_spin(
        protocol="experimental/always_on_fast_xy",
        chain="open",
        alpha=target_alpha,
        save_tag="optimum_gammas_end_n_mu_min",
        spins=n_ions,
    )
    mu = np.ones(1) * (data["mu"] * 1)  # 8 ions
    z_trap_frequency = data["z_trap_frequency"]

    ion_trap_xy = iontrap.IonTrapXY(
        n_ions=n_ions,
        mu=mu,
        omega=np.array([2 * np.pi * 6e6, 2 * np.pi * 5e6, z_trap_frequency]),
    )
    gs, omega_single_mode, omega_eff, delta = ion_trap_xy.single_mode_ms()
    ion_trap_xy.calculate_alpha()
    print(f"XY Js = {ion_trap_xy.Js}")
    print(f"alpha = {ion_trap_xy.alpha}")
    # time = 40 * 2 * np.pi / delta
    print(
        f"g vector = {gs}\ndelta = {delta}\nomega_eff = {omega_eff}\nomega_single_mode = {omega_single_mode}\nomega_eff/delta = {omega_eff/delta}"
    )
    Js = [
        [
            gi
            * gj
            * omega_single_mode
            * (1 / (8 * (omega_eff ** 2 - omega_single_mode ** 2)))
            for gi in gs
        ]
        for gj in gs
    ]
    hs = [
        gi ** 2 * omega_eff / (4 * (omega_eff ** 2 - omega_single_mode ** 2))
        for gi in gs
    ]

    print(f"single mode Js = {Js[0][0]}")
    # Optimal gamma
    gamma = 11792.258802298766

    # Initial state
    up = quimb.qu([1, 0], qtype="ket")
    down = quimb.qu([0, 1], qtype="ket")
    init_spin_l = [up] + [down for _ in range(n_ions - 1)]
    final_spin_l = [down for _ in range(n_ions - 1)] + [up]
    groundstate_l = [down for _ in range(n_ions)]

    init_boson = quimb.qu([1, 0, 0, 0], qtype="ket")
    single_boson = quimb.qu([0, 1, 0, 0], qtype="ket")
    init_spin_xy = quimb.kron(*init_spin_l)
    psi0 = quimb.kron(init_boson, init_spin_xy)
    groundstate = quimb.kron(*groundstate_l)

    # Hamiltonian
    ham_t = SpinBosonHamMS(
        n_phonons=n_phonons,
        n_ions=n_ions,
        omega_eff=omega_eff,
        gs=gs,
        omega_single_mode=omega_single_mode,
        init_t=0,
    )

    ham_xy = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs, r=r)
    evolution_xy = quimb.Evolution(init_spin_xy, ham_xy)

    def calc_t(t, _):
        return t

    def calc_fidelity(t, pt):
        rho = quimb.partial_trace(
            pt, [n_phonons] + ([2] * n_ions), [i for i in range(1, n_ions + 1, 1)]
        )
        evolution_xy.update_to(t)
        pt_xy = evolution_xy.pt
        return quimb.expectation(pt_xy, rho)

    compute = {"time": calc_t, "fidelity": calc_fidelity}

    # Evolution
    evolution_full = quimb.Evolution(
        psi0, ham_t, progbar=True, compute=compute, method="integrate"
    )

    evolution_full.update_to(time)

    print(f"Number of points = {len(evolution_full.results['time'])}")

    ham_xy = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs, r=r)
    evolution_xy = quimb.Evolution(init_spin_xy, ham_xy)

    def c_fidelity(t, pt):
        rho = quimb.partial_trace(
            pt, [n_phonons] + ([2] * n_ions), [i for i in range(1, n_ions + 1, 1)]
        )
        # rho = quimb.normalize(rho)
        evolution_xy.update_to(t)
        pt_xy = evolution_xy.pt
        return quimb.expectation(pt_xy, rho)

    # Hamiltonian
    hamD3_t = SpinBosonHamMSDyson3(
        n_phonons=n_phonons,
        n_ions=n_ions,
        omega_eff=omega_eff,
        gs=gs[0] / 2,
        omega_single_mode=omega_single_mode,
        init_t=0,
    )

    ham_xy = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs, r=r)
    evolution_xy = quimb.Evolution(init_spin_xy, ham_xy)

    # Evolution
    evolution_D3 = quimb.Evolution(
        psi0, hamD3_t, progbar=True, compute=compute, method="integrate"
    )

    evolution_full.update_to(time)

    def calculate_analytical_fidelity(t, N=1):
        E = (
            (1 - np.cos((omega_eff - omega_single_mode) * t))
            * np.power(gs[0], 2)
            / (2 * np.power(omega_eff - omega_single_mode, 2))
        )
        return 1 - E

    times = list(np.linspace(0, time, 1000))
    analytical_fidelities = [calculate_analytical_fidelity(t, n_ions) for t in times]

    # Plot results
    fig, ax = plt.subplots(figsize=[12, 8])
    ax.plot(
        evolution_full.results["time"],
        evolution_full.results["fidelity"],
    )

    ax.plot(
        evolution_D3.results["time"],
        evolution_D3.results["fidelity"],
    )

    ax.plot(
        times,
        analytical_fidelities,
    )

    ax.legend(
        [
            "Full model",
            # "XY with bounded Dyson series",
            "Truncated Dyson series",
            # "Truncated Dyson series",
            "Analytical",
        ]
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fidelity")

    ax.plot()
    plt.show()
