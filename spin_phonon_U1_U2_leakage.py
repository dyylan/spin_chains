import quimb
import numpy as np
import matplotlib.pyplot as plt
from quimb.core import expectation
from quimb.linalg.base_linalg import expm

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


class U1rwa:
    def __init__(self, n_phonons, n_ions, g, omega_eff, omega_single_mode):
        self.n_ions = n_ions
        self.n_phonons = n_phonons
        self.dims = [n_phonons] + ([2] * n_ions)
        self.g = g
        self.t = 0
        self.omega_eff = omega_eff
        self.omega_single_mode = omega_single_mode
        self.h_interaction_term_1 = self.generate_interaction_term("destroy", "+")
        self.h_interaction_term_2 = self.generate_interaction_term("create", "-")
        self.unitary(self.t)

    def unitary(self, t):
        self.t = t
        a01 = self.alpha_01(t)
        a10 = self.alpha_10(t)
        self.uni = -(a01 * self.h_interaction_term_1) - (
            a10 * self.h_interaction_term_2
        )
        return self.uni

    # @quimb.hamiltonian_builder
    def generate_interaction_term(self, boson_op="create", spin_op="+"):
        sx = quimb.spin_operator("x", sparse=True)
        sy = quimb.spin_operator("y", sparse=True)
        if spin_op == "+":
            s = sx + (1.0j * sy)
        elif spin_op == "-":
            s = sx - (1.0j * sy)
        if boson_op in ["create", "c"]:
            a = quimb.create(self.n_phonons)
        else:
            a = quimb.destroy(self.n_phonons)
        ham = sum(
            [
                quimb.ikron([a, self.g * s], self.dims, [0, i + 1])
                for i in range(self.n_ions)
            ]
        )
        return ham

    # @quimb.hamiltonian_builder
    def generate_marked_term(self, dims, gamma, mark_site=1):
        Sz = quimb.spin_operator("Z", sparse=True)
        return quimb.ikron(2 * gamma * Sz, dims, inds=[mark_site])

    def alpha_01(self, t):
        a = (1 - np.exp(-1.0j * (self.omega_eff - self.omega_single_mode) * t)) / (
            self.omega_eff - self.omega_single_mode
        )
        return a

    def alpha_10(self, t):
        a = (
            -1
            * (1 - np.exp(1.0j * (self.omega_eff - self.omega_single_mode) * t))
            / (self.omega_eff - self.omega_single_mode)
        )
        return a


class U2rwa:
    def __init__(self, n_phonons, n_ions, g, omega_eff, omega_single_mode):
        self.n_ions = n_ions
        self.n_phonons = n_phonons
        self.dims = [n_phonons] + ([2] * n_ions)
        self.g = g
        self.t = 0
        self.omega_eff = omega_eff
        self.omega_single_mode = omega_single_mode
        self.Delta = omega_eff - omega_single_mode
        self.Sum = omega_eff + omega_single_mode
        self.h_interaction_term_1001 = self.generate_interaction_term("1001")
        self.h_interaction_term_0110 = self.generate_interaction_term("0110")
        self.h_interaction_term_1100 = self.generate_interaction_term("1100")
        self.h_interaction_term_0011 = self.generate_interaction_term("0011")
        self.unitary(self.t)

    def unitary(self, t):
        self.t = t
        b1001 = self.beta_1001(t)
        b0110 = self.beta_0110(t)
        b1100 = self.beta_1001(t)
        b0011 = self.beta_1001(t)
        self.uni = (self.g ** 2) * (
            -(b1001 * self.h_interaction_term_1001)
            - (b0110 * self.h_interaction_term_0110)
            - (b1100 * self.h_interaction_term_1100)
            - (b0011 * self.h_interaction_term_0011)
        )
        return self.uni

    def generate_interaction_term(self, pattern):
        sx = quimb.spin_operator("x", sparse=True)
        sy = quimb.spin_operator("y", sparse=True)
        splus = sx + (1.0j * sy)
        sminus = sx - (1.0j * sy)
        spm = splus @ sminus
        smp = sminus @ splus
        create = quimb.create(self.n_phonons)
        destroy = quimb.destroy(self.n_phonons)
        cd = create @ destroy
        dc = destroy @ create
        if pattern == "1001":
            operators_ij = [cd, sminus, splus]
            operators_ii = [cd, smp]
        elif pattern == "0110":
            operators_ij = [dc, splus, sminus]
            operators_ii = [dc, spm]
        elif pattern == "1100":
            operators_ij = [cd, splus, sminus]
            operators_ii = [cd, spm]
        elif pattern == "0011":
            operators_ij = [dc, sminus, splus]
            operators_ii = [dc, smp]
        ham = sum(
            [
                quimb.ikron(
                    operators_ij,
                    self.dims,
                    [0, i + 1, j + 1],
                )
                for i in range(self.n_ions)
                for j in range(self.n_ions)
                if i != j
            ]
            + [
                quimb.ikron(
                    operators_ii,
                    self.dims,
                    [
                        0,
                        i + 1,
                    ],
                )
                for i in range(self.n_ions)
            ]
        )
        return ham

    # def beta_1001(self, t):
    #     a = (-1.0j * t / self.Delta) + (
    #         (1 - np.exp(-1.0j * self.Delta * t)) / (self.Delta ** 2)
    #     )
    #     return a

    # def beta_0110(self, t):
    #     a = (1.0j * t / self.Delta) + (
    #         (1 - np.exp(1.0j * self.Delta * t)) / (self.Delta ** 2)
    #     )
    #     return a

    # def beta_1100(self, t):
    #     a = (1.0j * t / self.Sum) + (
    #         (1 - np.exp(1.0j * self.Sum * t)) / (self.Sum ** 2)
    #     )
    #     return a

    # def beta_0011(self, t):
    #     a = (-1.0j * t / self.Sum) + (
    #         (1 - np.exp(-1.0j * self.Sum * t)) / (self.Sum ** 2)
    #     )
    #     return a

    def beta_1001(self, t):
        a = -1.0j * t / self.Delta
        return a

    def beta_0110(self, t):
        a = 1.0j * t / self.Delta
        return a

    def beta_1100(self, t):
        a = 1.0j * t / self.Sum
        return a

    def beta_0011(self, t):
        a = -1.0j * t / self.Sum
        return a


class U2rwaXY:
    def __init__(self, n_phonons, n_ions, js):
        self.n_ions = n_ions
        self.n_phonons = n_phonons
        self.dims = [n_phonons] + ([2] * n_ions)
        self.js = js
        self.t = 0
        self.ham = self.xy_hamiltonian(n_ions, js)
        self.unitary(self.t)

    def unitary(self, t):
        self.t = t
        coef = -1.0j * t
        self.uni = expm(coef * self.ham, herm=True)
        return self.uni

    def xy_hamiltonian(self, n_ions, js):
        all_terms = []
        for i in range(n_ions):
            for j in range(n_ions):
                if i != j:
                    Sx = quimb.spin_operator("x", sparse=True)
                    Sy = quimb.spin_operator("y", sparse=True)
                    all_terms.append(
                        quimb.ikron([4 * js[i][j] * Sx, Sx], self.dims, [i + 1, j + 1])
                    )
                    all_terms.append(
                        quimb.ikron([4 * js[i][j] * Sy, Sy], self.dims, [i + 1, j + 1])
                    )
        H = sum(all_terms)
        return H


class U3rwa:
    def __init__(self, n_phonons, n_ions, g, omega_eff, omega_single_mode):
        self.n_ions = n_ions
        self.n_phonons = n_phonons
        self.dims = [n_phonons] + ([2] * n_ions)
        self.g = g
        self.t = 0
        self.omega_eff = omega_eff
        self.omega_single_mode = omega_single_mode
        self.h_interaction_term_1 = self.generate_interaction_term("destroy", "+")
        self.h_interaction_term_2 = self.generate_interaction_term("create", "-")
        self.unitary(self.t)

    def unitary(self, t):
        self.t = t
        a01 = self.alpha_01(t)
        a10 = self.alpha_10(t)
        self.uni = -(a01 * self.h_interaction_term_1) - (
            a10 * self.h_interaction_term_2
        )
        return self.uni

    # @quimb.hamiltonian_builder
    def generate_interaction_term(self, boson_op="create", spin_op="+"):
        sx = quimb.spin_operator("x", sparse=True)
        sy = quimb.spin_operator("y", sparse=True)
        if spin_op == "+":
            s = sx + (1.0j * sy)
        elif spin_op == "-":
            s = sx - (1.0j * sy)
        if boson_op in ["create", "c"]:
            a = quimb.create(self.n_phonons)
        else:
            a = quimb.destroy(self.n_phonons)
        ham = sum(
            [
                quimb.ikron([a, self.g * s], self.dims, [0, i + 1])
                for i in range(self.n_ions)
            ]
        )
        return ham

    # @quimb.hamiltonian_builder
    def generate_marked_term(self, dims, gamma, mark_site=1):
        Sz = quimb.spin_operator("Z", sparse=True)
        return quimb.ikron(2 * gamma * Sz, dims, inds=[mark_site])

    def alpha_01(self, t):
        a = (1 - np.exp(-1.0j * (self.omega_eff - self.omega_single_mode) * t)) / (
            self.omega_eff - self.omega_single_mode
        )
        return a

    def alpha_10(self, t):
        a = (
            -1
            * (1 - np.exp(1.0j * (self.omega_eff - self.omega_single_mode) * t))
            / (self.omega_eff - self.omega_single_mode)
        )
        return a


if __name__ == "__main__":
    n_ions = 8
    n_phonons = 4
    steps = 1000
    print_step = 10
    times = list(np.linspace(0, 0.0001, steps))
    target_alpha = 0.2

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

    def calc_t(t, _):
        return t

    def calc_leakage(t, pt):
        rho = quimb.partial_trace(
            pt, [n_phonons] + ([2] * n_ions), [i for i in range(1, n_ions + 1, 1)]
        )
        return abs(quimb.expectation(groundstate, rho))

    compute = {"time": calc_t, "leakage": calc_leakage}

    # Evolution
    evolution_full = quimb.Evolution(
        psi0, ham_t, progbar=True, compute=compute, method="integrate"
    )

    evolution_full.update_to(times[-1])

    print(f"Number of points = {len(evolution_full.results['time'])}")

    dims = [n_phonons] + ([2] * n_ions)
    identity = quimb.ikron(quimb.identity(n_phonons), dims, 0)
    U1 = U1rwa(n_phonons, n_ions, gs[0] / 2, omega_eff, omega_single_mode)
    U2 = U2rwa(n_phonons, n_ions, gs[0] / 2, omega_eff, omega_single_mode)
    # U2 = U2rwaXY(n_phonons, n_ions, Js)
    rho = psi0 @ psi0.H

    # leakages = []
    # count = 0
    # for t in times:
    #     # U = U1.unitary(t) + U2.unitary(t)
    #     U = identity + U1.unitary(t) + U2.unitary(t)
    #     rho_t = U @ rho @ U.H
    #     leakages.append(calc_leakage(t, rho_t))
    #     count += 1
    #     if not count % print_step:
    #         print(f"--> Computed unitary for t = {t} ({count/steps*100}%)")
    #         print(
    #             f"\t - With Tr[U1(t)] = {quimb.tr(U1.unitary(t) @ rho @ U1.unitary(t).H)}"
    #         )

    def calculate_leakage(t, N=1, r=1):
        Et = (
            (1 - np.cos((r * omega_eff - omega_single_mode) * t))
            * np.power(gs[0], 2)
            / (2 * np.power(r * omega_eff - omega_single_mode, 2))
        )
        return Et

    Es = []
    # omega_eff_prefactors = [1, 1.089, 1.0893, 1.0896]
    omega_eff_prefactors = [1, 1.0003401, 1.000338]
    # delta_omegas = [6388, 6389, 6390]
    for r in omega_eff_prefactors:
        Es.append([calculate_leakage(t, n_ions, r=r) for t in times])

    # Plot results
    fig, ax = plt.subplots(figsize=[8, 8])
    ax.plot(
        evolution_full.results["time"],
        evolution_full.results["leakage"],
    )

    # ax.plot(
    #     times,
    #     leakages,
    # )

    for E in Es:
        ax.plot(times, E, linestyle="dashed")

    ax.legend(
        [
            "Full model",
            # "Truncated Dyson series"
        ]
        + [f"Analytical leakage, $r={r}$" for r in omega_eff_prefactors]
        # + [f"Analytical leakage, $\\delta\\omega ={r}$" for r in delta_omegas]
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Leakage")

    ax.plot()
    plt.show()
