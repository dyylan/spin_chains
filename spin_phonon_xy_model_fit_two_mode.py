import quimb
import numpy as np
import GPyOpt
import matplotlib.pyplot as plt
from quimb.core import expectation
from scipy.signal import find_peaks

from spin_chains.quantum import iontrap, chains, states
from spin_chains.data_analysis import data_handling, data_plots


class SpinBosonHamMS:
    def __init__(self, n_phonons, n_ions, omega_eff, gs, omega_modes, init_t=0):
        self.n_ions = n_ions
        self.n_phonons = n_phonons
        self.dims = [n_phonons] + [n_phonons] + ([2] * n_ions)
        self.omega_eff = omega_eff
        self.gs = gs
        self.omega_modes = omega_modes
        self.init_t = init_t
        self.h_interaction_term_01 = self.generate_interaction_term(0, "create", "x")
        self.h_interaction_term_02 = self.generate_interaction_term(0, "destroy", "x")
        self.h_interaction_term_03 = self.generate_interaction_term(0, "create", "y")
        self.h_interaction_term_04 = self.generate_interaction_term(0, "destroy", "y")
        self.h_interaction_term_11 = self.generate_interaction_term(1, "create", "x")
        self.h_interaction_term_12 = self.generate_interaction_term(1, "destroy", "x")
        self.h_interaction_term_13 = self.generate_interaction_term(1, "create", "y")
        self.h_interaction_term_14 = self.generate_interaction_term(1, "destroy", "y")
        self.marked_terms = 0

    def __call__(self, t):
        t += self.init_t
        e0 = np.exp(1j * self.omega_modes[0] * t)
        e1 = np.exp(1j * self.omega_modes[1] * t)
        e_d0 = np.exp(-1j * self.omega_modes[0] * t)
        e_d1 = np.exp(-1j * self.omega_modes[1] * t)
        c = np.cos(self.omega_eff * t)
        s = np.sin(self.omega_eff * t)
        ham = (
            -(e0 * c * self.h_interaction_term_01)
            - (e_d0 * c * self.h_interaction_term_02)
            + (e0 * s * self.h_interaction_term_03)
            + (e_d0 * s * self.h_interaction_term_04)
            - (e1 * c * self.h_interaction_term_11)
            - (e_d1 * c * self.h_interaction_term_12)
            + (e1 * s * self.h_interaction_term_13)
            + (e_d1 * s * self.h_interaction_term_14)
        ) + self.marked_terms
        return ham

    # @quimb.hamiltonian_builder
    def generate_interaction_term(self, mode, boson_op="create", spin_op="x"):
        s = quimb.spin_operator(spin_op, sparse=True)
        if boson_op in ["create", "c"]:
            a = quimb.create(self.n_phonons)
        else:
            a = quimb.destroy(self.n_phonons)
        ham = sum(
            [
                quimb.ikron([a, self.gs[i][mode] * s], self.dims, [mode, i + 2])
                for i in range(self.n_ions)
            ]
        )
        return ham

    def add_marked_term(self, gamma, mark_site=1):
        self.marked_terms += self.generate_marked_term(self.dims, gamma, mark_site)

    # @quimb.hamiltonian_builder
    def generate_marked_term(self, dims, gamma, mark_site=1):
        Sz = quimb.spin_operator("Z", sparse=True)
        return quimb.ikron(2 * gamma * Sz, dims, inds=[mark_site + 1])


def xy_hamiltonian(n_ions, js, r=1):
    dims = [2] * n_ions
    all_terms = []
    Sx = quimb.spin_operator("x", sparse=True)
    Sy = quimb.spin_operator("y", sparse=True)
    for i in range(n_ions):
        for j in range(n_ions):
            if i != j:
                all_terms.append(quimb.ikron([r * 4 * js[i][j] * Sx, Sx], dims, [i, j]))
                all_terms.append(quimb.ikron([r * 4 * js[i][j] * Sy, Sy], dims, [i, j]))
    H = sum(all_terms)
    return H


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


def optimise_r_BO(
    n_ions, n_phonons, omega_eff, gs, omega_single_mode, Js, hs, evolution_time, steps
):
    # Initial state
    up = quimb.qu([1, 0], qtype="ket")
    down = quimb.qu([0, 1], qtype="ket")
    init_spin_l = [up] + [down for _ in range(n_ions - 1)]
    final_spin_l = [down for _ in range(n_ions - 1)] + [up]
    groundstate_l = [down for _ in range(n_ions)]
    min_fid = []

    def cost(params):
        parameters = params.tolist()
        r = parameters[0][0]
        print(f"trial r = {r}")
        init_boson = quimb.qu([1, 0, 0, 0], qtype="ket")
        init_boson_2 = quimb.qu([1, 0, 0, 0], qtype="ket")
        init_spin_xy = quimb.kron(*init_spin_l)
        psi0 = quimb.kron(init_boson, init_boson_2, init_spin_xy)
        groundstate = quimb.kron(*groundstate_l)

        # Hamiltonian
        ham_t = SpinBosonHamMS(
            n_phonons=n_phonons,
            n_ions=n_ions,
            omega_eff=omega_eff,
            gs=gs,
            omega_modes=omega_modes,
            init_t=0,
        )

        for ion in range(1, n_ions):
            ham_t.add_marked_term(-hs[ion - 1], ion)

        ham_xy = xy_hamiltonian(n_ions=n_ions, js=Js, r=r)
        # ham_xy = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs, r=r)

        evolution_xy = quimb.Evolution(init_spin_xy, ham_xy)

        def calc_t(t, _):
            return t

        def calc_fidelity(t, pt):
            rho = quimb.partial_trace(
                pt,
                [n_phonons] + [n_phonons] + ([2] * n_ions),
                [i for i in range(2, n_ions + 2, 1)],
            )
            evolution_xy.update_to(t)
            pt_xy = evolution_xy.pt
            return quimb.expectation(pt_xy, rho)

        compute = {"time": calc_t, "fidelity": calc_fidelity}

        # Evolution
        evolution_full = quimb.Evolution(
            psi0, ham_t, progbar=True, compute=compute, method="integrate"
        )

        evolution_full.update_to(evolution_time * 0.000001)
        fidelity = evolution_full.results["fidelity"]
        times = evolution_full.results["time"]

        inverted_fidelity = [-1 * f for f in fidelity]

        peaks, _ = find_peaks(inverted_fidelity)

        min_times = [times[peak] for peak in peaks]
        # print(f"min times = {min_times}")

        min_fidelities = [fidelity[peak] for peak in peaks]
        # print(f"min fidelities = {min_fidelities}")

        min_fidelity = min(min_fidelities)
        print(f"min fidelity = {min_fidelity}")

        min_fid.append(min_fidelity)
        print(f"--> min fidelity = {min_fidelity} @ r = {r}")
        return 1 - min_fidelity

    bounds = [
        {
            "name": "r",
            "type": "continuous",
            "domain": (0.9, 1),
        }
    ]

    optimisation = GPyOpt.methods.BayesianOptimization(
        cost,
        domain=bounds,
        model_type="GP",
        acquisition_type="EI",
        normalize_Y=True,
        acquisition_weight=2,
        maximize=False,
    )

    max_iter = steps
    max_time = 30000
    optimisation.run_optimization(max_iter, max_time, verbosity=True)

    optimum_r = optimisation.x_opt[0]

    return optimum_r, min_fid


if __name__ == "__main__":
    n_ions = 8
    n_phonons = 4
    evolution_time = 400  # [i * 1 for i in range(0, 101, 1)]
    target_alpha = 0.2
    update_data = True
    calc_parameter = "fidelity"

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
    gs, omega_modes, omega_eff, deltas = ion_trap_xy.two_mode_ms()
    ion_trap_xy.calculate_alpha()
    print(f"XY Js = {ion_trap_xy.Js}")
    print(f"alpha = {ion_trap_xy.alpha}")
    # time = 40 * 2 * np.pi / delta
    print(
        f"-- Phonon mode 1:\n"
        f"g vector = {[g[0] for g in gs]}\ndelta = {deltas[0]}\nomega_eff = {omega_eff}\nomega_single_mode = {omega_modes[0]}\nomega_eff/delta = {omega_eff/deltas[0]}"
    )
    print(
        f"-- Phonon mode 2:\n"
        f"g vector = {[g[1] for g in gs]}\ndelta = {deltas[1]}\nomega_eff = {omega_eff}\nomega_single_mode = {omega_modes[1]}\nomega_eff/delta = {omega_eff/deltas[1]}"
    )
    Js = [
        [
            (
                gi[0]
                * gj[0]
                * omega_modes[0]
                * (1 / (8 * (omega_eff ** 2 - omega_modes[0] ** 2)))
            )
            + (
                gi[1]
                * gj[1]
                * omega_modes[1]
                * (1 / (8 * (omega_eff ** 2 - omega_modes[1] ** 2)))
            )
            for gi in gs
        ]
        for gj in gs
    ]
    hs = [
        (gi[0] ** 2 * omega_eff / (4 * (omega_eff ** 2 - omega_modes[0] ** 2)))
        + (gi[1] ** 2 * omega_eff / (4 * (omega_eff ** 2 - omega_modes[1] ** 2)))
        for gi in gs
    ]

    optimum_r, min_fid = optimise_r_BO(
        n_ions,
        n_phonons,
        omega_eff,
        gs,
        omega_modes,
        Js,
        hs,
        evolution_time,
        steps=100,
    )
    print(f"min fidelity = {max(min_fid)} @ r = {optimum_r} for alpha = {target_alpha}")
