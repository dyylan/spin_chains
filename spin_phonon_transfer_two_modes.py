import quimb
import numpy as np
import matplotlib.pyplot as plt

from spin_chains.quantum import iontrap, chains, states
from spin_chains.data_analysis import data_handling


class SpinBosonHamMStwomode:
    def __init__(
        self, n_phonons, n_ions, omega_eff, gs, omega_modes, init_t=0, marked_sites=0
    ):
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
        self.marked_terms = (
            self.generate_marked_term(marked_sites, 1)
            + self.generate_marked_term(marked_sites, n_ions)
            if marked_sites
            else 0
        )

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

    # @quimb.hamiltonian_builder
    def add_marked_term(self, gamma, mark_site=1):
        self.marked_terms += self.generate_marked_term(gamma / 2, mark_site)

    # @quimb.hamiltonian_builder
    def generate_marked_term(self, gamma, mark_site=1):
        Sz = quimb.spin_operator("Z", sparse=True)
        return quimb.ikron(2 * gamma * Sz, self.dims, inds=[mark_site + n_phonons - 1])


class SpinBosonHamMS:
    def __init__(
        self,
        n_phonons,
        n_ions,
        omega_eff,
        gs,
        omega_single_mode,
        init_t=0,
        marked_sites=0,
    ):
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
        self.marked_terms = (
            self.generate_marked_term(marked_sites, 1)
            + self.generate_marked_term(marked_sites, n_ions)
            if marked_sites
            else 0
        )

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
        ) + self.marked_terms
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
    def generate_marked_term(self, gamma, mark_site=1):
        Sz = quimb.spin_operator("Z", sparse=True)
        return quimb.ikron(2 * gamma * Sz, self.dims, inds=[mark_site])


def marked_hamiltonian(n_ions, marked_strength):
    dims = [n_phonons] + ([2] * n_ions)
    Sz = quimb.spin_operator("Z", sparse=True)
    H = quimb.ikron(2 * marked_strength * Sz, dims, inds=[1]) + quimb.ikron(
        2 * marked_strength * Sz, dims, inds=[n_ions]
    )
    return H


def marked_hamiltonian_spins_subspace(n_ions, marked_strength):
    dims = [2] * n_ions
    Sz = quimb.spin_operator("Z", sparse=True)
    H = quimb.ikron(2 * marked_strength * Sz, dims, inds=[0]) + quimb.ikron(
        2 * marked_strength * Sz, dims, inds=[n_ions - 1]
    )
    return H


def xy_hamiltonian(n_ions, js):
    dims = [2] * n_ions
    all_terms = []
    for i in range(n_ions):
        for j in range(n_ions):
            if i != j:
                Sx = quimb.spin_operator("x", sparse=True)
                Sy = quimb.spin_operator("y", sparse=True)
                all_terms.append(quimb.ikron([4 * js[i][j] * Sx, Sx], dims, [i, j]))
                all_terms.append(quimb.ikron([4 * js[i][j] * Sy, Sy], dims, [i, j]))
    H = sum(all_terms)
    return H


def xyh_hamiltonian(n_ions, js, hs):
    dims = [2] * n_ions
    all_terms = []
    Sx = quimb.spin_operator("x", sparse=True)
    Sy = quimb.spin_operator("y", sparse=True)
    Sz = quimb.spin_operator("z", sparse=True)
    for i in range(n_ions):
        for j in range(n_ions):
            if i != j:
                all_terms.append(quimb.ikron([4 * js[i][j] * Sx, Sx], dims, [i, j]))
                all_terms.append(quimb.ikron([4 * js[i][j] * Sy, Sy], dims, [i, j]))
            all_terms.append(quimb.ikron([2 * hs[j] * Sz], dims, [j]))
    H = sum(all_terms)
    return H


if __name__ == "__main__":
    n_ions = 8
    n_phonons = 4
    target_alpha = 0.2
    plot_steps = True
    update_data = True
    correct_hs = True
    time = 0.005

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

    def compute_J(gi, gj, r0=1, r1=1):
        return (
            gi[0]
            * gj[0]
            * omega_modes[0]
            * (1 / (8 * ((r0 * omega_eff) ** 2 - omega_modes[0] ** 2)))
        ) + (
            gi[1]
            * gj[1]
            * omega_modes[1]
            * (1 / (8 * ((r1 * omega_eff) ** 2 - omega_modes[1] ** 2)))
        )

    def compute_h(gi, r0=1, r1=1):
        return (
            gi[0] ** 2
            * r0
            * omega_eff
            / (4 * ((r0 * omega_eff) ** 2 - omega_modes[0] ** 2))
        ) + (
            gi[1] ** 2
            * r1
            * omega_eff
            / (4 * ((r1 * omega_eff) ** 2 - omega_modes[1] ** 2))
        )

    Js = [[compute_J(gi, gj) for gi in gs] for gj in gs]
    hs = [compute_h(gi) for gi in gs]

    print(f"two mode Js = {Js}")
    print(f"two mode hs = {hs}")

    # Optimal gamma
    gamma = 11792.258802298766

    # Initial state
    up = quimb.qu([1, 0], qtype="ket")
    down = quimb.qu([0, 1], qtype="ket")
    init_spin_l = [up] + [down for _ in range(n_ions - 1)]
    final_spin_l = [down for _ in range(n_ions - 1)] + [up]
    groundstate_l = [down for _ in range(n_ions)]
    init_boson = quimb.qu([1, 0, 0, 0], qtype="ket")
    init_boson_2 = quimb.qu([1, 0, 0, 0], qtype="ket")
    init_spin_xy = quimb.kron(*init_spin_l)
    final_spin_xy = quimb.kron(*final_spin_l)
    psi0_1mode = quimb.kron(init_boson, init_spin_xy)
    psi0_2mode = quimb.kron(init_boson, init_boson_2, init_spin_xy)
    groundstate = quimb.kron(*groundstate_l)

    # Hamiltonians
    ham_t_2mode = SpinBosonHamMStwomode(
        n_phonons=n_phonons,
        n_ions=n_ions,
        omega_eff=omega_eff,
        gs=gs,
        omega_modes=omega_modes,
        init_t=0,
        marked_sites=gamma,
    )

    ham_t_1mode = SpinBosonHamMS(
        n_phonons=n_phonons,
        n_ions=n_ions,
        omega_eff=omega_eff,
        gs=[g[0] for g in gs],
        omega_single_mode=omega_modes[0],
        init_t=0,
        marked_sites=gamma,
    )

    ham_xy_walk = xyh_hamiltonian(
        n_ions,
        Js,
        [gamma if (i == 0) or (i == n_ions - 1) else 0 for i in range(n_ions)],
    )

    ion_trap_xy.single_mode_ms()
    ion_trap_xy.calculate_alpha()
    print(f"XY Js = {ion_trap_xy.Js}")
    print(f"alpha = {ion_trap_xy.alpha}")
    # ham_xy = xy_hamiltonian(n_ions=n_ions, js=Js)

    def calc_t(t, _):
        return t

    def calc_fidelity_1mode(t, pt):
        rho = quimb.partial_trace(
            pt, [n_phonons] + ([2] * n_ions), [i for i in range(1, n_ions + 1, 1)]
        )
        return abs(quimb.expectation(final_spin_xy, rho))

    def calc_fidelity_2mode(t, pt):
        rho = quimb.partial_trace(
            pt,
            [n_phonons] + [n_phonons] + ([2] * n_ions),
            [i for i in range(2, n_ions + 2, 1)],
        )
        return abs(quimb.expectation(final_spin_xy, rho))

    def calc_fidelity_xy(t, pt):
        return abs(quimb.expectation(final_spin_xy, pt))

    compute_1mode = {"time": calc_t, "fidelity": calc_fidelity_1mode}
    compute_2mode = {"time": calc_t, "fidelity": calc_fidelity_2mode}
    compute_xy = {"time": calc_t, "fidelity": calc_fidelity_xy}

    # Full dynamics walk evolution

    evolution_2mode = quimb.Evolution(
        psi0_2mode, ham_t_2mode, progbar=True, compute=compute_2mode, method="integrate"
    )
    evolution_1mode = quimb.Evolution(
        psi0_1mode, ham_t_1mode, progbar=True, compute=compute_1mode, method="integrate"
    )
    evolution_xy = quimb.Evolution(
        init_spin_xy, ham_xy_walk, progbar=True, compute=compute_xy
    )

    evolution_2mode.update_to(time)
    evolution_1mode.update_to(time)
    evolution_xy.update_to(time)

    if update_data:
        data_handling.update_data(
            protocol="spin_boson_two_mode",
            chain="open",
            alpha=target_alpha,
            save_tag=f"transfer_fidelity_full_n={n_ions}_mu={mu[0]}_z_trap={z_trap_frequency}",
            data_dict={
                "time": evolution_2mode.results["time"],
                "fidelity": evolution_2mode.results["fidelity"],
            },
            replace=True,
        )
        data_handling.update_data(
            protocol="spin_boson_single_mode",
            chain="open",
            alpha=target_alpha,
            save_tag=f"transfer_fidelity_full_n={n_ions}_mu={mu[0]}_z_trap={z_trap_frequency}",
            data_dict={
                "time": evolution_1mode.results["time"],
                "fidelity": evolution_1mode.results["fidelity"],
            },
            replace=True,
        )
        data_handling.update_data(
            protocol="spin_boson_two_mode",
            chain="open",
            alpha=target_alpha,
            save_tag=f"transfer_fidelity_xy_n={n_ions}_mu={mu[0]}_z_trap={z_trap_frequency}",
            data_dict={
                "time": evolution_xy.results["time"],
                "fidelity": evolution_xy.results["fidelity"],
            },
            replace=True,
        )
    # Plot results
    fig, ax = plt.subplots(figsize=[8, 8])
    ax.plot(evolution_1mode.results["time"], evolution_1mode.results["fidelity"])
    ax.plot(evolution_2mode.results["time"], evolution_2mode.results["fidelity"])
    ax.plot(
        evolution_xy.results["time"],
        evolution_xy.results["fidelity"],
        linestyle="dotted",
    )

    ax.legend(["1 phonon mode", "2 phonon modes", "xy model"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fidelity of final state")

    ax.plot()
    plt.show()
