import quimb
import numpy as np
import matplotlib.pyplot as plt
from quimb.core import expectation

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
                quimb.ikron(
                    [a, self.gs[i][mode] * s], self.dims, [mode, i + 2], sparse=True
                )
                for i in range(self.n_ions)
            ]
        )
        return ham

    # @quimb.hamiltonian_builder
    def add_marked_term(self, gamma, mark_site=1):
        self.marked_terms += self.generate_marked_term(self.dims, gamma, mark_site)
        # self.marked_terms += self.generate_marked_term(self.dims, 0, mark_site)

    # @quimb.hamiltonian_builder
    def generate_marked_term(self, dims, gamma, mark_site=1):
        Sz = quimb.spin_operator("Z", sparse=True)
        return quimb.ikron(2 * gamma * Sz, dims, inds=[mark_site + 1], sparse=True)


def xy_hamiltonian(n_ions, js, rJ=1):
    dims = [2] * n_ions
    all_terms = []
    Sx = quimb.spin_operator("x", sparse=True)
    Sy = quimb.spin_operator("y", sparse=True)
    for i in range(n_ions):
        for j in range(n_ions):
            if i != j:
                all_terms.append(
                    quimb.ikron([4 * rJ * js[i][j] * Sx, Sx], dims, [i, j], sparse=True)
                )
                all_terms.append(
                    quimb.ikron([4 * rJ * js[i][j] * Sy, Sy], dims, [i, j], sparse=True)
                )
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
                all_terms.append(
                    quimb.ikron([4 * js[i][j] * Sx, Sx], dims, [i, j], sparse=True)
                )
                all_terms.append(
                    quimb.ikron([4 * js[i][j] * Sy, Sy], dims, [i, j], sparse=True)
                )
        all_terms.append(quimb.ikron([2 * hs[i] * Sz], dims, [i], sparse=True))
    H = sum(all_terms)
    return H


if __name__ == "__main__":
    n_ions = 8
    n_phonons = 4
    # times_microseconds = [1000]  #
    # times_microseconds = [i * 100 for i in range(0, 101, 1)]
    times_microseconds = [0, 1000, 2000, 3000]
    times_microseconds = [0, 1000]
    times_microseconds = [2500, 3000]
    # times_microseconds = [
    #     2000,
    #     2400,
    #     2800,
    #     3200,
    #     3600,
    #     4000,
    #     4400,
    #     4800,
    #     5000,
    # ]
    target_alpha = 0.2
    plot_steps = True
    update_data = True
    correct_hs = True
    calc_parameter = "fidelity"
    rs = [(1.000338, 1), (1.000338, 1.00015), (1.000338, 1.000338)]
    rs = [1.0006, 1.0007, 1.0008]
    rs = [1.000206, 1.000306, 1.000406]
    # rJ = 0.9716238998156347
    # rJ = 0.9567597060075298
    # rJ = 0.960
    # rJ = 0.948
    # rJ = 0.92
    rJ = 0.975
    # rJ = False
    # r_omega = 1.000123
    # r_omega = 1.000406
    # r_omega = 1.000306
    # r_omega = 1.000315
    # r_omega = 1.000338
    # r_omega = 1.000310  # n_ions=8
    # r_omega = 1
    r_omega = False

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
        _r0 = (1 + r0) / 2
        _r1 = (1 + r1) / 2
        return (
            gi[0]
            * gj[0]
            * omega_modes[0]
            * (1 / (8 * ((_r0 * omega_eff) ** 2 - omega_modes[0] ** 2)))
        ) + (
            gi[1]
            * gj[1]
            * omega_modes[1]
            * (1 / (8 * ((_r1 * omega_eff) ** 2 - omega_modes[1] ** 2)))
        )

    def compute_h(gi, r0=1, r1=1):
        _r0 = (1 + r0) / 2
        _r1 = (1 + r1) / 2
        return (
            gi[0] ** 2
            * _r0
            * omega_eff
            / (4 * ((_r0 * omega_eff) ** 2 - omega_modes[0] ** 2))
        ) + (
            gi[1] ** 2
            * _r1
            * omega_eff
            / (4 * ((_r1 * omega_eff) ** 2 - omega_modes[1] ** 2))
        )

    if rJ:
        Js = [[rJ * compute_J(gi, gj, 1, 1) for gi in gs] for gj in gs]
        hs = [rJ * compute_h(gi, 1, 1) for gi in gs]

        J_ratios = [
            [rJ * compute_J(gi, gj, 1, 1) / compute_J(gi, gj, 1, 1) for gi in gs]
            for gj in gs
        ]
        h_ratios = [rJ * compute_h(gi, 1, 1) / compute_h(gi) for gi in gs]

        # J_ratios = [
        #     [
        #         compute_J(gi, gj, 1.000306, 1.000306) / compute_J(gi, gj, 1, 1)
        #         for gi in gs
        #     ]
        #     for gj in gs
        # ]
        # h_ratios = [compute_h(gi, 1.000306, 1.000306) / compute_h(gi) for gi in gs]

        print(f"J ratios = {J_ratios} \n--- avg. {np.mean(J_ratios)}")
        print(f"h ratios = {h_ratios} \n--- avg. {np.mean(h_ratios)}")
    elif r_omega:
        Js = [[compute_J(gi, gj, r_omega, 1) for gi in gs] for gj in gs]
        hs = [compute_h(gi, r_omega, 1) for gi in gs]

        J_ratios = [
            [Js[j][i] / compute_J(gi, gj, 1, 1) for i, gi in enumerate(gs)]
            for j, gj in enumerate(gs)
        ]
        rJ = np.mean(J_ratios) if np.mean(J_ratios) != 1.0 else False
        print(f"{rJ}")
        print(f"J ratios = {J_ratios} \n--- avg. {np.mean(J_ratios)}")
    else:
        Js = [[compute_J(gi, gj) for gi in gs] for gj in gs]
        hs = [compute_h(gi) for gi in gs]

    Js_r1 = [[compute_J(gi, gj, rs[0]) for gi in gs] for gj in gs]
    Js_r2 = [[compute_J(gi, gj, rs[1]) for gi in gs] for gj in gs]
    Js_r3 = [[compute_J(gi, gj, rs[2]) for gi in gs] for gj in gs]

    hs_r1 = [compute_h(gi, rs[0]) for gi in gs]
    hs_r2 = [compute_h(gi, rs[1]) for gi in gs]
    hs_r3 = [compute_h(gi, rs[2]) for gi in gs]

    # Js_r1 = [[compute_J(gi, gj, rs[0], rs[0]) for gi in gs] for gj in gs]
    # Js_r2 = [[compute_J(gi, gj, rs[1], rs[1]) for gi in gs] for gj in gs]
    # Js_r3 = [[compute_J(gi, gj, rs[2], rs[2]) for gi in gs] for gj in gs]

    # hs_r1 = [compute_h(gi, rs[0], rs[0]) for gi in gs]
    # hs_r2 = [compute_h(gi, rs[1], rs[1]) for gi in gs]
    # hs_r3 = [compute_h(gi, rs[2], rs[2]) for gi in gs]

    # print(f"two mode Js = {Js}")
    # print(f"two mode hs = {hs}")

    # Optimal gamma
    gamma = 11792.258802298766

    # Initial state
    up = quimb.qu([1, 0], qtype="ket")
    down = quimb.qu([0, 1], qtype="ket")
    init_spin_l = [up] + [down for _ in range(n_ions - 1)]
    final_spin_l = [down for _ in range(n_ions - 1)] + [up]
    groundstate_l = [down for _ in range(n_ions)]

    correct_hs_tag = "_hs_correction" if correct_hs else ""

    for i, time in enumerate(times_microseconds[:-1]):
        if time:
            states_full = data_handling.read_state_data(
                "spin_boson_two_mode",
                "open",
                target_alpha,
                n_ions,
                mu[0],
                z_trap_frequency,
                "full_state" + correct_hs_tag,
                time,
                r=rJ,
            )
            states_xy = data_handling.read_state_data(
                "spin_boson_two_mode",
                "open",
                target_alpha,
                n_ions,
                mu[0],
                z_trap_frequency,
                "xy_state" + correct_hs_tag,
                time,
                r=rJ,
            )
            psi0 = quimb.qu(states_full, qtype="ket")
            init_spin_xy = quimb.qu(states_xy, qtype="ket")
        else:
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
            init_t=time * 0.000001,
        )

        if correct_hs:
            for ion in range(1, n_ions + 1):
                ham_t.add_marked_term(-hs[ion - 1], ion)

        if correct_hs:
            ham_xy = xy_hamiltonian(n_ions=n_ions, js=Js)
            ham_xy_r1 = xy_hamiltonian(n_ions=n_ions, js=Js_r1)
            ham_xy_r2 = xy_hamiltonian(n_ions=n_ions, js=Js_r2)
            ham_xy_r3 = xy_hamiltonian(n_ions=n_ions, js=Js_r3)
        else:
            ham_xy = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs)
            ham_xy_r1 = xyh_hamiltonian(n_ions=n_ions, js=Js_r1, hs=hs_r1)
            ham_xy_r2 = xyh_hamiltonian(n_ions=n_ions, js=Js_r2, hs=hs_r2)
            ham_xy_r3 = xyh_hamiltonian(n_ions=n_ions, js=Js_r3, hs=hs_r3)

        evolution_xy = quimb.Evolution(init_spin_xy, ham_xy)
        evolution_xy_r1 = quimb.Evolution(init_spin_xy, ham_xy_r1)
        evolution_xy_r2 = quimb.Evolution(init_spin_xy, ham_xy_r2)
        evolution_xy_r3 = quimb.Evolution(init_spin_xy, ham_xy_r3)

        def calc_t(t, _):
            return t + (time * 0.000001)

        def calc_fidelity(t, pt):
            rho = quimb.partial_trace(
                pt,
                [n_phonons] + [n_phonons] + ([2] * n_ions),
                [i for i in range(2, n_ions + 2, 1)],
            )
            evolution_xy.update_to(t)
            pt_xy = evolution_xy.pt
            return abs(quimb.expectation(pt_xy, rho))

        def calc_fidelity_r1(t, pt):
            rho = quimb.partial_trace(
                pt,
                [n_phonons] + [n_phonons] + ([2] * n_ions),
                [i for i in range(2, n_ions + 2, 1)],
            )
            evolution_xy_r1.update_to(t)
            pt_xy = evolution_xy_r1.pt
            return abs(quimb.expectation(pt_xy, rho))

        def calc_fidelity_r2(t, pt):
            rho = quimb.partial_trace(
                pt,
                [n_phonons] + [n_phonons] + ([2] * n_ions),
                [i for i in range(2, n_ions + 2, 1)],
            )
            evolution_xy_r2.update_to(t)
            pt_xy = evolution_xy_r2.pt
            return abs(quimb.expectation(pt_xy, rho))

        def calc_fidelity_r3(t, pt):
            rho = quimb.partial_trace(
                pt,
                [n_phonons] + [n_phonons] + ([2] * n_ions),
                [i for i in range(2, n_ions + 2, 1)],
            )
            evolution_xy_r3.update_to(t)
            pt_xy = evolution_xy_r3.pt
            return abs(quimb.expectation(pt_xy, rho))

        def calc_purity(t, pt):
            rho = quimb.partial_trace(
                pt,
                [n_phonons] + [n_phonons] + ([2] * n_ions),
                [i for i in range(2, n_ions + 2, 1)],
            )
            return quimb.trace(rho @ rho)

        def calc_leakage(t, pt):
            rho = quimb.partial_trace(
                pt,
                [n_phonons] + [n_phonons] + ([2] * n_ions),
                [i for i in range(2, n_ions + 2, 1)],
            )
            return abs(quimb.expectation(groundstate, rho))

        calc_dictionary = {
            "purity": calc_purity,
            "fidelity": calc_fidelity,
            "leakage": calc_leakage,
        }

        if calc_parameter == "fidelity_rs":
            compute = {
                "time": calc_t,
                "fidelity": calc_fidelity,
                "fidelity_r1": calc_fidelity_r1,
                "fidelity_r2": calc_fidelity_r2,
                "fidelity_r3": calc_fidelity_r3,
            }
        else:
            compute = {"time": calc_t, calc_parameter: calc_dictionary[calc_parameter]}
        # Evolution
        evolution_full = quimb.Evolution(
            psi0, ham_t, progbar=True, compute=compute, method="integrate"
        )

        evolution_full.update_to(
            (times_microseconds[i + 1] - times_microseconds[i]) * 0.000001
        )

        print(f"Number of points = {len(evolution_full.results['time'])}")

        data_dict = {
            "time_microseconds": times_microseconds[i + 1],
            "xy_state" + correct_hs_tag: evolution_xy.pt,
            "full_state" + correct_hs_tag: evolution_full.pt,
        }
        if update_data:
            if calc_parameter == "fidelity_rs":
                data_handling.update_data(
                    protocol="spin_boson_two_mode",
                    chain="open",
                    alpha=target_alpha,
                    save_tag=f"{calc_parameter}{correct_hs_tag}_n={n_ions}_mu={mu[0]}_z_trap={z_trap_frequency}",
                    data_dict={
                        "time": evolution_full.results["time"],
                        "fidelity": evolution_full.results["fidelity"],
                        "fidelity_r1": evolution_full.results["fidelity_r1"],
                        "fidelity_r2": evolution_full.results["fidelity_r2"],
                        "fidelity_r3": evolution_full.results["fidelity_r3"],
                    },
                    replace=True,
                    replace_col="time",
                )
            else:
                data_handling.update_data(
                    protocol="spin_boson_two_mode",
                    chain="open",
                    alpha=target_alpha,
                    save_tag=f"{calc_parameter}{correct_hs_tag}_n={n_ions}_mu={mu[0]}_z_trap={z_trap_frequency}"
                    + (f"_r={rJ}" if rJ else ""),
                    data_dict={
                        "time": evolution_full.results["time"],
                        calc_parameter: evolution_full.results[calc_parameter],
                    },
                    replace=True,
                    replace_col="time",
                )
            if calc_parameter == "fidelity":
                data_handling.update_state_data(
                    protocol="spin_boson_two_mode",
                    chain="open",
                    alpha=target_alpha,
                    n_ions=n_ions,
                    mu=mu[0],
                    z_trap_frequency=z_trap_frequency,
                    data_dict=data_dict,
                    data_keys=[
                        "xy_state" + correct_hs_tag,
                        "full_state" + correct_hs_tag,
                    ],
                    r=rJ,
                    replace=True,
                )
            print(
                f">>> Updated state data from time {time * 0.000001} to time {times_microseconds[i+1] * 0.000001} <<<"
            )

        # Plot results
        if plot_steps:
            legend = []
            fig, ax = plt.subplots(figsize=[8, 8])
            if calc_parameter == "fidelity_rs":
                ax.plot(
                    evolution_full.results["time"],
                    evolution_full.results["fidelity"],
                )
                ax.plot(
                    evolution_full.results["time"],
                    evolution_full.results["fidelity_r1"],
                )
                ax.plot(
                    evolution_full.results["time"],
                    evolution_full.results["fidelity_r2"],
                )
                ax.plot(
                    evolution_full.results["time"],
                    evolution_full.results["fidelity_r3"],
                )
            else:
                ax.plot(
                    evolution_full.results["time"],
                    evolution_full.results[calc_parameter],
                )
            # ax.plot(
            #     evolution_full.results["time"],
            #     [r[0] for r in evolution_full.results[calc_parameter]],
            # )
            # ax.plot(
            #     evolution_full.results["time"],
            #     [r[1] for r in evolution_full.results[calc_parameter]],
            # )
            # ax.plot(
            #     evolution_full.results["time"],
            #     [sum(r) for r in evolution_full.results[calc_parameter]],
            # )

            if calc_parameter == "leakage":

                def calculate_leakage(t, ion=1, r0=1, r1=1):
                    E = (
                        (
                            (1 - np.cos((r0 * omega_eff - omega_modes[0]) * t))
                            * np.power(gs[ion - 1][0], 2)
                            / (2 * np.power(r0 * omega_eff - omega_modes[0], 2))
                        )
                        + (
                            (
                                1
                                - np.cos((r0 * omega_eff - omega_modes[0]) * t)
                                - np.cos((r1 * omega_eff - omega_modes[1]) * t)
                                + np.cos(((omega_modes[0]) - omega_modes[1]) * t)
                            )
                            * gs[ion - 1][0]
                            * gs[ion - 1][1]
                        )
                        / (
                            2
                            * (r0 * omega_eff - omega_modes[0])
                            * (r1 * omega_eff - omega_modes[1])
                        )
                        + (
                            (1 - np.cos((r1 * omega_eff - omega_modes[1]) * t))
                            * np.power(gs[ion - 1][1], 2)
                            / (2 * np.power(r1 * omega_eff - omega_modes[1], 2))
                        )
                    )
                    return E

                # Es_1 = [
                #     calculate_leakage(t, ion=1) for t in evolution_full.results["time"]
                # ]
                # Es_4 = [
                #     calculate_leakage(t, ion=4) for t in evolution_full.results["time"]
                # ]
                rs = [1.00030, 1.00032, 1.00034]
                E_averaged = [
                    np.mean([calculate_leakage(t, ion=i) for i in range(n_ions)])
                    for t in evolution_full.results["time"]
                ]
                Es_1 = [
                    np.mean(
                        [
                            calculate_leakage(t, ion=i, r0=rs[0], r1=rs[0])
                            for i in range(n_ions)
                        ]
                    )
                    for t in evolution_full.results["time"]
                ]
                Es_2 = [
                    np.mean(
                        [
                            calculate_leakage(t, ion=i, r0=rs[1], r1=rs[1])
                            for i in range(n_ions)
                        ]
                    )
                    for t in evolution_full.results["time"]
                ]
                Es_3 = [
                    np.mean(
                        [
                            calculate_leakage(t, ion=i, r0=rs[2], r1=rs[2])
                            for i in range(n_ions)
                        ]
                    )
                    for t in evolution_full.results["time"]
                ]
                # legend = [
                #     "$E(t)$",
                #     "$\\Vert \\mathcal{E}(t) \\Vert$ for ion 1",
                #     "$\\Vert \\mathcal{E}(t) \\Vert$ for ion 4",
                #     "$\\Vert \\mathcal{E}(t) \\Vert$ averaged",
                # ]
                legend = [
                    "$E(t)$",
                    "$\\Vert \\mathcal{E}(t) \\Vert$",
                    f"$\\Vert \\mathcal{{E}}(t) \\Vert$ for r={rs[0]}",
                    f"$\\Vert \\mathcal{{E}}(t) \\Vert$ for r={rs[1]}",
                    f"$\\Vert \\mathcal{{E}}(t) \\Vert$ for r={rs[2]}",
                ]
                # ax.plot(evolution_full.results["time"], Es_1, linestyle="dotted")
                # ax.plot(evolution_full.results["time"], Es_4, linestyle="dotted")
                ax.plot(evolution_full.results["time"], E_averaged, linestyle="dashed")
                ax.plot(evolution_full.results["time"], Es_1, linestyle="dashed")
                ax.plot(evolution_full.results["time"], Es_2, linestyle="dashed")
                ax.plot(evolution_full.results["time"], Es_3, linestyle="dashed")

            # ax.plot(times, fidelity)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel(calc_parameter.capitalize())

            if calc_parameter == "fidelity_rs":
                ax.set_ylabel("Fidelity")
                legend = ["$r = 1$"] + [f"$r = {r}$" for r in rs]
            else:
                ax.set_ylabel(calc_parameter.capitalize())

            ax.legend(legend)
            ax.plot()
            plt.show()
            # data_plots.plot_spin_boson_fidelity_comparisons(
            #     n_ions=8, n_phonons=4, alphas=[0.4]
            # )
