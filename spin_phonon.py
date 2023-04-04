import quimb
import numpy as np
import matplotlib.pyplot as plt
from quimb.core import expectation

from spin_chains.quantum import iontrap, chains, states
from spin_chains.data_analysis import data_handling, data_plots


class SpinBosonHamAC:
    def __init__(self, n_phonons, n_ions, B, gs, delta):
        self.n_ions = n_ions
        self.n_phonons = n_phonons
        self.dims = [n_phonons] + ([2] * n_ions)
        self.B = B
        self.gs = gs
        self.delta = delta
        self.h_interaction_term_1 = self.generate_interaction_term("destroy", "y")
        self.h_interaction_term_2 = self.generate_interaction_term("create", "y")
        self.h_interaction_term_3 = self.generate_interaction_term("destroy", "x")
        self.h_interaction_term_4 = self.generate_interaction_term("create", "x")

    def __call__(self, t):
        e = np.exp(1j * self.delta * t)
        e_d = np.exp(-1j * self.delta * t)
        c = np.cos(self.B * t)
        s = np.sin(self.B * t)
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


class SpinBosonHamMS:
    def __init__(
        self,
        n_phonons,
        n_ions,
        omega_eff,
        gs,
        omega_single_mode,
        init_t=0,
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
                quimb.ikron([a, self.gs[i] * s], self.dims, [0, i + 1], sparse=True)
                for i in range(self.n_ions)
            ]
        )
        return ham

    # @quimb.hamiltonian_builder
    def generate_marked_term(self, dims, gamma, mark_site=1):
        Sz = quimb.spin_operator("Z", sparse=True)
        return quimb.ikron(2 * gamma * Sz, dims, inds=[mark_site], sparse=True)


def xy_hamiltonian(n_ions, js):
    dims = [2] * n_ions
    all_terms = []
    for i in range(n_ions):
        for j in range(n_ions):
            if i != j:
                Sx = quimb.spin_operator("x", sparse=True)
                Sy = quimb.spin_operator("y", sparse=True)
                all_terms.append(
                    quimb.ikron([4 * js[i][j] * Sx, Sx], dims, [i, j], sparse=True)
                )
                all_terms.append(
                    quimb.ikron([4 * js[i][j] * Sy, Sy], dims, [i, j], sparse=True)
                )
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
                all_terms.append(
                    quimb.ikron([r * 4 * js[i][j] * Sx, Sx], dims, [i, j], sparse=True)
                )
                all_terms.append(
                    quimb.ikron([r * 4 * js[i][j] * Sy, Sy], dims, [i, j], sparse=True)
                )
        all_terms.append(quimb.ikron([r * 2 * hs[i] * Sz], dims, [i], sparse=True))
    H = sum(all_terms)
    return H


def xyhn_hamiltonian(n_ions, js, hs, r=1, n=0):
    dims = [2] * n_ions
    all_terms = []
    Sx = quimb.spin_operator("x", sparse=True)
    Sy = quimb.spin_operator("y", sparse=True)
    Sz = quimb.spin_operator("z", sparse=True)
    for i in range(n_ions):
        for j in range(n_ions):
            if i != j:
                all_terms.append(
                    quimb.ikron([r * 4 * js[i][j] * Sx, Sx], dims, [i, j], sparse=True)
                )
                all_terms.append(
                    quimb.ikron([r * 4 * js[i][j] * Sy, Sy], dims, [i, j], sparse=True)
                )
        all_terms.append(
            quimb.ikron([(r + (2 * n)) * 2 * hs[i] * Sz], dims, [i], sparse=True)
        )
    H = sum(all_terms)
    return H


def xyz_hamiltonian(n_ions, js, hs, r=1):
    dims = [2] * n_ions
    all_terms = []
    Sx = quimb.spin_operator("x", sparse=True)
    Sy = quimb.spin_operator("y", sparse=True)
    Sz = quimb.spin_operator("z", sparse=True)
    for i in range(n_ions):
        for j in range(n_ions):
            if i != j:
                all_terms.append(
                    quimb.ikron([r * 4 * js[i][j] * Sx, Sx], dims, [i, j], sparse=True)
                )
                all_terms.append(
                    quimb.ikron([r * 4 * js[i][j] * Sy, Sy], dims, [i, j], sparse=True)
                )
                all_terms.append(
                    quimb.ikron([r * 4 * js[i][j] * Sz, Sz], dims, [i, j], sparse=True)
                )
        all_terms.append(quimb.ikron([r * 2 * hs[i] * Sz], dims, [i], sparse=True))
    H = sum(all_terms)
    return H


if __name__ == "__main__":
    n_ions = 8
    n_phonons = 4
    times_microseconds = [
        0,
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
        5500,
        6000,
        6500,
        7000,
        7500,
        8000,
        8500,
        9000,
        9500,
        10000,
    ]
    times_microseconds = [i * 1000 for i in range(0, 2, 1)]
    target_alpha = 0.2
    plot_steps = True
    update_data = False
    Omega_factor = 1
    Omega = 2 * np.pi * 1e6
    # r_omega = 1.000338  # n_ions = 8
    r_omega = 1.000321  # n_ions = 10
    # r = 0.9567597060075298 # n_ions = 8
    # r = 0.9395086408877635  # n_ions = 10
    # r = 0.975
    # r = 0.970
    r = 1
    # r = 0.92
    calc_parameter = "fidelity"

    r = False if r == 1 else r
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
        Omega=Omega_factor * Omega,
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
    # Js = [
    #     [gi * gj * (1 / (8 * (omega_eff - omega_single_mode))) for gi in gs]
    #     for gj in gs
    # ]
    omega_eff_prime = omega_eff * (1 + r_omega) / 2

    J_r = (
        gs[0]
        * gs[0]
        * omega_single_mode
        * (1 / (8 * (omega_eff_prime ** 2 - omega_single_mode ** 2)))
    )

    print(f"single mode Js = {Js[0][0]}")
    print(f"single mode with r={r}, J = {J_r}")
    print(f"Js r_omega={r_omega}, ratio = {J_r/Js[0][0]}")
    print(
        f"4 order term size = {omega_single_mode * np.power(gs[0],4)/np.power(omega_eff - omega_single_mode, 4)}"
    )
    print(
        f"4 order term ratio = {(n_ions-1) * (np.power(gs[0],4)/np.power(omega_eff - omega_single_mode, 4))/Js[0][0]}"
    )
    print(
        f"4 order term ratio = {np.power(gs[0],2)*(omega_eff ** 2 - omega_single_mode ** 2)/(np.power(omega_eff - omega_single_mode, 4) * omega_single_mode)}"
    )
    print(
        f"4 order term ratio = {np.power(gs[0],2)/np.power(2*(omega_eff - omega_single_mode), 2)}"
    )
    # Optimal gamma
    gamma = 11792.258802298766

    # Initial state
    up = quimb.qu([1, 0], qtype="ket")
    down = quimb.qu([0, 1], qtype="ket")
    init_spin_l = [up] + [down for _ in range(n_ions - 1)]
    final_spin_l = [down for _ in range(n_ions - 1)] + [up]
    groundstate_l = [down for _ in range(n_ions)]

    for i, time in enumerate(times_microseconds[:-1]):
        if time:
            states_full = data_handling.read_state_data(
                "spin_boson_single_mode",
                "open",
                target_alpha,
                n_ions,
                mu[0],
                z_trap_frequency,
                "full_state",
                time,
                r=r,
            )
            states_xy = data_handling.read_state_data(
                "spin_boson_single_mode",
                "open",
                target_alpha,
                n_ions,
                mu[0],
                z_trap_frequency,
                "xy_state",
                time,
                r=r,
            )
            psi0 = quimb.qu(states_full, qtype="ket")
            init_spin_xy = quimb.qu(states_xy, qtype="ket")
        else:
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
            init_t=time * 0.000001,
        )

        # ham_xy = xy_hamiltonian(n_ions=n_ions, js=Js)
        if r:
            ham_xy = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs, r=r)
        else:
            # ham_xy = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs)
            # ham_xy = xyhn_hamiltonian(n_ions=n_ions, js=Js, hs=hs, r=1, n=0.04)
            ham_xy = xyz_hamiltonian(n_ions=n_ions, js=Js, hs=hs)

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

        def calc_purity(t, pt):
            rho = quimb.partial_trace(
                pt, [n_phonons] + ([2] * n_ions), [i for i in range(1, n_ions + 1, 1)]
            )
            return quimb.trace(rho @ rho)

        def calc_leakage(t, pt):
            rho = quimb.partial_trace(
                pt, [n_phonons] + ([2] * n_ions), [i for i in range(1, n_ions + 1, 1)]
            )
            return abs(quimb.expectation(groundstate, rho))

        # def calc_leakage(t, pt):
        #     rho = quimb.partial_trace(pt, [n_phonons] + ([2] * n_ions), [0])
        #     return abs(quimb.expectation(quimb.num(n_phonons), rho))

        calc_dictionary = {
            "purity": calc_purity,
            "fidelity": calc_fidelity,
            "leakage": calc_leakage,
        }

        compute = {"time": calc_t, calc_parameter: calc_dictionary[calc_parameter]}

        # Evolution
        evolution_full = quimb.Evolution(
            psi0, ham_t, progbar=True, compute=compute, method="integrate"
        )

        evolution_full.update_to((times_microseconds[i + 1] - time) * 0.000001)

        print(f"Number of points = {len(evolution_full.results['time'])}")

        data_dict = {
            "time_microseconds": times_microseconds[i + 1],
            "xy_state": evolution_xy.pt,
            "full_state": evolution_full.pt,
        }
        if update_data:
            data_handling.update_data(
                protocol="spin_boson_single_mode",
                chain="open",
                alpha=target_alpha,
                save_tag=f"{calc_parameter}_n={n_ions}_mu={mu[0]}_z_trap={z_trap_frequency}"
                + (f"_r={r}" if r else f""),
                # data_dict={
                #     "time": evolution_full.results["time"],
                #     calc_parameter: evolution_full.results[calc_parameter],
                # },
                data_dict={
                    "time": [
                        t + (time * 0.000001) for t in evolution_full.results["time"]
                    ],
                    calc_parameter: evolution_full.results[calc_parameter],
                },
                replace=True,
                replace_col="time",
            )
            if calc_parameter == "fidelity":
                data_handling.update_state_data(
                    protocol="spin_boson_single_mode",
                    chain="open",
                    alpha=target_alpha,
                    n_ions=n_ions,
                    mu=mu[0],
                    z_trap_frequency=z_trap_frequency,
                    data_dict=data_dict,
                    data_keys=["xy_state", "full_state"],
                    r=r,
                    replace=True,
                )
            print(
                f">>> Updated state data from time {time * 0.000001} to time {times_microseconds[i+1] * 0.000001} <<<"
            )

        # Plot results
        if plot_steps:
            fig, ax = plt.subplots(figsize=[8, 8])
            ax.plot(
                [t + time for t in evolution_full.results["time"]],
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

                # def calculate_leakage(t):
                #     E = (
                #         (1 - np.cos((omega_eff - omega_single_mode) * t))
                #         * np.power(gs[0], 2)
                #         / (2 * np.power(omega_eff - omega_single_mode, 2))
                #     )
                #     return E

                # def calculate_leakage(t, f=1):
                #     E = (
                #         (1 - np.cos(f * (omega_eff - omega_single_mode) * t))
                #         * np.power(gs[0], 2)
                #         / (2 * np.power(omega_eff - omega_single_mode, 2))
                #     )
                #     return E

                def calculate_leakage(t, N=1):
                    delta = omega_eff - omega_single_mode
                    wm = omega_single_mode
                    half = 1 / 2

                    # amp = (
                    #     2
                    #     * np.sin(half * t * sum_1)
                    #     * (
                    #         (-2 * t * wm * delta_1 * np.cos(half * t * delta_3))
                    #         + (4 * wm * np.sin(half * t * delta_3))
                    #         + (
                    #             sum_3 * np.sin(half * t * sum_1)
                    #             - (delta_1 * np.sin(half * t * sum_5))
                    #         )
                    #     )
                    #     / (np.power(delta_1, 4) * wm)
                    # )

                    amp = (
                        4
                        - (8 * np.cos(t * delta))
                        + (4 * np.cos(2 * t * delta))
                        # + (2 * t * delta * np.sin(2 * t * delta))
                    ) / (np.power(delta, 4))

                    E = (1 - np.cos((omega_eff - omega_single_mode) * t)) * np.power(
                        gs[0], 2
                    ) / (2 * np.power(omega_eff - omega_single_mode, 2)) + (
                        N * np.power(gs[0], 4) * amp / 16
                    )

                    return E

                Es = [
                    calculate_leakage(t, n_ions) for t in evolution_full.results["time"]
                ]

                ax.plot(evolution_full.results["time"], Es, linestyle="dashed")

            # ax.plot(times, fidelity)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel(calc_parameter.capitalize())

            ax.plot()
            plt.show()
            # data_plots.plot_spin_boson_fidelity_comparisons(
            #     n_ions=8, n_phonons=4, alphas=[0.4]
            # )
