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
    times_microseconds = [0, 100]  # [i * 1 for i in range(0, 101, 1)]
    target_alpha = 0.2
    plot_steps = True
    update_data = False
    calc_parameter = "phonons"

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
    # Js = [
    #     [gi * gj * (1 / (8 * (omega_eff - omega_single_mode))) for gi in gs]
    #     for gj in gs
    # ]
    print(f"single mode Js = {Js}")
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
            )
            psi0 = quimb.qu(states_full, qtype="ket")
            init_spin_xy = quimb.qu(states_xy, qtype="ket")
        else:
            zero_phonons = [1] + [0 for _ in range(n_phonons - 1)]
            init_boson = quimb.qu(zero_phonons, qtype="ket")
            init_spin_xy = quimb.kron(*init_spin_l)
            psi0 = quimb.kron(init_boson, init_spin_xy)
        groundstate = quimb.kron(*groundstate_l)

        if calc_parameter == "phonons":
            phonon_basis = []
            for phonon in range(n_phonons):
                phonon_basis.append(
                    quimb.qu(
                        [1 if phonon == i else 0 for i in range(n_phonons)], qtype="ket"
                    )
                )

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
        ham_xy = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs)

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

        def calc_phonons(t, pt):
            rho_phonons = quimb.partial_trace(pt, [n_phonons] + ([2] * n_ions), [0])
            expecs = [
                abs(quimb.expectation(phonon_basis[i], rho_phonons))
                for i in range(n_phonons)
            ]
            # return abs(quimb.expectation(phonon_basis[-1], rho_phonons))
            return expecs

        def calc_n_phonons(t, pt):
            rho_phonons = quimb.partial_trace(pt, [n_phonons] + ([2] * n_ions), [0])
            return abs(quimb.expectation(quimb.num(n_phonons), rho_phonons))

        calc_dictionary = {
            "purity": calc_purity,
            "fidelity": calc_fidelity,
            "leakage": calc_leakage,
            "phonons": calc_phonons,
            "n_phonons": calc_n_phonons,
        }

        compute = {"time": calc_t, calc_parameter: calc_dictionary[calc_parameter]}

        # Evolution
        evolution_full = quimb.Evolution(
            psi0, ham_t, progbar=True, compute=compute, method="integrate"
        )

        evolution_full.update_to(times_microseconds[i + 1] * 0.000001)

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
                save_tag=f"{calc_parameter}_n={n_ions}_n_phonons={n_phonons}_mu={mu[0]}_z_trap={z_trap_frequency}",
                data_dict={
                    "time": evolution_full.results["time"],
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
                    replace=True,
                )
            print(
                f">>> Updated state data from time {time * 0.000001} to time {times_microseconds[i+1] * 0.000001} <<<"
            )

        # Plot results
        if plot_steps:
            fig, ax = plt.subplots(figsize=[8, 8])
            if calc_parameter == "phonons":
                for i in range(n_phonons):
                    y = [result[i] for result in evolution_full.results[calc_parameter]]
                    ax.plot(evolution_full.results["time"], y)
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

                def calculate_leakage(t):
                    E = (
                        (1 - np.cos((omega_eff - omega_single_mode) * t))
                        * np.power(gs[0], 2)
                        / (2 * np.power(omega_eff - omega_single_mode, 2))
                    )
                    return E

                Es = [calculate_leakage(t) for t in evolution_full.results["time"]]
                ax.plot(evolution_full.results["time"], Es, linestyle="dashed")

            # ax.plot(times, fidelity)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel(calc_parameter.capitalize())

            if calc_parameter == "phonons":
                ax.legend([f"{i} phonons" for i in range(n_phonons)])
            ax.plot()
            plt.show()
            # data_plots.plot_spin_boson_fidelity_comparisons(
            #     n_ions=8, n_phonons=4, alphas=[0.4]
            # )
