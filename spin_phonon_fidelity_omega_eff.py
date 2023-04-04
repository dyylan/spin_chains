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
    n_ions = 10
    n_phonons = 4
    times_microseconds = [0, 100]
    target_alpha = 0.2
    plot_steps = True
    update_data = False
    r = 0.9567597060075298
    # r = 0.975
    # r = 1
    # omega_eff_prefactor = 1.00017
    r_omega = 1.000321
    # delta_omega = 6388
    delta_omega = 0

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

    # omega_eff_prime = r_omega * omega_eff
    # omega_single_mode_prime = r_omega * omega_single_mode
    gs_prime = [g for g in gs]

    # omega_eff_prime = omega_eff + delta_omega
    # omega_single_mode_prime = omega_single_mode - delta_omega

    # Delta_eff = omega_eff_prime - omega_single_mode_prime
    # Sum_eff = omega_eff_prime + omega_single_mode_prime

    # Js_prime = [
    #     [
    #         0.9580 * gi * gj * omega_single_mode_prime * (1 / (8 * Sum_eff * Delta_eff))
    #         for gi in gs_prime
    #     ]
    #     for gj in gs_prime
    # ]

    omega_eff_prime = omega_eff * (1 + r_omega) / 2

    hs_prime = [
        gi ** 2 * omega_eff / (4 * (omega_eff_prime ** 2 - omega_single_mode ** 2))
        for gi in gs_prime
    ]

    Js_prime = [
        [
            gi
            * gj
            * omega_single_mode
            * (1 / (8 * (omega_eff_prime ** 2 - omega_single_mode ** 2)))
            for gi in gs_prime
        ]
        for gj in gs_prime
    ]

    print(f"single mode Js = {Js[0][0]}")
    print(f"single mode Js prime = {Js_prime[0][0]}")
    print(f"Js ratio = {Js_prime[0][0]/Js[0][0]}")

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
        # # Hamiltonian
        # ham_t_omega_eff = SpinBosonHamMS(
        #     n_phonons=n_phonons,
        #     n_ions=n_ions,
        #     omega_eff=omega_eff_prime,
        #     gs=gs,
        #     omega_single_mode=omega_single_mode,
        #     init_t=time * 0.000001,
        # )

        # ham_xy = xy_hamiltonian(n_ions=n_ions, js=Js)
        ham_xy = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs, r=1)
        ham_xy_r1 = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs, r=r)
        ham_xy_omega_eff = xyh_hamiltonian(n_ions=n_ions, js=Js_prime, hs=hs_prime, r=1)

        evolution_xy = quimb.Evolution(init_spin_xy, ham_xy)
        evolution_xy_r = quimb.Evolution(init_spin_xy, ham_xy_r1)
        evolution_xy_omega_eff = quimb.Evolution(init_spin_xy, ham_xy_omega_eff)

        xy_init_state = quimb.kron(*init_spin_l)

        def calc_t(t, _):
            return t

        def calc_fidelity(t, pt):
            rho = quimb.partial_trace(
                pt, [n_phonons] + ([2] * n_ions), [i for i in range(1, n_ions + 1, 1)]
            )
            evolution_xy.update_to(t)
            pt_xy = evolution_xy.pt
            return quimb.expectation(pt_xy, rho)

        def calc_fidelity_r(t, pt):
            rho = quimb.partial_trace(
                pt, [n_phonons] + ([2] * n_ions), [i for i in range(1, n_ions + 1, 1)]
            )
            evolution_xy_r.update_to(t)
            pt_xy_r = evolution_xy_r.pt
            return quimb.expectation(pt_xy_r, rho)

        def calc_fidelity_omega_eff(t, pt):
            rho = quimb.partial_trace(
                pt, [n_phonons] + ([2] * n_ions), [i for i in range(1, n_ions + 1, 1)]
            )
            evolution_xy_omega_eff.update_to(t)
            pt_xy_omega_eff = evolution_xy_omega_eff.pt
            return quimb.expectation(pt_xy_omega_eff, rho)

        compute = {
            "time": calc_t,
            "fidelity": calc_fidelity,
            "fidelity_r": calc_fidelity_r,
            "fidelity_omega_eff": calc_fidelity_omega_eff,
        }

        # Evolution
        evolution_full = quimb.Evolution(
            psi0, ham_t, progbar=True, compute=compute, method="integrate"
        )

        # evolution_full_omega_eff = quimb.Evolution(
        #     psi0, ham_t_omega_eff, progbar=True, compute=compute, method="integrate"
        # )

        evolution_full.update_to((times_microseconds[i + 1] - time) * 0.000001)
        # evolution_full_omega_eff.update_to(
        #     (times_microseconds[i + 1] - time) * 0.000001
        # )

        print(f"Number of points = {len(evolution_full.results['time'])}")

        data_dict = {
            "time_microseconds": times_microseconds[i + 1],
            "xy_state": evolution_xy.pt,
            "xy_r_state": evolution_xy_r.pt,
            "full_state": evolution_full.pt,
        }
        if update_data:
            data_handling.update_data(
                protocol="spin_boson_single_mode",
                chain="open",
                alpha=target_alpha,
                save_tag=f"fidelity_n={n_ions}_mu={mu[0]}_z_trap={z_trap_frequency}_r={r}_omega_eff={omega_eff}",
                # data_dict={
                #     "time": evolution_full.results["time"],
                #     calc_parameter: evolution_full.results[calc_parameter],
                # },
                data_dict={
                    "time": [
                        t + (time * 0.000001) for t in evolution_full.results["time"]
                    ],
                    "fidelity": evolution_full.results["fidelity"],
                    "fidelity_r": evolution_full.results["fidelity_r"],
                    "fidelity_omega_eff": evolution_full.results["fidelity_omega_eff"],
                },
                replace=True,
                replace_col="time",
            )
            data_handling.update_state_data(
                protocol="spin_boson_single_mode",
                chain="open",
                alpha=target_alpha,
                n_ions=n_ions,
                mu=mu[0],
                z_trap_frequency=z_trap_frequency,
                data_dict=data_dict,
                data_keys=["xy_state", "xy_r_state", "full_state"],
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
                evolution_full.results["time"],
                evolution_full.results["fidelity"],
            )
            ax.plot(
                evolution_full.results["time"],
                evolution_full.results["fidelity_r"],
            )
            ax.plot(
                evolution_full.results["time"],
                evolution_full.results["fidelity_omega_eff"],
            )
            ax.legend(
                [
                    "Full model",
                    f"Full model with $r = {r}$",
                    f"Full model with $\\omega_\\textrm{{eff}}^\\prime = {r_omega} \\omega_\\textrm{{eff}}$",
                ]
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Fidelity")

            ax.plot()
            plt.show()
