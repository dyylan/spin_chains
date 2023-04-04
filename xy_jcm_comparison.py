from json.tool import main
import quimb
import numpy as np
import matplotlib.pyplot as plt
from quimb.core import expectation

from spin_chains.quantum import iontrap, chains, states
from spin_chains.data_analysis import data_handling, data_plots


class SpinBosonHamJCM:
    def __init__(self, n_ions, J, h, g, delta):
        self.n_ions = n_ions
        self.dims = [2] * (n_ions + 1)
        self.J = J
        self.h = h
        self.g = g
        self.delta = delta
        self.h_xy = self.generate_xy_hamiltonian()
        self.h_jcm = self.generate_jcm_hamiltonian()

    def __call__(self, t):
        e = (
            2
            * np.power(self.g, 2)
            * (1 - np.cos(self.delta * t))
            / np.power(self.delta, 2)
        )
        ham = self.h_xy + (e * self.h_jcm)
        return ham

    # @quimb.hamiltonian_builder
    def generate_xy_hamiltonian(self):
        all_terms = []
        Sx = quimb.spin_operator("x", sparse=True)
        Sy = quimb.spin_operator("y", sparse=True)
        Sz = quimb.spin_operator("z", sparse=True)
        for i in range(n_ions):
            for j in range(n_ions):
                if i != j:
                    all_terms.append(
                        quimb.ikron([4 * self.J * Sx, Sx], self.dims, [i, j])
                    )
                    all_terms.append(
                        quimb.ikron([4 * self.J * Sy, Sy], self.dims, [i, j])
                    )
                else:
                    all_terms.append(quimb.ikron([2 * self.h * Sz], self.dims, [i]))
        H = sum(all_terms)
        return H

    def generate_jcm_hamiltonian(self):
        all_terms = []
        Sx = quimb.spin_operator("x", sparse=True)
        Sy = quimb.spin_operator("y", sparse=True)
        for i in range(n_ions):
            all_terms.append(
                quimb.ikron([8 * self.g * Sx, Sx], self.dims, [i, self.n_ions])
            )
            all_terms.append(
                quimb.ikron([8 * self.g * Sy, Sy], self.dims, [i, self.n_ions])
            )
        H = sum(all_terms)
        return H

    # @quimb.hamiltonian_builder
    def generate_marked_term(self, dims, gamma, mark_site=1):
        Sz = quimb.spin_operator("Z", sparse=True)
        return quimb.ikron(2 * gamma * Sz, dims, inds=[mark_site])


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
            else:
                all_terms.append(quimb.ikron([r * 2 * hs[i] * Sz], dims, [i]))
    H = sum(all_terms)
    return H


def xyh_jcm_hamiltonian(n_ions, J, h, g):
    dims = [2] * (n_ions + 1)
    all_terms = []
    Sx = quimb.spin_operator("x", sparse=True)
    Sy = quimb.spin_operator("y", sparse=True)
    Sz = quimb.spin_operator("z", sparse=True)
    for i in range(n_ions):
        for j in range(n_ions):
            if i != j:
                all_terms.append(quimb.ikron([4 * J * Sx, Sx], dims, [i, j]))
                all_terms.append(quimb.ikron([4 * J * Sy, Sy], dims, [i, j]))
            else:
                all_terms.append(quimb.ikron([2 * h * Sz], dims, [i]))
        all_terms.append(quimb.ikron([-8 * g * Sx, Sx], dims, [i, n_ions]))
        all_terms.append(quimb.ikron([-8 * g * Sy, Sy], dims, [i, n_ions]))
    H = sum(all_terms)
    return H


def xy_jcm_time_independent_fidelity(
    times, n_ions, Js, hs, gs, Delta, leakage_amplitude
):
    leakage_prefactor_xy = 1
    leakage_prefactor_jcm = leakage_amplitude
    up = quimb.qu([1, 0], qtype="ket")
    down = quimb.qu([0, 1], qtype="ket")
    spin_xy = [up] + [down for _ in range(n_ions - 1)]
    spin_jcm = [up] + [down for _ in range(n_ions)]

    init_spin_xy = quimb.kron(*spin_xy)
    init_spin_jcm = quimb.kron(*spin_jcm)

    ham_xy = xyh_hamiltonian(
        n_ions=n_ions,
        js=Js,
        hs=hs,
        r=r * leakage_prefactor_xy,
    )
    ham_jcm = xyh_jcm_hamiltonian(
        n_ions=n_ions,
        J=leakage_prefactor_xy * Js[0][0],
        h=leakage_prefactor_xy * hs[0],
        g=gs[0] * leakage_prefactor_jcm / 2,
    )

    evolution_xy = quimb.Evolution(init_spin_xy, ham_xy)
    evolution_jcm = quimb.Evolution(init_spin_jcm, ham_jcm)

    def calc_fidelity(t, pt):
        rho = quimb.partial_trace(pt, [2] * (n_ions + 1), [i for i in range(n_ions)])
        evolution_xy.update_to(t)
        pt_xy = evolution_xy.pt
        return quimb.expectation(pt_xy, rho)

    fidelities = []
    c = 0
    count = 1000
    for t in times:
        evolution_jcm.update_to(t)
        f = calc_fidelity(t, evolution_jcm.pt)
        fidelities.append(f)
        c += 1
        if not c % count:
            print(f"--> {count} time steps completed <--")

    fig, ax = plt.subplots(figsize=[8, 8])

    ax.plot(times, fidelities)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fidelity")

    ax.plot()
    plt.show()


def xy_jcm_time_dependent_fidelity(times, n_ions, Js, hs, gs, Delta, leakage_amplitude):
    leakage_prefactor_xy = 1
    leakage_prefactor_jcm = leakage_amplitude
    up = quimb.qu([1, 0], qtype="ket")
    down = quimb.qu([0, 1], qtype="ket")
    spin_xy = [up] + [down for _ in range(n_ions - 1)]
    spin_jcm = [up] + [down for _ in range(n_ions)]

    init_spin_xy = quimb.kron(*spin_xy)
    init_spin_jcm = quimb.kron(*spin_jcm)

    ham_xy = xyh_hamiltonian(
        n_ions=n_ions,
        js=Js,
        hs=hs,
        r=r * leakage_prefactor_xy,
    )
    ham_t = SpinBosonHamJCM(
        n_ions=n_ions,
        J=Js[0][0],
        h=hs[0],
        g=gs[0] / 2,
        delta=Delta,
    )

    evolution_xy = quimb.Evolution(init_spin_xy, ham_xy)

    def calc_t(t, _):
        return t

    def calc_fidelity(t, pt):
        rho = quimb.partial_trace(pt, [2] * (n_ions + 1), [i for i in range(n_ions)])
        evolution_xy.update_to(t)
        pt_xy = evolution_xy.pt
        return quimb.expectation(pt_xy, rho)

    compute = {"time": calc_t, "fidelity": calc_fidelity}
    evolution_full = quimb.Evolution(
        init_spin_jcm, ham_t, progbar=True, compute=compute, method="integrate"
    )

    evolution_full.update_to(times[-1])

    fig, ax = plt.subplots(figsize=[8, 8])
    ax.plot(
        evolution_full.results["time"],
        evolution_full.results["fidelity"],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fidelity")

    ax.plot()
    plt.show()


def xy_jcm_time_independent_init_state_overlap(
    times, n_ions, Js, hs, gs, r, leakage_amplitude
):
    up = quimb.qu([1, 0], qtype="ket")
    down = quimb.qu([0, 1], qtype="ket")
    spin_xy = [up] + [down for _ in range(n_ions - 1)]
    spin_jcm = [up] + [down for _ in range(n_ions)]

    init_spin_xy = quimb.kron(*spin_xy)
    init_spin_xy_r1 = quimb.kron(*spin_xy)
    init_spin_jcm = quimb.kron(*spin_jcm)

    ham_xy = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs, r=r)
    ham_xy_r1 = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs, r=1)
    ham_jcm = xyh_jcm_hamiltonian(
        n_ions=n_ions, J=Js[0][0], h=hs[0], g=gs[0] * leakage_amplitude / 2
    )

    evolution_xy = quimb.Evolution(init_spin_xy, ham_xy)
    evolution_xy_r1 = quimb.Evolution(init_spin_xy, ham_xy_r1)
    evolution_jcm = quimb.Evolution(init_spin_jcm, ham_jcm)

    fidelities_xy = []
    fidelities_xy_r1 = []
    fidelities_jcm = []
    c = 0
    count = 1000
    for t in times:
        evolution_xy.update_to(t)
        evolution_xy_r1.update_to(t)
        evolution_jcm.update_to(t)
        fidelities_xy.append(quimb.expectation(evolution_xy.pt, init_spin_xy))
        fidelities_xy_r1.append(quimb.expectation(evolution_xy_r1.pt, init_spin_xy))
        fidelities_jcm.append(quimb.expectation(evolution_jcm.pt, init_spin_jcm))
        c += 1
        if not c % count:
            print(f"--> {count} time steps completed <--")

    fig, ax = plt.subplots(figsize=[8, 8])

    ax.plot(times, fidelities_jcm)
    ax.plot(times, fidelities_xy)
    ax.plot(times, fidelities_xy_r1)

    ax.legend(["$H_{JCM}$", "$H_{XY}$", f"$H_{{XY}}, r={r}$"])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fidelity")

    ax.plot()
    plt.show()


def xy_jcm_time_dependent_init_state_overlap(
    times, n_ions, Js, hs, gs, Delta, leakage_amplitude
):
    leakage_prefactor_xy = 1
    leakage_prefactor_jcm = leakage_amplitude
    up = quimb.qu([1, 0], qtype="ket")
    down = quimb.qu([0, 1], qtype="ket")
    spin_xy = [up] + [down for _ in range(n_ions - 1)]
    spin_jcm = [up] + [down for _ in range(n_ions)]

    init_spin_xy = quimb.kron(*spin_xy)
    init_spin_jcm = quimb.kron(*spin_jcm)

    ham_t = SpinBosonHamJCM(
        n_ions=n_ions,
        J=Js[0][0],
        h=hs[0],
        g=gs[0] / 2,
        delta=Delta,
    )

    def calc_t(t, _):
        return t

    def calc_overlap(t, pt):
        rho = quimb.partial_trace(pt, [2] * (n_ions + 1), [i for i in range(n_ions)])
        return quimb.expectation(init_spin_xy, rho)

    compute = {"time": calc_t, "overlap": calc_overlap}
    evolution_full = quimb.Evolution(
        init_spin_jcm, ham_t, progbar=True, compute=compute, method="integrate"
    )

    evolution_full.update_to(times[-1])

    ham_xy = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs, r=r)
    ham_jcm = xyh_jcm_hamiltonian(
        n_ions=n_ions, J=Js[0][0], h=hs[0], g=gs[0] * leakage_amplitude / 2
    )

    evolution_xy = quimb.Evolution(init_spin_xy, ham_xy)
    evolution_jcm = quimb.Evolution(init_spin_jcm, ham_jcm)

    fidelities_xy = []
    fidelities_jcm = []
    c = 0
    count = 1000
    for t in times:
        evolution_xy.update_to(t)
        evolution_jcm.update_to(t)
        fidelities_xy.append(quimb.expectation(evolution_xy.pt, init_spin_xy))
        fidelities_jcm.append(quimb.expectation(evolution_jcm.pt, init_spin_jcm))
        c += 1
        if not c % count:
            print(f"--> {count} time steps completed <--")

    # print(times)
    # print(evolution_full.results["time"])
    # print(evolution_full.results["overlap"])

    fig, ax = plt.subplots(figsize=[8, 8])
    ax.plot(
        evolution_full.results["time"],
        evolution_full.results["overlap"],
    )
    ax.plot(times, fidelities_xy)
    # ax.plot(times, fidelities_jcm)

    ax.legend(["Time dependent $H_{JCM}$", "$H_{XY}$", "$H_{JCM}$"])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Init state fidelity")

    ax.plot()
    plt.show()


if __name__ == "__main__":
    n_ions = 8
    target_alpha = 0.2
    times = list(np.linspace(0, 0.001, 1000))
    # r = 0.9567597060075298
    # r = 1 / 0.9567597060075298
    r = 0.95

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
    Delta = omega_eff - omega_single_mode
    leakage_amplitude = np.power(gs[0] / (np.sqrt(2) * Delta), 2)
    # leakage_amplitude = gs[0] / (np.sqrt(2) * Delta)

    print(f"single mode J = {Js[0][0]}")
    print(f"single mode h = {hs[0]}")
    print(f"single mode g = {gs[0]/2}")
    print(f"Delta = {Delta}")
    print(f"leakage_amplitude = {leakage_amplitude}")

    # xy_jcm_time_independent_fidelity(
    #     times, n_ions, Js, hs, gs, Delta, leakage_amplitude
    # )
    # xy_jcm_time_dependent_fidelity(times, n_ions, Js, hs, gs, Delta, leakage_amplitude)

    # xy_jcm_time_dependent_init_state_overlap(
    #     times, n_ions, Js, hs, gs, Delta, leakage_amplitude
    # )
    xy_jcm_time_independent_init_state_overlap(
        times, n_ions, Js, hs, gs, 0.95, leakage_amplitude
    )
