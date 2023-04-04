import quimb
import numpy as np
import matplotlib.pyplot as plt

from spin_chains.quantum import iontrap, chains, states
from spin_chains.data_analysis import data_handling


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


class SpinBosonHamMSStrobe:
    def __init__(
        self, n_phonons, n_ions, omega_eff, gs, omega_single_mode, gamma, strobe_time
    ):
        self.n_ions = n_ions
        self.n_phonons = n_phonons
        self.dims = [n_phonons] + ([2] * n_ions)
        self.omega_eff = omega_eff
        self.gs = gs
        self.omega_single_mode = omega_single_mode
        self.gamma = gamma
        self.strobe_time = strobe_time
        self.marked_term_1 = self.generate_marked_term(marked_site=1)
        self.marked_term_2 = self.generate_marked_term(marked_site=n_ions)
        self.h_interaction_term_1 = self.generate_interaction_term("create", "x")
        self.h_interaction_term_2 = self.generate_interaction_term("destroy", "x")
        self.h_interaction_term_3 = self.generate_interaction_term("create", "y")
        self.h_interaction_term_4 = self.generate_interaction_term("destroy", "y")

    def __call__(self, t):
        if t % (2 * self.strobe_time) < self.strobe_time:
            time = t % (2 * self.strobe_time)
            e = np.exp(1j * self.omega_single_mode * time)
            e_d = np.exp(-1j * self.omega_single_mode * time)
            c = np.cos(self.omega_eff * time)
            s = np.sin(self.omega_eff * time)
            ham = (
                -(e * c * self.h_interaction_term_1)
                - (e_d * c * self.h_interaction_term_2)
                + (e * s * self.h_interaction_term_3)
                + (e_d * s * self.h_interaction_term_4)
            )
        else:
            ham = self.marked_term_1 + self.marked_term_2
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
    def generate_marked_term(self, marked_site=1):
        Sz = quimb.spin_operator("Z", sparse=True)
        return quimb.ikron(2 * self.gamma * Sz, self.dims, inds=[marked_site])


def marked_hamiltonian(n_ions, marked_strength):
    dims = [n_phonons] + ([2] * n_ions)
    Sz = quimb.spin_operator("Z", sparse=True)
    H = -quimb.ikron(2 * marked_strength * Sz, dims, inds=[1]) - quimb.ikron(
        2 * marked_strength * Sz, dims, inds=[n_ions]
    )
    return H


def marked_hamiltonian_spins_subspace(n_ions, marked_strength):
    dims = [2] * n_ions
    Sz = quimb.spin_operator("Z", sparse=True)
    H = -quimb.ikron(2 * marked_strength * Sz, dims, inds=[0]) - quimb.ikron(
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
    n_ions = 6
    n_phonons = 4
    time = 0.03
    segments = 30
    target_alpha = 0.8
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
    mu = np.ones(1) * data["mu"]  # 8 ions
    z_trap_frequency = data["z_trap_frequency"]
    optimum_time = data["time"]

    ion_trap_xy = iontrap.IonTrapXY(
        n_ions=n_ions,
        mu=mu,
        omega=np.array([2 * np.pi * 6e6, 2 * np.pi * 5e6, z_trap_frequency]),
    )
    gs, omega_single_mode, omega_eff, delta = ion_trap_xy.single_mode_ms()
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
        gi ** 2 * omega_eff / (8 * (omega_eff ** 2 - omega_single_mode ** 2))
        for gi in gs
    ]

    # Initial state
    up = quimb.qu([1, 0], qtype="ket")
    down = quimb.qu([0, 1], qtype="ket")

    init_spin_l = [up] + [down for _ in range(n_ions - 1)]
    final_spin_l = [down for _ in range(n_ions - 1)] + [up]
    groundstate_l = [down for _ in range(n_ions)]

    init_boson = quimb.qu([1, 0, 0, 0], qtype="ket")
    init_spin = quimb.kron(*init_spin_l)
    final_spin = quimb.kron(*final_spin_l)
    groundstate = quimb.kron(*groundstate_l)

    psi = quimb.kron(init_boson, init_spin)

    ion_trap_xy.single_mode_ms()
    ion_trap_xy.calculate_alpha()
    print(f"XY Js = {ion_trap_xy.Js}")
    print(f"alpha = {ion_trap_xy.alpha}")

    ham_xy = xy_hamiltonian(n_ions=n_ions, js=Js)
    # ham_xy = xyh_hamiltonian(n_ions=n_ions, js=Js, hs=hs)

    evolution_xy = quimb.Evolution(init_spin, ham_xy)

    def calc_t(t, _):
        return t

    def calc_fidelity(t, pt):
        rho = quimb.partial_trace(
            pt, [n_phonons] + ([2] * n_ions), [i for i in range(1, n_ions + 1, 1)]
        )
        evolution_xy.update_to(t + current_time)
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

    calc_dictionary = {
        "purity": calc_purity,
        "fidelity": calc_fidelity,
        "leakage": calc_leakage,
    }

    compute = {"time": calc_t, calc_parameter: calc_dictionary[calc_parameter]}

    # Evolution
    segment_time = time / segments
    segment_count = 0
    calc_values = []
    times = []
    current_time = 0
    while current_time < time:
        # Full dynamics walk evolution
        ham_full = SpinBosonHamMS(
            n_phonons=n_phonons,
            n_ions=n_ions,
            omega_eff=omega_eff,
            gs=gs,
            omega_single_mode=omega_single_mode,
            init_t=current_time,
        )
        evolution_full = quimb.Evolution(
            psi, ham_full, progbar=True, compute=compute, method="integrate"
        )
        evolution_full.update_to(segment_time)
        times.extend([t + current_time for t in evolution_full.results["time"]])
        calc_values.extend(evolution_full.results[calc_parameter])
        current_time = times[-1]
        print(f"Number of points = {len(evolution_full.results['time'])}")

        segment_count += 1
        psi = evolution_full.pt

        print(
            f">>>> Segment {segment_count} complete: Current time = {current_time} <<<<\n"
        )

    if update_data:
        data_handling.update_data(
            protocol="spin_boson_single_mode",
            chain="open",
            alpha=target_alpha,
            save_tag=f"{calc_parameter}_n={n_ions}_mu={mu[0]}_z_trap={z_trap_frequency}_up_to_t={time}",
            data_dict={"time": times, calc_parameter: calc_values},
            replace=True,
        )

    # Plot results
    fig, ax = plt.subplots(figsize=[8, 8])
    ax.plot(times, calc_values)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(calc_parameter.capitalize())

    ax.plot()
    plt.show()
