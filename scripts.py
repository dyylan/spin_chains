import numpy as np
from matplotlib import pyplot as plt

from spin_chains.data_analysis import data_plots
from spin_chains.quantum import iontrap


# def coupling_strength_plots(n_ions, target_alpha=0.5):
#     Omega = np.ones((n_ions, 1)) * 2 * np.pi * 1e6  # rabi frequency
#     mu = np.ones(1) * 2 * np.pi * 6.01e6  # raman beatnote detuning
#     mu = np.ones(1) * 37873857.62495645
#     # mu = np.ones(1) * 2 * np.pi * 6.132e6
#     delta_k = 2 * 2 * np.pi / 355e-9 * np.array([1, 0, 0])  # wave vector difference

#     print(couplings_calculations.omega)
#     z_trap = 10 * (2 * np.log(n_ions) * 6e6) / (3 * n_ions)
#     omega = 2 * np.pi * np.array([6e6, 5e6, z_trap])
#     print(omega)
#     print(couplings_calculations.omega)
#     couplings_calculations.set_trap_frequencies(omega)

#     x_ep = couplings_calculations.calculate_equilibrium_positions(n_ions)
#     couplings_calculations.plot_ion_chain_positions(x_ep)
#     hessian = couplings_calculations.calculate_equilibrium_hessian(x_ep, n_ions)
#     w, b = couplings_calculations.calculate_normal_mode_eigenpairs(hessian, n_ions)
#     Js = couplings_calculations.spin_interaction_graph(n_ions, Omega, mu, delta_k, w, b)

#     # plt.imshow(Js)
#     # plt.show()
#     start_mu = np.ones(1) * 2 * np.pi * 6.005e6
#     end_mu = np.ones(1) * 2 * np.pi * 6.06e6
#     mu, alpha, Js = couplings_calculations.mu_for_specific_alpha_BO(
#         target_alpha=target_alpha,
#         mu_bounds=[start_mu, end_mu],
#         steps=20,
#         n_ions=n_ions,
#         Omega=Omega,
#         delta_k=delta_k,
#         verbosity=True,
#     )
#     print(mu)
#     couplings_calculations.plot_end_spin_interactions(Js[0], n_ions, mu[0])
#     couplings_calculations.plot_spin_interactions(
#         Js[6], n_ions, mu[0], x=6, exp_fit=True
#     )


def coupling_strength_plots(n_ions, target_alpha=0.5):
    z_factor = (2 * np.log(n_ions) * 6e6) / (3 * n_ions)
    print()
    ion_trap = iontrap.IonTrap(
        n_ions,
        mu=np.ones(1) * 2 * np.pi * 6.01e6,
        omega=2 * np.pi * np.array([6e6, 5e6, 0.35 * z_factor]),
    )
    ion_trap.optimise_z_trap_frequency()

    mu_bounds = [np.ones(1) * 2 * np.pi * 6.000005e6, np.ones(1) * 2 * np.pi * 6.07e6]
    steps = 50
    ion_trap.update_mu_for_target_alpha(target_alpha, mu_bounds, steps)
    ion_trap.plot_spin_interactions(
        savefig=True, include_ion_chain=True, plot_label="(a)"
    )
    # ion_trap.plot_Js()

    # ion_trap2 = iontrap.IonTrap(
    #     n_ions,
    #     mu=np.ones(1) * 2 * np.pi * 6.01e6,
    #     omega=2 * np.pi * np.array([6e6, 5e6, 0.25e6]),
    # )
    # ion_trap2.optimise_z_trap_frequency()
    # # ion_trap2.plot_ion_chain_positions(savefig=True)
    # # ion_trap2.update_mu_for_target_alpha(target_alpha, mu_bounds, steps)
    # ion_trap2.plot_spin_interactions(
    #     savefig=True, include_ion_chain=True, plot_label="(b)"
    # )

    # ion_trap3 = iontrap.IonTrap(
    #     n_ions,
    #     mu=np.ones(1) * 2 * np.pi * 6.01e6,
    #     omega=2 * np.pi * np.array([6e6, 5e6, 2.33 * z_factor]),
    # )
    # # ion_trap3.plot_ion_chain_positions(savefig=True)
    # # ion_trap3.update_mu_for_target_alpha(target_alpha, mu_bounds, steps)
    # ion_trap3.plot_spin_interactions(
    #     savefig=True, include_ion_chain=True, plot_label="(c)"
    # )


if __name__ == "__main__":
    # data_plots.plot_ao_low_n_fideilties(alpha=1)
    # data_plots.plot_ao_low_n_fidelities_simple(alpha=0.5)
    # data_plots.plot_rs_low_n_fidelities_simple(alpha=0.5)

    # data_plots.plot_ao_low_n_times(alpha=1)
    # data_plots.plot_ao_open_low_n_fidelities_various_end_n(alpha=1)

    # data_plots.plot_ao_open_low_n_fidelities_noise(alpha=0.5)
    # data_plots.plot_rs_open_low_n_fidelities_noise(alpha=0.5)
    # data_plots.plot_rs_open_low_n_fidelities_noise(alpha=0.5)

    # coupling_strength_plots(n_ions=32, target_alpha=0.4)
    # data_plots.plot_exp_ao_fidelities(alpha=0.5)
    # data_plots.plot_exp_ao_fidelities_peak_connectivity(alpha=0.2)
    # data_plots.plot_exp_ao_specific_n_fidelity(alpha=0.2, spins=44)
    # data_plots.plot_exp_ion_spacings(spins=20)
    data_plots.plot_exp_ao_multiple_fidelities_peak_connectivity(
        alphas=[0.2, 0.4, 0.5], subplot="alpha"
    )
