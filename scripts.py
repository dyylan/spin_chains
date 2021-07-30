from typing import final
import numpy as np
from matplotlib import pyplot as plt

from spin_chains.data_analysis import data_plots
from spin_chains.quantum import iontrap


def coupling_strength_plots(n_ions, target_alpha=0.5):
    z_factor = (2 * np.log(n_ions) * 6e6) / (3 * n_ions)
    ion_trap = iontrap.IonTrap(
        n_ions,
        mu=np.ones(1) * 2 * np.pi * 6.01e6,
        omega=2 * np.pi * np.array([6e6, 5e6, 0.35 * z_factor]),
    )
    print(f"1. Minimum mu = {ion_trap.calculate_minimum_mu()}")

    ion_trap.optimise_z_trap_frequency()

    print(f"2. Minimum mu = {ion_trap.calculate_minimum_mu()}")

    mu_bounds = [np.ones(1) * 2 * np.pi * 6.000005e6, np.ones(1) * 2 * np.pi * 6.07e6]
    steps = 5
    ion_trap.update_mu_for_target_alpha(target_alpha, mu_bounds, steps)
    print(f"3. Minimum mu = {ion_trap.calculate_minimum_mu()}")
    print(f"4. Mu = {ion_trap.mu}")

    # ion_trap.plot_spin_interactions(
    #     savefig=True, include_ion_chain=True, plot_label="(a)"
    # )
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

    # coupling_strength_plots(n_ions=10, target_alpha=0.2)
    # data_plots.plot_exp_ao_fidelities(alpha=0.5)
    # data_plots.plot_exp_ao_fidelities_peak_connectivity(alpha=0.2)
    # data_plots.plot_exp_ao_specific_n_fidelity(alpha=0.5, spins=4)
    # data_plots.plot_exp_ion_spacings(spins=20)
    # data_plots.plot_exp_ao_multiple_fidelities_peak_connectivity(
    #     alphas=[0.2, 0.4, 0.5], subplot="alpha", minimum_delta_mu=1
    # )

    high_alphas = [0.3, 0.4, 0.5]
    mid_alphas = [0.1, 0.2, 0.4]
    low_alphas = [0.1, 0.2, 0.3]
    alphas_dict = {
        "high_alphas": high_alphas,
        "mid_alphas": mid_alphas,
        "low_alphas": low_alphas,
    }

    alphas = "mid_alphas"
    # data_plots.plot_exp_ao_multiple_fidelities_peak_connectivity_mu_min(
    #     alphas=alphas_dict[alphas],
    #     subplot=["mu", "alpha"],
    #     dashed_lines=["ideal"],
    #     save_tag=f"{alphas}",
    # )
    # data_plots.plot_exp_ao_normalised_times(
    #     alphas=alphas_dict[alphas],
    #     save_tag=f"{alphas}",
    # )

    # data_plots.plot_exp_ao_multiple_fidelities_peak_connectivity_mid_end_comparison(
    #     alphas=alphas_dict["mid_alphas"],
    #     subplot=["mu", "alpha"],
    #     save_tag=f"{alphas}",
    # )
    # data_plots.plot_exp_ao_normalised_times(
    #     alphas=alphas_dict["mid_alphas"], save_tag=f"{alphas}", final_site="end_n"
    # )

    # Noise
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # data_plots.plot_exp_ao_specific_n_fidelity_with_noise(
    #     alpha=0.5, spins=52, t2_list=[0.01, 0.001, 0.0001, 0.00001], samples=100
    # )
    # data_plots.plot_exp_ao_multiple_t2s(
    #     alpha=0.1, t2_list=[0.01], plot_title=True, subplot="alpha"
    # )
    data_plots.plot_exp_ao_multiple_t2s(alpha=0.4, t2_list=[0.01], plot_title=True)
    # data_plots.plot_exp_broken_spin(
    #     alpha=0.1, t2=0.01, ratios=[1, 30, 50, 100], subplot="alpha", plot_title=True
    # )
    # data_plots.plot_exp_broken_spin(
    #     alpha=0.2, t2=0.01, ratios=[1, 30, 50, 100], subplot=None, plot_title=True
    # )
