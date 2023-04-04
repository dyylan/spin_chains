from typing import final
import numpy as np
from matplotlib import pyplot as plt

from spin_chains.data_analysis import data_plots, data_handling
from spin_chains.quantum import iontrap


def mu_minimums(n_ions_list, z_trap_freqs):
    mu_mins = []
    for i, n in enumerate(n_ions_list):
        ion_trap = iontrap.IonTrapXY(
            n,
            omega=np.array([2 * np.pi * 6e6, 2 * np.pi * 5e6, z_trap_freqs[i]]),
        )
        mu_mins.append(ion_trap.calculate_minimum_mu()[0])
    print(mu_mins)


def coupling_strength_plots(n_ions, target_alpha=0.5):
    # z_factor = (2 * np.log(n_ions) * 6e6) / (3 * n_ions)
    # ion_trap = iontrap.IonTrapXY(
    #     n_ions,
    #     mu=np.ones(1) * 2 * np.pi * 6.01e6,
    #     omega=2 * np.pi * np.array([6e6, 5e6, 0.35 * z_factor]),
    # )
    # print(f"1. Minimum mu = {ion_trap.calculate_minimum_mu()}")

    # ion_trap.optimise_z_trap_frequency()

    # print(f"2. Minimum mu = {ion_trap.calculate_minimum_mu()}")

    # mu_bounds = [np.ones(1) * 2 * np.pi * 6.000005e6, np.ones(1) * 2 * np.pi * 6.07e6]
    # steps = 5
    # ion_trap.update_mu_for_target_alpha(target_alpha, mu_bounds, steps)
    # print(f"3. Minimum mu = {ion_trap.calculate_minimum_mu()}")
    # print(f"4. Mu = {ion_trap.mu}")
    data = data_handling.read_data_spin(
        protocol="experimental/always_on_fast_xy",
        chain="open",
        alpha=target_alpha,
        save_tag="optimum_gammas_end_n_mu_min",
        spins=n_ions,
    )
    mu = np.ones(1) * (data["mu"] * 1)  # 8 ions
    z_trap_frequency = data["z_trap_frequency"]

    ion_trap = iontrap.IonTrapXY(
        n_ions,
        mu=mu,
        omega=np.array([2 * np.pi * 6e6, 2 * np.pi * 5e6, data["z_trap_frequency"]]),
    )

    print(ion_trap.calculate_t_0())

    # ion_trap.plot_spin_interactions(
    #     savefig=False,
    #     include_ion_chain=True,
    #     all_interactions=True,
    #     use_z_position=True,
    #     plot_label="(a)",
    #     exp_fit=False,
    # )
    ion_trap.plot_spin_interactions(
        savefig=True,
        include_ion_chain=False,
        exp_fit=False,
        all_interactions=True,
        use_z_position=True,
        fit_end_and_mid=True,
        plot_label="(a)",
    )
    # ion_trap.plot_spin_interactions(
    #     savefig=False,
    #     include_ion_chain=True,
    #     all_interactions=False,
    #     use_z_position=True,
    #     plot_label="(a)",
    # )
    # ion_trap.plot_spin_interactions(
    #     savefig=False, include_ion_chain=True, plot_label="(a)"
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

    # coupling_strength_plots(n_ions=24, target_alpha=0.4)
    # data_plots.plot_exp_ao_fidelities(alpha=0.5)
    # data_plots.plot_exp_ao_fidelities_peak_connectivity(alpha=0.2)
    # data_plots.plot_exp_ao_specific_n_fidelity(alpha=0.5, spins=4)
    # data_plots.plot_exp_ion_spacings(spins=20)
    # data_plots.plot_exp_ao_multiple_fidelities_peak_connectivity(
    #     alphas=[0.2, 0.4, 0.5], subplot="alpha", minimum_delta_mu=1
    # )

    high_alphas = [0.3, 0.4, 0.5]
    mid_alphas = [0.2, 0.4, 0.6]
    low_alphas = [0.1, 0.2, 0.3]
    alphas_dict = {
        "high_alphas": high_alphas,
        "mid_alphas": mid_alphas,
        "low_alphas": low_alphas,
    }

    alphas = "mid_alphas"

    n_ions_list = list(range(6, 54, 2))
    z_trap_freqs = [
        10657532.409930686,
        8296603.004056687,
        6828692.485320178,
        5828901.475286691,
        5092946.4874604,
        4529379.067831489,
        4096256.107093891,
        3726902.7190767513,
        3425262.682000916,
        3178287.9302193266,
        2960453.9752684766,
        2781611.189932981,
        2607183.1166638355,
        2463394.2646559505,
        2332978.245799682,
        2226573.672497269,
        2117152.9013584764,
        2028067.4037789672,
        1945856.1543771382,
        1858910.6953071095,
        1788516.0156862822,
        1722912.6929953285,
        1661605.6657080948,
        1604167.1570190943,
    ]
    # mu_minimums(n_ions_list, z_trap_freqs)
    # data_plots.plot_exp_ao_multiple_fidelities_peak_connectivity_mu_min(
    #     alphas=alphas_dict[alphas],
    #     subplot=["mu", "time"],
    #     dashed_lines=["ideal"],
    #     save_tag=f"{alphas}",
    #     use_xy=True,
    # )
    # data_plots.plot_exp_ao_normalised_times(
    #     alphas=alphas_dict[alphas],
    #     save_tag=f"{alphas}",
    # )
    # data_plots.plot_exp_ao_multiple_fidelities_peak_connectivity_mid_end_comparison(
    #     alphas=alphas_dict["mid_alphas"],
    #     subplot=["mu", "alpha"],
    #     save_tag=f"{alphas}",
    #     use_xy=True,
    # )
    # data_plots.plot_exp_ao_normalised_times(
    #     alphas=[0.2, 0.4],
    #     save_tag=f"{alphas}",
    #     final_site="end_n",
    #     use_xy=True,
    # )
    # data_plots.plot_exp_ao_specific_n_fidelity(alpha=0.2, spins=12)
    # data_plots.plot_exp_ao_specific_n_fidelity_strobe(alpha=0.2, spins=52, n_strobe=5)
    # data_plots.plot_exp_ao_specific_n_fidelity_strobe_comparison(
    #     alpha=0.2, spins_list=[12], n_strobes=[4, 6, 14], use_xy=True, show_legend=False
    # )

    # Noise
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # data_plots.plot_exp_ao_specific_n_fidelity_with_noise(
    #     alpha=0.5, spins=52, t2_list=[0.01, 0.001, 0.0001, 0.00001], samples=100
    # )
    # data_plots.plot_exp_ao_multiple_t2s(
    #     alpha=0.1, t2_list=[0.01], plot_title=True, subplot="alpha"
    # )
    # data_plots.plot_exp_ao_multiple_t2s(
    #     alpha=0.4, t2_list=[0.01], plot_title=True, use_xy=False
    # )
    # data_plots.plot_exp_ao_multiple_alphas(
    #     alpha_list=[0.2, 0.4, 0.6],
    #     t2=0.01,
    #     samples=500,
    #     plot_title=False,
    #     use_xy=True,
    # )
    # data_plots.plot_exp_broken_spin(
    #     alpha=0.1, t2=0.01, ratios=[1, 30, 50, 100], subplot="alpha", plot_title=True
    # )
    # data_plots.plot_exp_broken_spin(
    #     alpha=0.2, t2=0.01, ratios=[1, 30, 50, 100], subplot=None, plot_title=True
    # )

    # Spin boson fidelities and purities
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # data_plots.plot_spin_boson_fidelity_comparisons(n_ions=8, n_phonons=4, alphas=[0.4])
    # data_plots.plot_spin_boson_fidelity_comparisons(n_ions=8, n_phonons=4, alphas=[0.4])
    # data_plots.plot_spin_boson_fom_for_alpha(
    #     n_ions=8,
    #     alphas=[0.2, 0.4, 0.8],
    #     fom="fidelity",
    #     save_fig=True,
    #     plot_title=True,
    #     plot_fits=False,
    #     plot_dephasing=True,
    # )
    # data_plots.plot_spin_boson_leakage_with_fit(
    #     n_ions=10,
    #     alpha=0.8,
    #     save_fig=True,
    #     plot_title=True,
    #     plot_label="(a)",
    # )
    # data_plots.plot_spin_boson_leakage_with_fit(
    #     n_ions=10,
    #     alpha=0.4,
    #     save_fig=True,
    #     plot_title=True,
    #     plot_label="(b)",
    # )
    # data_plots.plot_spin_boson_leakage_with_fit(
    #     n_ions=10,
    #     alpha=0.2,
    #     save_fig=True,
    #     plot_title=True,
    #     plot_label="(c)",
    #     omega_eff_r=1.000321,
    # )
    # data_plots.plot_spin_boson_leakage_two_phonon_modes_with_fit(
    #     n_ions=10,
    #     alpha=0.2,
    #     save_fig=True,
    #     plot_title=True,
    #     plot_label="(d)",
    #     hs_correction=True,
    #     # omega_eff_r=1.000321,
    #     omega_eff_r=1.000306,
    # )
    # data_plots.plot_spin_boson_leakage_two_phonon_modes_with_fit(
    #     n_ions=8,
    #     alpha=0.2,
    #     save_fig=True,
    #     plot_title=True,
    #     plot_label="",
    #     hs_correction=True,
    #     # omega_eff_r=1.000321,
    #     omega_eff_r=1.00031,
    # )
    # data_plots.plot_phonon_occupation_numbers(
    #     n_ions=10,
    #     n_phonons=4,
    #     alpha=0.2,
    #     save_fig=True,
    #     plot_title=True,
    #     plot_label="(a)",
    # )
    # data_plots.plot_phonon_number_expectation(
    #     n_ions=8,
    #     n_phonons=4,
    #     alphas=[0.2, 0.4, 0.8],
    #     save_fig=True,
    #     plot_title=False,
    #     plot_label=False,
    # )
    # data_plots.plot_full_stroboscopic(
    #     n_ions=8,
    #     alpha=0.8,
    #     strobes=12,
    #     save_fig=True,
    #     plot_title=True,
    #     plot_label="(c)",
    #     plot_xy=False,
    # )
    # data_plots.plot_spin_boson_fidelity_xy_with_r_comparisons(
    #     n_ions=8,
    #     n_phonons=4,
    #     alpha=0.2,
    #     rs=[0.975, 0.9567597060075298],
    #     plot_title=True,
    #     plot_label="(a)",
    #     save_fig=True,
    # )
    # data_plots.plot_spin_boson_fidelity_xy_with_r_comparisons(
    #     n_ions=10,
    #     n_phonons=4,
    #     alpha=0.2,
    #     rs=[0.92, 0.9395086408877635, 0.97],
    #     plot_title=True,
    #     plot_label="(a)",
    #     save_fig=True,
    # )
    # data_plots.plot_spin_boson_fidelity_xy_with_r_comparisons(
    #     n_ions=8,
    #     n_phonons=4,
    #     alpha=0.2,
    #     # rs=[0.9716238998156347, 0.9567597060075298, 0.975],
    #     # rs=[0.9567597060075298, 0.975],
    #     rs=[0.9596524966470248, 0.975],
    #     # rs=[],
    #     phonon_modes=2,
    #     plot_title=True,
    #     plot_label="(b)",
    #     save_fig=True,
    #     hs_correction=True,
    # )
    # data_plots.plot_spin_boson_fidelity_xy_with_r_comparisons(
    #     n_ions=10,
    #     n_phonons=4,
    #     alpha=0.2,
    #     # rs=[0.9716238998156347, 0.9567597060075298, 0.975],
    #     # rs=[0.9567597060075298, 0.975],
    #     rs=[0.9232806569998392, 0.9410856698474532, 0.9754715299109832],
    #     # rs=[],
    #     phonon_modes=2,
    #     plot_title=True,
    #     plot_label="(b)",
    #     save_fig=True,
    #     hs_correction=True,
    # )
    # data_plots.plot_spin_boson_fidelity_xy_with_r_comparisons(
    #     n_ions=8,
    #     n_phonons=4,
    #     alpha=0.2,
    #     # rs=[0.9716238998156347, 0.9567597060075298, 0.975],
    #     # rs=[0.9567597060075298, 0.975],
    #     # rs=[0.9881801871694981, 0.975],
    #     rs=[0.9567597060075298, 0.9668466447067687],
    #     # rs=[],
    #     phonon_modes=1,
    #     plot_title=True,
    #     plot_label="(c)",
    #     save_fig=True,
    #     subspace=4,
    # )
    # data_plots.plot_spin_boson_fidelity_xy_with_r_comparisons(
    #     n_ions=10,
    #     n_phonons=4,
    #     alpha=0.2,
    #     # rs=[0.9716238998156347, 0.9567597060075298, 0.975],
    #     # rs=[0.9567597060075298, 0.975],
    #     # rs=[0.9881801871694981, 0.975],
    #     rs=[0.92, 0.95007, 0.975],
    #     # rs=[],
    #     phonon_modes=1,
    #     plot_title=True,
    #     plot_label="(c)",
    #     save_fig=True,
    #     subspace=5,
    # )
    # data_plots.plot_spin_boson_fidelity_xy_with_r_comparisons(
    #     n_ions=8,
    #     n_phonons=4,
    #     alpha=0.2,
    #     # rs=[0.9716238998156347, 0.9567597060075298, 0.975],
    #     # rs=[0.975, 0.9567597060075298],
    #     rs=[0.975, 0.960],
    #     # rs=[],
    #     phonon_modes=2,
    #     plot_title=True,
    #     plot_label="(b)",
    #     save_fig=True,
    #     hs_correction=False,
    # )
    # data_plots.plot_spin_boson_compare_phonon_modes(
    #     n_ions=8,
    #     n_phonons=4,
    #     alpha=0.2,
    #     phonon_modes=[1, 2],
    #     correct_hs=[2],
    #     time=100,
    #     plot_title=False,
    #     save_fig=False,
    # )
    data_plots.plot_phonon_number_expectation_higher_subspace(
        n_ions=8,
        n_phonons=4,
        alpha=0.2,
        rs=[1, 1.000338],
        subspace=8,
        save_fig=True,
        plot_title=True,
        plot_label="(a)",
        points=5800,
    )
    data_plots.plot_phonon_number_expectation_higher_subspace(
        n_ions=8,
        n_phonons=4,
        alpha=0.2,
        rs=[1, 1.0002584],
        subspace=4,
        save_fig=True,
        plot_title=True,
        plot_label="(b)",
        points=5800,
    )
    data_plots.plot_phonon_number_expectation_higher_subspace(
        n_ions=8,
        n_phonons=4,
        alpha=0.2,
        rs=[1, 1.000096],
        subspace=2,
        save_fig=True,
        plot_title=True,
        plot_label="(c)",
        points=5800,
    )
    # data_plots.plot_phonon_number_expectation_higher_subspace(
    #     n_ions=10,
    #     n_phonons=4,
    #     alpha=0.2,
    #     rs=[1, 1.000321],
    #     subspace=10,
    #     save_fig=True,
    #     plot_title=True,
    #     plot_label="(a)",
    #     points=5800,
    # )
    # data_plots.plot_phonon_number_expectation_higher_subspace(
    #     n_ions=10,
    #     n_phonons=4,
    #     alpha=0.2,
    #     rs=[1, 1.000262],
    #     subspace=5,
    #     save_fig=True,
    #     plot_title=True,
    #     plot_label="(b)",
    #     points=5800,
    # )
    # data_plots.plot_phonon_number_expectation_higher_subspace(
    #     n_ions=10,
    #     n_phonons=4,
    #     alpha=0.2,
    #     rs=[1, 1.0000788],
    #     subspace=2,
    #     save_fig=True,
    #     plot_title=True,
    #     plot_label="(c)",
    #     points=5800,
    # )
