from spin_chains.data_analysis import data_plots


if __name__ == "__main__":
    # data_plots.plot_ao_low_n_fideilties(alpha=1)
    data_plots.plot_ao_low_n_fidelities_simple(alpha=1)
    # data_plots.plot_ao_low_n_times(alpha=1)
    data_plots.plot_ao_open_low_n_fidelities_various_end_n(alpha=1)

    data_plots.plot_ao_open_low_n_fidelities_noise(alpha=1)
