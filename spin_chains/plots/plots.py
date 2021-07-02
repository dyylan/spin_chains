import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from ..functions.fits import power_fit, sqrt_power_fit

plt.rc("text", usetex=True)
font = {"family": "serif", "size": 16, "serif": ["computer modern roman"]}
plt.rc("font", **font)

fontsize_title = 24
fontsize_axis = 22
fontsize_legend = 18
fontsize_ticks = 18
figure_size = [10, 8]


def plot_fidelity(
    spins,
    naive_fidelities,
    fidelities,
):
    ys = [
        fidelities,
        # asymptotic_fidelities,
        naive_fidelities,
    ]
    fig, ax = plt.subplots(figsize=[8, 8])
    linestyles = ["solid", "dashed", "dashed", "dashed"]
    for i, y in enumerate(ys):
        ax.plot(spins, y, linestyle=linestyles[i])
    ax.legend(
        [
            "Fidelity",
            # "Analytical fidelity expression",
            # # "new large $n$ analytical fidelity expression",
            "Spatial search fidelity squared",
        ],
        loc=4,
    )
    ax.set(xlabel="$n$")
    ax.grid()
    plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    plt.show()


def plot_time(spins, analytical_times, times):
    ys = [times, analytical_times]
    fig, ax = plt.subplots()
    linestyles = ["solid", "solid", "dashed"]
    for i, y in enumerate(ys):
        ax.plot(spins, y, linestyle=linestyles[i])
    ax.legend(["Time", "Always on time", "Analytical time"], loc=4)
    ax.set(xlabel="$n$")
    ax.grid()
    ax.set(xscale="log")
    ax.set(yscale="log")
    plt.show()


def plot_always_on_fidelity(spins, naive_fidelities, fidelities_1, fidelities_2):
    ys = [
        fidelities_1["fidelity"],
        fidelities_2["fidelity"],
        naive_fidelities,
    ]
    fig, ax = plt.subplots()
    linestyles = ["solid", "solid", "dashed"]
    for i, y in enumerate(ys):
        ax.plot(spins, y, linestyle=linestyles[i])
    ax.legend(
        [
            f"Fidelity for {fidelities_1['name']}",
            f"Fidelity for {fidelities_2['name']}",
            "Spatial search fidelity squared",
        ],
        loc=0,
    )
    ax.set(xlabel="$n$")
    ax.grid()
    plt.show()


def plot_always_on_time(alpha, spins, times_fast, times_slow, times_3_slow):

    fast_popt, fast_pcov = curve_fit(
        sqrt_power_fit, spins, times_fast, bounds=(0, [10.0, 0.1])
    )
    print(f"Fit for fast time: y = {fast_popt[0]} * x^0.5 + {fast_popt[1]}")

    slow_popt, slow_pcov = curve_fit(
        power_fit, spins, times_slow, bounds=(0, [10.0, 2.0, 0.1])
    )
    print(f"Fit for slow time: y = {slow_popt[0]} * x^{slow_popt[1]} + {slow_popt[2]}")

    slow_3_popt, slow_3_pcov = curve_fit(
        power_fit, spins, times_3_slow, bounds=(0, [10.0, 2.0, 0.1])
    )
    print(
        f"Fit for slow time: y = {slow_3_popt[0]} * x^{slow_3_popt[1]} + {slow_3_popt[2]}"
    )

    times_1_fit = power_fit(spins, fast_popt[0], 0.5, 0)
    times_2_fit = power_fit(spins, slow_popt[0], slow_popt[1], 0)
    times_3_fit = power_fit(spins, slow_3_popt[0], slow_3_popt[1], 0)

    ys = [times_fast, times_1_fit, times_slow, times_2_fit, times_3_slow, times_3_fit]

    fig, ax = plt.subplots(figsize=figure_size)
    linestyles = ["solid", "dotted", "solid", "dotted", "solid", "dotted"]
    colours = [
        "darkslategrey",
        "b",
        "g",
        "darkorchid",
        "red",
        "navy",
        "darkorange",
        "teal",
    ]
    colours = ["b", "b", "g", "g", "darkorchid", "darkorchid"]
    for i, y in enumerate(ys):
        ax.plot(spins, y, linestyle=linestyles[i], color=colours[i])
    ax.legend(
        [
            f"Time with optimum $\\gamma$",
            f"${fast_popt[0]:.2f} n^{{0.5}}$",
            f"Time with $\\gamma / 2$",
            f"${slow_popt[0]:.2f} n^{{ {slow_popt[1]:.2f} }}$",
            f"Time with $\\gamma / 3$",
            f"${slow_3_popt[0]:.2f} n^{{ {slow_3_popt[1]:.2f} }}$",
        ],
        loc=0,
        fontsize=fontsize_legend,
    )
    ax.set_xlabel(xlabel="$n$", fontsize=fontsize_axis)
    ax.set_ylabel(ylabel="Time", fontsize=fontsize_axis)

    ax.grid()
    ax.set(xscale="log")
    ax.set(yscale="log")
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.savefig(
        f"plots/alpha={alpha}/plot_ao_time_scaling_comparisons.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_always_on_time_comparison(
    alpha1, alpha2, spins, times_fast1, times_slow1, times_fast2, times_slow2
):

    fast_popt1, fast_pcov1 = curve_fit(
        sqrt_power_fit, spins, times_fast1, bounds=(0, [10.0, 0.1])
    )
    print(
        f"Fit for fast time with alpha = {alpha1}: y = {fast_popt1[0]} * x^0.5 + {fast_popt1[1]}"
    )

    fast_popt2, fast_pcov2 = curve_fit(
        sqrt_power_fit, spins, times_fast2, bounds=(0, [10.0, 0.1])
    )
    print(
        f"Fit for fast time with alpha = {alpha2}: y = {fast_popt2[0]} * x^0.5 + {fast_popt2[1]}"
    )

    slow_popt1, slow_pcov1 = curve_fit(
        power_fit, spins, times_slow1, bounds=(0, [10.0, 2, 0.1])
    )
    print(
        f"Fit for slow time with alpha = {alpha1}: y = {slow_popt1[0]} * x^{slow_popt1[1]}+ {slow_popt1[2]}"
    )

    slow_popt2, slow_pcov2 = curve_fit(
        power_fit, spins, times_slow2, bounds=(0, [10.0, 2, 0.1])
    )
    print(
        f"Fit for slow time with alpha = {alpha2}: y = {slow_popt2[0]} * x^{slow_popt2[1]} + {slow_popt2[2]}"
    )

    times_fast_1_fit = power_fit(spins, fast_popt1[0], 0.5, 0)
    times_fast_2_fit = power_fit(spins, fast_popt2[0], 0.5, 0)

    times_slow_1_fit = power_fit(spins, slow_popt1[0], slow_popt1[1], 0)
    times_slow_2_fit = power_fit(spins, slow_popt2[0], slow_popt2[1], 0)

    ys = [
        times_fast1,
        times_fast_1_fit,
        times_slow1,
        times_slow_1_fit,
        times_fast2,
        times_fast_2_fit,
        times_slow2,
        times_slow_2_fit,
    ]

    fig, ax = plt.subplots(figsize=figure_size)
    linestyles = [
        "solid",
        "dotted",
        "solid",
        "dotted",
        "solid",
        "dotted",
        "solid",
        "dotted",
    ]
    markers = ["x", "", "x", "", "D", "", "D", ""]
    colours = [
        "b",
        "b",
        "g",
        "g",
        "b",
        "b",
        "g",
        "g",
    ]
    for i, y in enumerate(ys):
        ax.plot(spins, y, linestyle=linestyles[i], color=colours[i], marker=markers[i])

    marker_line_1 = mlines.Line2D(
        [], [], color="black", marker=markers[0], label=f"$\\alpha = ${alpha1}"
    )
    marker_line_2 = mlines.Line2D(
        [], [], color="black", marker=markers[4], label=f"$\\alpha = ${alpha2}"
    )

    legend0 = plt.legend(
        handles=[marker_line_1, marker_line_2],
        bbox_to_anchor=(1, 0.76),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend0)

    dotted_line_1 = mlines.Line2D(
        [],
        [],
        color=colours[0],
        linestyle=linestyles[1],
        marker=markers[0],
        label=f"fit: ${fast_popt1[0]:.2f}n^{{0.5}}$",
    )
    dotted_line_2 = mlines.Line2D(
        [],
        [],
        color=colours[0],
        linestyle=linestyles[1],
        marker=markers[4],
        label=f"fit: ${fast_popt2[0]:.2f}n^{{0.5}}$",
    )
    dotted_line_3 = mlines.Line2D(
        [],
        [],
        color=colours[2],
        linestyle=linestyles[1],
        marker=markers[0],
        label=f"fit: ${slow_popt1[0]:.2f}n^{{ {slow_popt1[1]:.2f} }}$",
    )
    dotted_line_4 = mlines.Line2D(
        [],
        [],
        color=colours[2],
        linestyle=linestyles[1],
        marker=markers[4],
        label=f"fit: ${slow_popt2[0]:.2f}n^{{ {slow_popt2[1]:.2f} }}$",
    )

    # legend1 = plt.legend(handles=[solid_line, dotted_line, x_marker], loc=3, fontsize=fontsize_legend)
    legend1 = plt.legend(
        handles=[dotted_line_1, dotted_line_2, dotted_line_3, dotted_line_4],
        bbox_to_anchor=(1, 0.50),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend1)

    patches = []
    patch1 = mpatches.Patch(color=colours[0], label=f"Optimum $\\gamma$")
    patch2 = mpatches.Patch(color=colours[2], label=f"$\\gamma / 2$")
    patches.append(patch1)
    patches.append(patch2)
    legend2 = plt.legend(
        handles=patches, bbox_to_anchor=(1, 0.63), fontsize=fontsize_legend
    )
    ax.add_artist(legend2)

    ax.set_xlabel(xlabel="$n$", fontsize=fontsize_axis)
    ax.set_ylabel(ylabel="Time", fontsize=fontsize_axis)

    ax.grid()
    ax.set(xscale="log")
    ax.set(yscale="log")
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.savefig(
        f"plots/plot_ao_time_scaling_comparisons_multi_alpha.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_time_comparisons(
    spins, ao_times, rs_times, fast_spins, fast_times, plot_sequential=True
):

    ys = [ao_times, rs_times]
    fig, ax = plt.subplots()
    linestyles = ["solid", "solid", "dotted", "dashed"]
    for i, y in enumerate(ys):
        ax.plot(spins, y, linestyle=linestyles[i])
    ax.plot(fast_spins, fast_times, linestyle=linestyles[-1])

    if plot_sequential:
        J = 0.25
        sequential_swaps_time = [np.pi * n / (2 * J) for n in spins]
        ax.plot(spins, sequential_swaps_time, linestyle="dotted")
        ax.legend(
            [
                f"Time for AO protocol",
                f"Time for RS protocol",
                f"Time for fast algorithm",
                f"Time for sequential SWAP gates",
            ],
            loc=0,
        )
    else:
        ax.legend(
            [
                f"Time for AO protocol",
                f"Time for RS protocol",
                f"Time for fast algorithm",
            ],
            loc=0,
        )

    ax.set(xlabel="$n$")
    ax.grid()
    # plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    plt.show()


def plot_fidelity_comparisons(
    spins, ao_fidelities, rs_fidelities, fast_spins, fast_fidelities
):
    # sequential_swaps_time = [np.pi * n / 2 for n in spins]

    ys = [ao_fidelities, rs_fidelities]
    fig, ax = plt.subplots()
    linestyles = ["solid", "solid", "dotted", "dashed"]
    for i, y in enumerate(ys):
        ax.plot(spins, y, linestyle=linestyles[i])
    ax.plot(fast_spins, fast_fidelities, linestyle=linestyles[-1])
    ax.legend(
        [
            f"Fidelity for AO protocol",
            f"Fidelity for RS protocol",
            # f"Fidelity for sequential SWAP gates",
            f"Fidelity for fast algorithm",
        ],
        loc="lower left",
    )
    ax.set(xlabel="$n$")
    ax.grid()
    # plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    plt.show()


def plot_deltas(alpha, n):
    gamma = s_parameter(1, alpha, n)
    y = []
    x = [i + 1 for i in range(n)]
    for k in range(1, n):
        y.append(delta_k(gamma, k, alpha, n))
    y.append(delta_n(alpha, n))
    fig, ax = plt.subplots()
    ax.plot(x, y, linestyle="solid")
    ax.legend("$ \delta_k $", loc=4)
    ax.set(xlabel="$k$")
    ax.grid()
    plt.show()


def plot_fidelity_time_error_comparisons(spins, fidelities, fidelities_te, time_error):
    ys = [fidelities, fidelities_te]
    fig, ax = plt.subplots()
    linestyles = ["solid", "solid", "dotted", "dashed"]
    for i, y in enumerate(ys):
        ax.plot(spins, y, linestyle=linestyles[i])
    ax.legend(
        [
            f"Fidelity for AO protocol",
            f"Fidelity with time error of ${time_error}$",
        ],
        loc="lower left",
    )
    ax.set(xlabel="$n$")
    ax.grid()
    # plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    plt.show()


def plot_fidelity_time_specific_n(times, fidelities_list, gammas, analytical_fidelity):
    fig, ax = plt.subplots()

    colours = [
        "darkslategrey",
        "b",
        "g",
        "darkorchid",
        "red",
        "navy",
        "darkorange",
        "teal",
    ]

    ax.plot(times, fidelities_list[1], color=colours[0], linestyle="dashed")
    ax.plot(times, fidelities_list[0], color=colours[1])
    ax.plot(times, fidelities_list[2], color=colours[7], linestyle="dashed")
    ax.plot(times, fidelities_list[3], color=colours[2])

    ax.plot(
        times,
        [analytical_fidelity for _ in range(len(times))],
        color="red",
        linestyle="dotted",
    )

    ax.plot()
    ax.legend(
        [
            f"$\gamma$ = {gammas[1]:.4f}",
            f"$\gamma$ = {gammas[0]:.4f} (optimum)",
            f"$\gamma$ = {gammas[2]:.4f}",
            f"Analytical $\gamma$ = {gammas[3]:.4f}",
            f"Analytical fidelity",
        ],
        loc="center right",
    )

    ax.set(xlabel="Time")
    ax.set(ylabel="Fidelity")
    ax.grid()
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.show()


def plot_fidelity_time_fast_slow_specific_n(times_list, fidelities_list, gammas):
    fig, ax = plt.subplots(figsize=[8, 8])

    colours = [
        "darkslategrey",
        "b",
        "g",
        "darkorchid",
        "red",
        "navy",
        "darkorange",
        "teal",
    ]

    ax.plot(times_list[0], fidelities_list[0], color=colours[4])
    ax.plot(times_list[1], fidelities_list[1], color=colours[2])
    ax.plot(times_list[2], fidelities_list[2], color=colours[1])
    ax.plot(times_list[3], fidelities_list[3], color=colours[3])

    ax.plot()
    ax.legend(
        [
            f"$\gamma_0$ = {gammas[0]:.4f}",
            f"$\gamma = \gamma_0 / 2$",
            f"$\gamma = \gamma_0 / 3$",
            f"$\gamma = \gamma_0 / 4$",
        ],
        loc="upper right",
        fontsize=fontsize_legend,
    )

    ax.set_xlabel("Time", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    ax.grid()
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=fontsize_ticks
    )
    plt.savefig(
        f"plots/plot_ao_slow_open_various_gamma_alpha=1_n=60_end_n.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_fidelity_time_fast_slow_specific_n_2(
    times_list, fidelities_list, alphas, gammas
):
    fig, ax = plt.subplots(figsize=[8, 8])

    colours = [
        "darkslategrey",
        "b",
        "g",
        "darkorchid",
        "red",
        "navy",
        "darkorange",
        "teal",
    ]

    ax.plot(times_list[0], fidelities_list[0], color=colours[4])
    ax.plot(times_list[1], fidelities_list[1], color=colours[2])
    ax.plot(times_list[2], fidelities_list[2], color=colours[1])
    ax.plot(times_list[3], fidelities_list[3], color=colours[3])

    ax.plot()
    ax.legend(
        [
            f"$\\alpha$ = {alphas[0]}",
            f"$\\alpha$ = {alphas[1]}",
            f"$\\alpha$ = {alphas[2]}",
            f"$\\alpha$ = {alphas[3]}",
        ],
        loc="upper right",
        fontsize=fontsize_legend,
    )

    ax.set_xlabel("Time", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    ax.grid()
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    plt.savefig(
        f"plots/plot_ao_slow_open_various_alpha_n=60_end_n.pdf",
        bbox_inches="tight",
    )
    plt.show()
