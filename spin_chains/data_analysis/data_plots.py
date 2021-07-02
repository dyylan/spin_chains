import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

from .data_handling import read_data, read_data_spin, update_data
from ..functions.protocols import quantum_communication_exp
from ..quantum import iontrap

plt.rc("text", usetex=True)
font = {"family": "serif", "size": 16, "serif": ["computer modern roman"]}
plt.rc("font", **font)

fontsize_title = 24
fontsize_axis = 22
fontsize_legend = 18
fontsize_ticks = 18
figure_size = [10, 8]


# Always-on protocol plots
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_ao_low_n_fidelities(alpha=1, plot_title=False):
    open_mid_n_data = read_data("always_on_fast", "open", alpha, "optimum_gammas_mid_n")
    open_end_n_data = read_data("always_on_fast", "open", alpha, "optimum_gammas_end_n")

    closed_mid_n_data = read_data(
        "always_on_fast", "closed", alpha, "optimum_gammas_mid_n"
    )
    closed_end_n_data = read_data(
        "always_on_fast", "closed", alpha, "optimum_gammas_end_n"
    )

    fig, ax = plt.subplots(figsize=figure_size)
    if plot_title:
        fig.suptitle(
            f"Always-On fidelity for $\\alpha={alpha}$",
            fontsize=fontsize_title,
        )

    linestyles = ["solid", "dashed", "dotted"]
    colours = ["red", "blue"]
    ax.plot(
        open_mid_n_data["spins"],
        open_mid_n_data["fidelity"],
        linestyle=linestyles[1],
        color=colours[0],
    )
    ax.plot(
        open_end_n_data["spins"],
        open_end_n_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[0],
    )
    ax.plot(
        closed_mid_n_data["spins"],
        closed_mid_n_data["fidelity"],
        linestyle=linestyles[1],
        color=colours[1],
    )
    ax.plot(
        closed_end_n_data["spins"],
        closed_end_n_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[1],
    )
    ax.plot(
        open_end_n_data["spins"],
        open_end_n_data["naive_fidelity"],
        linestyle=linestyles[2],
        color=colours[0],
    )
    ax.plot(
        closed_mid_n_data["spins"],
        closed_mid_n_data["naive_fidelity"],
        linestyle=linestyles[2],
        color=colours[1],
    )

    solid_line = mlines.Line2D([], [], color="black", label="$w = 1, f = n$")
    dashed_line = mlines.Line2D(
        [],
        [],
        color="black",
        linestyle=linestyles[1],
        label="$w = 1, f = n/2$",
    )
    dotted_line = mlines.Line2D(
        [],
        [],
        color="black",
        linestyle=linestyles[2],
        label="Analytical spatial search fidelity squared",
    )

    # legend1 = plt.legend(handles=[solid_line, dotted_line, x_marker], loc=3, fontsize=fontsize_legend)
    legend1 = plt.legend(
        handles=[solid_line, dashed_line, dotted_line],
        bbox_to_anchor=(1, 0.5),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend1)

    patches = []
    patch1 = mpatches.Patch(color=colours[0], label=f"Open chain")
    patch2 = mpatches.Patch(color=colours[1], label=f"Closed chain")
    patches.append(patch1)
    patches.append(patch2)
    legend2 = plt.legend(
        handles=patches, bbox_to_anchor=(1, 0.65), fontsize=fontsize_legend
    )
    ax.add_artist(legend2)

    ax.set_xlabel("$n$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    ax.grid()
    plt.yticks(
        [
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.0,
        ],
        fontsize=fontsize_ticks,
    )
    plt.savefig(
        f"plots/alpha={alpha}/plot_ao_low_n_fideilties.pdf", bbox_inches="tight"
    )
    plt.show()


def plot_ao_low_n_fidelities_simple(alpha=1, plot_title=False):
    open_end_n_data = read_data("always_on_fast", "open", alpha, "optimum_gammas_end_n")

    closed_mid_n_data = read_data(
        "always_on_fast", "closed", alpha, "optimum_gammas_mid_n"
    )

    fig, ax = plt.subplots(figsize=figure_size)
    if plot_title:
        fig.suptitle(
            f"Always-on fidelity for $\\alpha={alpha}$",
            fontsize=fontsize_title,
        )

    linestyles = ["solid", "dashed", "dotted"]
    colours = ["red", "blue"]
    ax.plot(
        open_end_n_data["spins"],
        open_end_n_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[0],
    )
    ax.plot(
        open_end_n_data["spins"],
        open_end_n_data["naive_fidelity"],
        linestyle=linestyles[2],
        color=colours[0],
    )
    ax.plot(
        closed_mid_n_data["spins"],
        closed_mid_n_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[1],
    )
    ax.plot(
        closed_mid_n_data["spins"],
        closed_mid_n_data["naive_fidelity"],
        linestyle=linestyles[2],
        color=colours[1],
    )

    ax.legend(
        [
            "Open chain: $w=1, f=n$",
            "Open chain: analytical spatial search fidelity squared",
            "Closed chain: $w=1, f=n/2$",
            "Closed chain: analytical spatial search fidelity squared",
        ],
        bbox_to_anchor=(1, 0.22),
        fontsize=fontsize_legend - 4,
    )

    ax.set_xlabel("$n$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    ax.grid()
    plt.yticks(
        # [
        #     0.65,
        #     0.70,
        #     0.75,
        #     0.80,
        #     0.85,
        #     0.90,
        #     0.95,
        #     1.0,
        # ],
        fontsize=fontsize_ticks,
    )
    plt.savefig(
        f"plots/alpha={alpha}/plot_ao_low_n_fidelities_simple.pdf", bbox_inches="tight"
    )
    plt.show()


def plot_ao_open_low_n_fidelities_noise(alpha=1, plot_title=False):
    open_noise_0_data = read_data(
        "always_on_fast", "open", alpha, "optimum_gammas_end_n"
    )
    open_noise_1_data = read_data(
        "always_on_fast", "open", alpha, "optimum_gammas_end_n_noise=0.005"
    )
    open_noise_2_data = read_data(
        "always_on_fast", "open", alpha, "optimum_gammas_end_n_noise=0.01"
    )
    open_noise_3_data = read_data(
        "always_on_fast", "open", alpha, "optimum_gammas_end_n_noise=0.02"
    )

    fig, ax = plt.subplots(figsize=figure_size)
    if plot_title:
        fig.suptitle(
            f"Always-on fidelity for open chain for $\\alpha={alpha}$",
            fontsize=fontsize_title,
        )

    linestyles = ["solid", "dashed", "dotted"]
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
    colours = [
        "r",
        "darkslategrey",
        "g",
        "darkorchid",
    ]
    ax.plot(
        open_noise_0_data["spins"],
        open_noise_0_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[0],
    )
    ax.plot(
        open_noise_1_data["spins"],
        open_noise_1_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[1],
    )
    ax.plot(
        open_noise_2_data["spins"],
        open_noise_2_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[2],
    )
    ax.plot(
        open_noise_3_data["spins"],
        open_noise_3_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[3],
    )

    ax.legend(
        [
            "Noise = 0",
            "Noise = 0.005",
            "Noise = 0.01",
            "Noise = 0.02",
        ],
        loc="lower left",
        fontsize=fontsize_legend - 4,
    )

    ax.set_xlabel("$n$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    ax.grid()
    plt.yticks(
        # [
        #     0.65,
        #     0.70,
        #     0.75,
        #     0.80,
        #     0.85,
        #     0.90,
        #     0.95,
        #     1.0,
        # ],
        fontsize=fontsize_ticks,
    )
    plt.savefig(
        f"plots/alpha={alpha}/plot_ao_open_low_n_fidelities_noise.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_ao_low_n_times(alpha=1, plot_title=False):
    open_mid_n_data = read_data("always_on_fast", "open", alpha, "optimum_gammas_mid_n")
    open_end_n_data = read_data("always_on_fast", "open", alpha, "optimum_gammas_end_n")

    closed_mid_n_data = read_data(
        "always_on_fast", "closed", alpha, "optimum_gammas_mid_n"
    )
    closed_end_n_data = read_data(
        "always_on_fast", "closed", alpha, "optimum_gammas_end_n"
    )

    fig, ax = plt.subplots(figsize=figure_size)
    if plot_title:
        fig.suptitle(
            f"Always-On time for $\\alpha={alpha}$",
            fontsize=fontsize_title,
        )

    linestyles = ["solid", "dashed"]
    colours = ["red", "blue"]
    ax.plot(
        open_mid_n_data["spins"],
        open_mid_n_data["time"],
        linestyle=linestyles[1],
        color=colours[0],
    )
    ax.plot(
        open_end_n_data["spins"],
        open_end_n_data["time"],
        linestyle=linestyles[0],
        color=colours[0],
    )
    ax.plot(
        closed_mid_n_data["spins"],
        closed_mid_n_data["time"],
        linestyle=linestyles[1],
        color=colours[1],
    )
    ax.plot(
        closed_end_n_data["spins"],
        closed_end_n_data["time"],
        linestyle=linestyles[0],
        color=colours[1],
    )

    solid_line = mlines.Line2D(
        [], [], color="black", label="initial site = 1, final site = n"
    )
    dashed_line = mlines.Line2D(
        [],
        [],
        color="black",
        linestyle=linestyles[1],
        label="initial site = 1, final site = n/2",
    )

    # legend1 = plt.legend(handles=[solid_line, dotted_line, x_marker], loc=3, fontsize=fontsize_legend)
    legend1 = plt.legend(
        handles=[solid_line, dashed_line],
        bbox_to_anchor=(0.56, 1),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend1)

    patches = []
    patch1 = mpatches.Patch(color=colours[0], label=f"Open chain")
    patch2 = mpatches.Patch(color=colours[1], label=f"Closed chain")
    patches.append(patch1)
    patches.append(patch2)
    legend2 = plt.legend(
        handles=patches, bbox_to_anchor=(0.32, 0.85), fontsize=fontsize_legend
    )
    ax.add_artist(legend2)

    ax.set_xlabel("$n$", fontsize=fontsize_axis)
    ax.set_ylabel("Time", fontsize=fontsize_axis)
    ax.grid()
    plt.yticks(fontsize=fontsize_ticks)
    plt.savefig(f"plots/alpha=1/plot_ao_low_n_times.pdf", bbox_inches="tight")
    plt.show()


def plot_ao_open_low_n_fidelities_various_end_n(alpha=1, plot_title=False):
    open_mid_n_data = read_data("always_on_fast", "open", alpha, "optimum_gammas_mid_n")
    open_end_n_data = read_data("always_on_fast", "open", alpha, "optimum_gammas_end_n")
    open_n_over_3_data = read_data(
        "always_on_fast", "open", alpha, "optimum_gammas_n_over_3"
    )
    open_n_over_4_data = read_data(
        "always_on_fast", "open", alpha, "optimum_gammas_n_over_4"
    )
    open_n_minus_1_data = read_data(
        "always_on_fast", "open", alpha, "optimum_gammas_n_minus_1"
    )
    open_n_minus_2_data = read_data(
        "always_on_fast", "open", alpha, "optimum_gammas_n_minus_2"
    )
    open_n_minus_3_data = read_data(
        "always_on_fast", "open", alpha, "optimum_gammas_n_minus_3"
    )

    fig, ax = plt.subplots(figsize=figure_size)
    if plot_title:
        fig.suptitle(
            f"Always-on fidelity for open chain for $\\alpha={alpha}$",
            fontsize=fontsize_title,
        )

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

    ax.plot(
        open_end_n_data["spins"],
        open_end_n_data["fidelity"],
        color=colours[0],
    )
    ax.plot(
        open_n_minus_1_data["spins"],
        open_n_minus_1_data["fidelity"],
        color=colours[1],
    )
    ax.plot(
        open_n_minus_2_data["spins"],
        open_n_minus_2_data["fidelity"],
        color=colours[2],
    )
    ax.plot(
        open_n_minus_3_data["spins"],
        open_n_minus_3_data["fidelity"],
        color=colours[3],
    )
    ax.plot(
        open_mid_n_data["spins"],
        open_mid_n_data["fidelity"],
        color=colours[4],
    )
    # ax.plot(
    #     open_n_over_3_data["spins"],
    #     open_n_over_3_data["fidelity"],
    #     color=colours[5],
    # )
    # ax.plot(
    #     open_n_over_4_data["spins"],
    #     open_n_over_4_data["fidelity"],
    #     color=colours[6],
    # )
    ax.legend(
        [
            "$w=1, f=n$",
            "$w=1, f=n-1$",
            "$w=1, f=n-2$",
            "$w=1, f=n-3$",
            "$w=1, f=n/2$",
            # "$w=1, f=n/3$",
            # "$w=1, f=n/4$",
        ],
        bbox_to_anchor=(1, 0.5),
        fontsize=fontsize_legend,
    )

    ax.set_xlabel("$n$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    ax.grid()
    plt.yticks(
        [
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.0,
        ],
        fontsize=fontsize_ticks,
    )
    plt.savefig(
        f"plots/alpha=1/plot_ao_open_low_n_fidelities_various_end_n.pdf",
        bbox_inches="tight",
    )
    plt.show()


# Reverse search protocol plots
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_rs_low_n_fidelities(alpha=1, plot_title=False):
    open_mid_n_data = read_data("reverse_search", "open", alpha, "optimum_gammas_mid_n")
    open_end_n_data = read_data("reverse_search", "open", alpha, "optimum_gammas_end_n")

    closed_mid_n_data = read_data(
        "reverse_search", "closed", alpha, "optimum_gammas_mid_n"
    )
    closed_end_n_data = read_data(
        "reverse_search", "closed", alpha, "optimum_gammas_end_n"
    )

    fig, ax = plt.subplots(figsize=figure_size)
    if plot_title:
        fig.suptitle(
            f"Reverse search fidelity for $\\alpha={alpha}$",
            fontsize=fontsize_title,
        )

    linestyles = ["solid", "dashed", "dotted"]
    colours = ["red", "blue"]
    ax.plot(
        open_mid_n_data["spins"],
        open_mid_n_data["fidelity"],
        linestyle=linestyles[1],
        color=colours[0],
    )
    ax.plot(
        open_end_n_data["spins"],
        open_end_n_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[0],
    )
    ax.plot(
        closed_mid_n_data["spins"],
        closed_mid_n_data["fidelity"],
        linestyle=linestyles[1],
        color=colours[1],
    )
    ax.plot(
        closed_end_n_data["spins"],
        closed_end_n_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[1],
    )
    ax.plot(
        open_end_n_data["spins"],
        open_end_n_data["naive_fidelity"],
        linestyle=linestyles[2],
        color=colours[0],
    )
    ax.plot(
        closed_mid_n_data["spins"],
        closed_mid_n_data["naive_fidelity"],
        linestyle=linestyles[2],
        color=colours[1],
    )

    solid_line = mlines.Line2D([], [], color="black", label="$w = 1, f = n$")
    dashed_line = mlines.Line2D(
        [],
        [],
        color="black",
        linestyle=linestyles[1],
        label="$w = 1, f = n/2$",
    )
    dotted_line = mlines.Line2D(
        [],
        [],
        color="black",
        linestyle=linestyles[2],
        label="Analytical spatial search fidelity squared",
    )

    # legend1 = plt.legend(handles=[solid_line, dotted_line, x_marker], loc=3, fontsize=fontsize_legend)
    legend1 = plt.legend(
        handles=[solid_line, dashed_line, dotted_line],
        bbox_to_anchor=(1, 0.5),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend1)

    patches = []
    patch1 = mpatches.Patch(color=colours[0], label=f"Open chain")
    patch2 = mpatches.Patch(color=colours[1], label=f"Closed chain")
    patches.append(patch1)
    patches.append(patch2)
    legend2 = plt.legend(
        handles=patches, bbox_to_anchor=(1, 0.65), fontsize=fontsize_legend
    )
    ax.add_artist(legend2)

    ax.set_xlabel("$n$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    ax.grid()
    plt.yticks(
        [
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.0,
        ],
        fontsize=fontsize_ticks,
    )
    plt.savefig(f"plots/alpha=1/plot_rs_low_n_fideilties.pdf", bbox_inches="tight")
    plt.show()


def plot_rs_low_n_fidelities_simple(alpha=1, plot_title=False):
    open_end_n_data = read_data("reverse_search", "open", alpha, "optimum_gammas_end_n")

    closed_mid_n_data = read_data(
        "reverse_search", "closed", alpha, "optimum_gammas_mid_n"
    )

    fig, ax = plt.subplots(figsize=figure_size)
    if plot_title:
        fig.suptitle(
            f"Reverse search fidelity for $\\alpha={alpha}$",
            fontsize=fontsize_title,
        )

    linestyles = ["solid", "dashed", "dotted"]
    colours = ["red", "blue"]
    ax.plot(
        open_end_n_data["spins"],
        open_end_n_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[0],
    )
    ax.plot(
        open_end_n_data["spins"],
        open_end_n_data["naive_fidelity"],
        linestyle=linestyles[2],
        color=colours[0],
    )
    ax.plot(
        closed_mid_n_data["spins"],
        closed_mid_n_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[1],
    )
    ax.plot(
        closed_mid_n_data["spins"],
        closed_mid_n_data["naive_fidelity"],
        linestyle=linestyles[2],
        color=colours[1],
    )

    ax.legend(
        [
            "Open chain: $w=1, f=n$",
            "Open chain: analytical spatial search fidelity squared",
            "Closed chain: $w=1, f=n/2$",
            "Closed chain: analytical spatial search fidelity squared",
        ],
        bbox_to_anchor=(1, 0.22),
        fontsize=fontsize_legend - 4,
    )

    ax.set_xlabel("$n$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    ax.grid()
    plt.yticks(
        # [
        #     0.65,
        #     0.70,
        #     0.75,
        #     0.80,
        #     0.85,
        #     0.90,
        #     0.95,
        #     1.0,
        # ],
        fontsize=fontsize_ticks,
    )
    plt.savefig(
        f"plots/alpha={alpha}/plot_rs_low_n_fidelities_simple.pdf", bbox_inches="tight"
    )
    plt.show()


def plot_rs_open_low_n_fidelities_noise(alpha=1, plot_title=False):
    open_noise_0_data = read_data(
        "reverse_search", "open", alpha, "optimum_gammas_end_n"
    )
    open_noise_1_data = read_data(
        "reverse_search", "open", alpha, "optimum_gammas_end_n_noise=0.005"
    )
    open_noise_2_data = read_data(
        "reverse_search", "open", alpha, "optimum_gammas_end_n_noise=0.01"
    )
    open_noise_3_data = read_data(
        "reverse_search", "open", alpha, "optimum_gammas_end_n_noise=0.02"
    )

    fig, ax = plt.subplots(figsize=figure_size)
    if plot_title:
        fig.suptitle(
            f"Reverse search fidelity for open chain for $\\alpha={alpha}$",
            fontsize=fontsize_title,
        )

    linestyles = ["solid", "dashed", "dotted"]
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
    colours = [
        "r",
        "darkslategrey",
        "g",
        "darkorchid",
    ]
    ax.plot(
        open_noise_0_data["spins"],
        open_noise_0_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[0],
    )
    ax.plot(
        open_noise_1_data["spins"],
        open_noise_1_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[1],
    )
    ax.plot(
        open_noise_2_data["spins"],
        open_noise_2_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[2],
    )
    ax.plot(
        open_noise_3_data["spins"],
        open_noise_3_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[3],
    )

    ax.legend(
        [
            "Noise = 0",
            "Noise = 0.005",
            "Noise = 0.01",
            "Noise = 0.02",
        ],
        loc="lower left",
        fontsize=fontsize_legend - 4,
    )

    ax.set_xlabel("$n$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    ax.grid()
    plt.yticks(
        # [
        #     0.65,
        #     0.70,
        #     0.75,
        #     0.80,
        #     0.85,
        #     0.90,
        #     0.95,
        #     1.0,
        # ],
        fontsize=fontsize_ticks,
    )
    plt.savefig(
        f"plots/alpha={alpha}/plot_rs_open_low_n_fidelities_noise.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_rs_low_n_times(alpha=1, plot_title=False):
    open_mid_n_data = read_data("reverse_search", "open", alpha, "optimum_gammas_mid_n")
    open_end_n_data = read_data("reverse_search", "open", alpha, "optimum_gammas_end_n")

    closed_mid_n_data = read_data(
        "reverse_search", "closed", alpha, "optimum_gammas_mid_n"
    )
    closed_end_n_data = read_data(
        "reverse_search", "closed", alpha, "optimum_gammas_end_n"
    )

    fig, ax = plt.subplots(figsize=figure_size)
    if plot_title:
        fig.suptitle(
            f"Reverse search time for $\\alpha={alpha}$",
            fontsize=fontsize_title,
        )

    linestyles = ["solid", "dashed"]
    colours = ["red", "blue"]
    ax.plot(
        open_mid_n_data["spins"],
        open_mid_n_data["time"],
        linestyle=linestyles[1],
        color=colours[0],
    )
    ax.plot(
        open_end_n_data["spins"],
        open_end_n_data["time"],
        linestyle=linestyles[0],
        color=colours[0],
    )
    ax.plot(
        closed_mid_n_data["spins"],
        closed_mid_n_data["time"],
        linestyle=linestyles[1],
        color=colours[1],
    )
    ax.plot(
        closed_end_n_data["spins"],
        closed_end_n_data["time"],
        linestyle=linestyles[0],
        color=colours[1],
    )

    solid_line = mlines.Line2D(
        [], [], color="black", label="initial site = 1, final site = n"
    )
    dashed_line = mlines.Line2D(
        [],
        [],
        color="black",
        linestyle=linestyles[1],
        label="initial site = 1, final site = n/2",
    )

    # legend1 = plt.legend(handles=[solid_line, dotted_line, x_marker], loc=3, fontsize=fontsize_legend)
    legend1 = plt.legend(
        handles=[solid_line, dashed_line],
        bbox_to_anchor=(0.56, 1),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend1)

    patches = []
    patch1 = mpatches.Patch(color=colours[0], label=f"Open chain")
    patch2 = mpatches.Patch(color=colours[1], label=f"Closed chain")
    patches.append(patch1)
    patches.append(patch2)
    legend2 = plt.legend(
        handles=patches, bbox_to_anchor=(0.32, 0.85), fontsize=fontsize_legend
    )
    ax.add_artist(legend2)

    ax.set_xlabel("$n$", fontsize=fontsize_axis)
    ax.set_ylabel("Time", fontsize=fontsize_axis)
    ax.grid()
    plt.yticks(fontsize=fontsize_ticks)
    plt.savefig(f"plots/alpha=1/plot_rs_low_n_times.pdf", bbox_inches="tight")
    plt.show()


def plot_rs_open_low_n_fidelities_various_end_n(alpha=1, plot_title=False):
    open_mid_n_data = read_data("reverse_search", "open", alpha, "optimum_gammas_mid_n")
    open_end_n_data = read_data("reverse_search", "open", alpha, "optimum_gammas_end_n")
    open_n_over_3_data = read_data(
        "reverse_search", "open", alpha, "optimum_gammas_n_over_3"
    )
    open_n_over_4_data = read_data(
        "reverse_search", "open", alpha, "optimum_gammas_n_over_4"
    )
    open_n_minus_1_data = read_data(
        "reverse_search", "open", alpha, "optimum_gammas_n_minus_1"
    )
    open_n_minus_2_data = read_data(
        "reverse_search", "open", alpha, "optimum_gammas_n_minus_2"
    )
    open_n_minus_3_data = read_data(
        "reverse_search", "open", alpha, "optimum_gammas_n_minus_3"
    )

    fig, ax = plt.subplots(figsize=figure_size)
    if plot_title:
        fig.suptitle(
            f"Always-on fidelity for open chain for $\\alpha={alpha}$",
            fontsize=fontsize_title,
        )

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

    ax.plot(
        open_end_n_data["spins"],
        open_end_n_data["fidelity"],
        color=colours[0],
    )
    ax.plot(
        open_n_minus_1_data["spins"],
        open_n_minus_1_data["fidelity"],
        color=colours[1],
    )
    ax.plot(
        open_n_minus_2_data["spins"],
        open_n_minus_2_data["fidelity"],
        color=colours[2],
    )
    ax.plot(
        open_n_minus_3_data["spins"],
        open_n_minus_3_data["fidelity"],
        color=colours[3],
    )
    ax.plot(
        open_mid_n_data["spins"],
        open_mid_n_data["fidelity"],
        color=colours[4],
    )
    # ax.plot(
    #     open_n_over_3_data["spins"],
    #     open_n_over_3_data["fidelity"],
    #     color=colours[5],
    # )
    # ax.plot(
    #     open_n_over_4_data["spins"],
    #     open_n_over_4_data["fidelity"],
    #     color=colours[6],
    # )
    ax.legend(
        [
            "$w=1, f=n$",
            "$w=1, f=n-1$",
            "$w=1, f=n-2$",
            "$w=1, f=n-3$",
            "$w=1, f=n/2$",
            # "$w=1, f=n/3$",
            # "$w=1, f=n/4$",
        ],
        bbox_to_anchor=(1, 0.5),
        fontsize=fontsize_legend,
    )

    ax.set_xlabel("$n$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    ax.grid()
    plt.yticks(
        [
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.0,
        ],
        fontsize=fontsize_ticks,
    )
    plt.savefig(
        f"plots/alpha=1/plot_ao_open_low_n_fidelities_various_end_n.pdf",
        bbox_inches="tight",
    )
    plt.show()


# Experimental always-on protocol plots
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_exp_ao_specific_n_fidelity(alpha=0.5, spins=4, plot_title=False):
    experimental_data = read_data_spin(
        "experimental/always_on_fast", "open", alpha, "optimum_gammas_end_n", spins
    )
    times, psi_states, final_state, chain = quantum_communication_exp(
        spins=spins,
        marked_strength=experimental_data["optimum_gamma"],
        switch_time=np.pi * np.sqrt(spins / 2) / experimental_data["analytical_gamma"],
        mu=np.ones(1) * experimental_data["mu"],
        omega=2
        * np.pi
        * np.array([6e6, 5e6, experimental_data["z_trap_frequency"] / (2 * np.pi)]),
        open_chain=True,
        start_site=1,
        final_site=spins,
        always_on=True,
        dt=1e-8,
        gamma_rescale=False,
    )
    qst_fidelity = chain.overlaps_evolution(final_state.subspace_ket, psi_states)

    fig, ax = plt.subplots()
    ax.plot(times, qst_fidelity)
    ax.set(xlabel="$Time~(s/\hbar)$")
    ax.grid()
    plt.show()


def plot_exp_ao_fidelities(alpha=0.5, plot_title=False):
    experimental_data = read_data(
        "experimental/always_on_fast", "open", alpha, "optimum_gammas_end_n"
    )

    ideal_data = read_data("always_on_fast", "open", alpha, "optimum_gammas_end_n")

    fig, ax = plt.subplots(figsize=figure_size)
    if plot_title:
        fig.suptitle(
            f"Always-On fidelity for target $\\alpha={alpha}$",
            fontsize=fontsize_title,
        )

    linestyles = ["solid", "dashed", "dotted"]
    colours = ["red", "blue"]
    ax.plot(
        experimental_data["spins"],
        experimental_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[0],
    )
    ax.plot(
        ideal_data["spins"][:11],
        ideal_data["fidelity"][:11],
        linestyle=linestyles[0],
        color=colours[1],
    )
    ax.plot(
        experimental_data["spins"],
        experimental_data["naive_fidelity"],
        linestyle=linestyles[1],
        color=colours[0],
    )
    ax.plot(
        ideal_data["spins"][:11],
        ideal_data["naive_fidelity"][:11],
        linestyle=linestyles[1],
        color=colours[1],
    )

    solid_line = mlines.Line2D([], [], color="black", label="Fidelity")
    dashed_line = mlines.Line2D(
        [],
        [],
        color="black",
        linestyle=linestyles[1],
        label="Analytical spatial search fidelity squared",
    )

    # legend1 = plt.legend(handles=[solid_line, dotted_line, x_marker], loc=3, fontsize=fontsize_legend)
    legend1 = plt.legend(
        handles=[solid_line, dashed_line],
        # bbox_to_anchor=(1, 0.5),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend1)

    patches = []
    patch1 = mpatches.Patch(color=colours[0], label=f"Experimental couplings")
    patch2 = mpatches.Patch(color=colours[1], label=f"Idealised couplings")
    patches.append(patch1)
    patches.append(patch2)
    legend2 = plt.legend(
        handles=patches,
        bbox_to_anchor=(1, 0.31),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend2)

    ax.set_xlabel("$N$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    ax.grid()
    plt.yticks(
        [
            0.80,
            0.85,
            0.90,
            0.95,
            1.0,
        ],
        fontsize=fontsize_ticks,
    )
    plt.savefig(
        f"plots/experimental/alpha={alpha}/plot_ao_low_n_fideilties.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_exp_ao_fidelities_peak_connectivity(alpha=0):
    experimental_data = read_data(
        "experimental/always_on_fast", "open", alpha, "optimum_gammas_end_n"
    )

    fig, ax = plt.subplots(figsize=figure_size)

    linestyles = ["solid", "dashed", "dotted"]
    colours = ["red", "blue"]
    ax.plot(
        experimental_data["spins"],
        experimental_data["fidelity"],
        linestyle=linestyles[0],
        color=colours[0],
    )
    ax.plot(
        experimental_data["spins"],
        experimental_data["naive_fidelity"],
        linestyle=linestyles[1],
        color=colours[0],
    )
    ax.set_xlabel("$N$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    ax.legend(
        ["Fidelity", "Analytical fidelity"],
        loc="center right",
        fontsize=fontsize_legend,
    )
    ax.set_yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

    ax2 = fig.add_axes([0.62, 0.205, 0.32, 0.26])
    ax2.plot(experimental_data["spins"], experimental_data["alpha"], color="black")

    ax2.set_xlabel("$N$", fontsize=18)
    ax2.set_ylabel("$\\alpha$", fontsize=18)
    ax2.set_xticks([10, 20, 30, 40, 50])
    ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    plt.tight_layout()
    plt.savefig(f"plots/experimental/alpha=0/plot_ao_low_n_peak_fidelities.pdf")
    plt.show()


def plot_exp_ao_multiple_fidelities_peak_connectivity(alphas=[0.5], subplot=None):
    experimental_datas = [
        read_data("experimental/always_on_fast", "open", alpha, "optimum_gammas_end_n")
        for alpha in alphas
    ]

    fig, ax = plt.subplots(figsize=figure_size)
    linestyles = ["solid", "dashed", "dotted"]
    colours = [
        "b",
        "g",
        "darkorchid",
        "red",
        "navy",
        "darkorange",
        "teal",
    ]

    for i, experimental_data in enumerate(experimental_datas):
        ax.plot(
            experimental_data["spins"],
            experimental_data["fidelity"],
            linestyle=linestyles[0],
            color=colours[i],
        )
        ax.plot(
            experimental_data["spins"],
            experimental_data["naive_fidelity"],
            linestyle=linestyles[1],
            color=colours[i],
        )
    ax.set_xlabel("$N$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)

    # ax.legend(
    #     ["Fidelity", "Analytical fidelity"],
    #     loc="center right",
    #     fontsize=fontsize_legend,
    # )
    patches = []
    for i, alpha in enumerate(alphas):
        patch = mpatches.Patch(color=colours[i], label=f"$\\alpha = {alpha}$")
        patches.append(patch)
    legend_patch = plt.legend(
        handles=patches, bbox_to_anchor=(0.97, 0.63), fontsize=fontsize_legend
    )
    ax.add_artist(legend_patch)

    ax.set_yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])

    savefigure = f"plots/experimental/alphas/plot_ao_low_n_peak_fidelities"
    if subplot == "alpha":
        ax2 = fig.add_axes([0.62, 0.205, 0.32, 0.26])
        for i, experimental_data in enumerate(experimental_datas):
            ax2.plot(
                experimental_data["spins"], experimental_data["alpha"], color=colours[i]
            )

        ax2.set_xlabel("$N$", fontsize=18)
        ax2.set_ylabel("$\\alpha$", fontsize=18)
        ax2.set_xticks([10, 20, 30, 40, 50])
        ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        savefigure += "_alpha"
    elif subplot == "mu":
        l = 1e-6
        ax3 = fig.add_axes([0.62, 0.205, 0.32, 0.26])
        for i, experimental_data in enumerate(experimental_datas):
            mus = [mu * l for mu in experimental_data["mu"]]
            ax3.plot(
                experimental_data["spins"],
                mus,
                color=colours[i],
            )
        ax3.set(xscale="log")
        ax3.set_xticks([10, 20, 30, 40, 50])
        ax3.set(yscale="log")
        ax3.set_yticks([37.8, 38.4, 39.0])

        ax3.set_xlabel("$N$", fontsize=18)
        ax3.set_ylabel("Detuning $\\mu~(\\mathrm{MHz})$", fontsize=18)
        savefigure += "_mu"

    plt.tight_layout()

    plt.savefig(savefigure + ".pdf")
    plt.show()


def plot_exp_ao_times():
    pass


def plot_exp_ion_spacings(spins=4):
    ion_trap = iontrap.IonTrap(spins, use_optimal_omega=True)

    l = 1e6
    spacings = [
        (ion_trap.positions[i][2] - ion_trap.positions[i - 1][2]) * l
        for i in range(1, spins, 1)
    ]
    fig, ax = plt.subplots(figsize=figure_size)
    ax.scatter(list(range(1, spins, 1)), spacings, marker="x", color="b")

    ax.set_xlabel("Ion interval", fontsize=fontsize_axis)
    ax.set_ylabel("$\\Delta z~(\\mathrm{\\mu m})$ ", fontsize=fontsize_axis)
    ax.grid()
    plt.yticks(fontsize=fontsize_ticks)

    ax2 = fig.add_axes([0.29, 0.4, 0.45, 0.42], projection="3d")
    max_position = 0
    for position in ion_trap.positions:
        x, y, z = position[2] * l, position[0] * l, position[1] * l
        ax2.scatter(x, y, z, color="blue")
        if abs(x) > max_position:
            max_position = abs(x)
    ticks = [
        -max_position,
        -max_position / 2,
        0,
        max_position / 2,
        max_position,
    ]
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_zticks(ticks)
    ax2.set_xlabel("Axial $z~(\\mathrm{\\mu m})$", fontsize=fontsize_axis - 4)
    ax2.set_ylabel("$x~(\\mathrm{\\mu m})$", fontsize=fontsize_axis - 4)
    ax2.set_zlabel("$y~(\\mathrm{\\mu m})$", fontsize=fontsize_axis - 4)
    ax2.tick_params(axis="both", labelsize=fontsize_ticks - 8)

    plt.savefig(
        f"plots/experimental/alphas/plot_exp_ion_spacings_n={spins}.pdf",
        bbox_inches="tight",
    )
    plt.show()
