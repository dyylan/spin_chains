import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import quimb
from ast import literal_eval
from brokenaxes import brokenaxes

from .data_handling import read_data, read_data_spin, update_data
from ..functions.protocols import (
    quantum_communication_exp,
    quantum_communication_exp_noise,
    quantum_communication_exp_strobe,
)
from ..functions.fits import sqrt_power_fit
from ..quantum import iontrap
from spin_chains.data_analysis import data_handling

plt.rc("text", usetex=True)
font = {"family": "serif", "size": 16, "serif": ["computer modern roman"]}
plt.rc("font", **font)

fontsize_title = 24
fontsize_axis = 22
fontsize_legend = 18
fontsize_ticks = 18
figure_size = [10, 8]

alpha_colours = {
    0.1: "b",
    0.2: "blue",
    0.3: "darkorange",
    0.4: "green",
    0.5: "navy",
    0.6: "purple",
    0.8: "red",
}

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
def plot_exp_ao_specific_n_fidelity(alpha=0.5, spins=4, plot_title=False, use_xy=False):
    protocol = (
        "experimental/always_on_fast"
        if not use_xy
        else "experimental/always_on_fast_xy"
    )
    experimental_data = read_data_spin(
        protocol, "open", alpha, "optimum_gammas_end_n", spins
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
    ax.set(xlabel="$Time~(s)$")
    ax.grid()
    plt.show()


def plot_exp_ao_fidelities(alpha=0.5, plot_title=False, use_xy=False):
    protocol = (
        "experimental/always_on_fast"
        if not use_xy
        else "experimental/always_on_fast_xy"
    )
    experimental_data = read_data(protocol, "open", alpha, "optimum_gammas_end_n")

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
        f"plots/experimental/alpha={alpha}/plot_ao_low_n_fideilties{'_xy' if use_xy else ''}.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_exp_ao_fidelities_peak_connectivity(alpha=0, use_xy=False):
    protocol = (
        "experimental/always_on_fast"
        if not use_xy
        else "experimental/always_on_fast_xy"
    )
    experimental_data = read_data(protocol, "open", alpha, "optimum_gammas_end_n")

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
    plt.savefig(
        f"plots/experimental/alpha=0/plot_ao_low_n_peak_fidelities{'_xy' if use_xy else ''}.pdf"
    )
    plt.show()


def plot_exp_ao_multiple_fidelities_peak_connectivity(
    alphas=[0.5], subplot=None, minimum_delta_mu=None, use_xy=False
):
    protocol = (
        "experimental/always_on_fast"
        if not use_xy
        else "experimental/always_on_fast_xy"
    )
    mu_tag = "" if not minimum_delta_mu else f"_mu_stability={minimum_delta_mu}kHz"
    experimental_datas = [
        read_data(
            protocol,
            "open",
            alpha,
            "optimum_gammas_end_n" + mu_tag,
        )
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
    savefigure += "_xy" if use_xy else ""
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
        if minimum_delta_mu:
            savefigure += "_mu"
            ax3.plot(
                experimental_data["spins"],
                [
                    ((2 * np.pi * minimum_delta_mu * 1e3) + (2 * np.pi * 6e6)) * l
                    for _ in range(len(experimental_data["spins"]))
                ],
                color="r",
                linestyle=linestyles[2],
            )
            savefigure += f"_min={minimum_delta_mu}kHz"
        ax3.set(xscale="log")
        ax3.set_xticks([10, 20, 30, 40, 50])
        ax3.set(yscale="log")
        ax3.set_yticks([37.8, 38.4, 39.0])

        ax3.set_xlabel("$N$", fontsize=18)
        ax3.set_ylabel("Detuning $\\mu~(\\mathrm{MHz})$", fontsize=18)

    plt.tight_layout()

    plt.savefig(savefigure + ".pdf")
    plt.show()


def plot_exp_ao_multiple_fidelities_peak_connectivity_mu_min(
    alphas=[0.5],
    subplot=[],
    plot_mu_min=True,
    dashed_lines=["analytical"],
    save_tag="",
    final_site="end_n",
    use_xy=False,
):
    protocol = (
        "experimental/always_on_fast"
        if not use_xy
        else "experimental/always_on_fast_xy"
    )
    experimental_datas = [
        read_data(
            protocol,
            "open",
            alpha,
            f"optimum_gammas_{final_site}_mu_min",
        )
        for alpha in alphas
    ]

    if "ideal" in dashed_lines:
        ideal_datas = [
            read_data(
                "always_on_fast",
                "open",
                alpha,
                "optimum_gammas_end_n",
            )
            for alpha in alphas
        ]

    fig, ax = plt.subplots(figsize=[8, 8])
    linestyles = ["solid", "dashed", "dotted"]
    colours = [
        "b",
        "g",
        "darkorchid",
        "teal",
        "red",
        "navy",
        "darkorange",
    ]

    for i, experimental_data in enumerate(experimental_datas):
        ax.plot(
            experimental_data["spins"],
            experimental_data["fidelity"],
            linestyle=linestyles[0],
            color=alpha_colours[alphas[i]],
        )
        if "analytical" in dashed_lines:
            ax.plot(
                experimental_data["spins"],
                experimental_data["naive_fidelity"],
                linestyle=linestyles[1],
                color=alpha_colours[alphas[i]],
            )
        elif "ideal" in dashed_lines:
            ax.plot(
                ideal_datas[i]["spins"],
                ideal_datas[i]["fidelity"],
                linestyle=linestyles[1],
                color=alpha_colours[alphas[i]],
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
        patch = mpatches.Patch(
            color=alpha_colours[alphas[i]],
            label=f"$\\alpha_{{\\mathrm{{target}} }} = {alpha}$",
        )
        patches.append(patch)
    legend_patch = plt.legend(
        handles=patches,
        # bbox_to_anchor=(1, 0.72),
        bbox_to_anchor=(1, 0.625),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend_patch)

    solid_line = mlines.Line2D(
        [], [], color="black", label="experimental" + " $J_{i,j}$"
    )
    dashed_line = mlines.Line2D(
        [],
        [],
        color="black",
        linestyle=linestyles[1],
        label=dashed_lines[0] + " $J_{i,j}$",
    )
    lines = [solid_line, dashed_line]
    legend_line = plt.legend(
        handles=lines,
        # bbox_to_anchor=(1, 0.59),
        bbox_to_anchor=(0.485, 0.573),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend_line)

    if "analytical" in dashed_lines:
        ax.set_yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    elif "ideal" in dashed_lines:
        # ax.set_yticks([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
        # ax.set_yticks([0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
        ax.set_yticks(
            [0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
        )

    savefigure = (
        f"plots/experimental/alphas/plot_ao_low_n_peak_fidelities_{final_site}"
        + "_"
        + dashed_lines[0]
    )
    savefigure += "_xy" if use_xy else ""
    if "alpha" in subplot:
        ax2 = fig.add_axes([0.66, 0.205, 0.28, 0.26])
        for i, experimental_data in enumerate(experimental_datas):
            ax2.plot(
                experimental_data["spins"],
                experimental_data["alpha"],
                color=alpha_colours[alphas[i]],
            )
        ax2.set_xlabel("$N$", fontsize=18)
        ax2.set_ylabel("$\\alpha$", fontsize=18)
        ax2.set_xticks([10, 20, 30, 40, 50])
        ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        savefigure += "_alpha"
    if "mu" in subplot:
        l = 1e-3
        if ("alpha" in subplot) or ("time" in subplot):
            ax3 = fig.add_axes([0.24, 0.205, 0.32, 0.26])
        else:
            ax3 = fig.add_axes([0.62, 0.205, 0.32, 0.26])
        for i, experimental_data in enumerate(experimental_datas):
            mus = [(mu - (2 * np.pi * 6e6)) * l for mu in experimental_data["mu"]]
            min_mus = [
                (min_mu - (2 * np.pi * 6e6)) * l
                for min_mu in experimental_data["minimum_mu"]
            ]
            ax3.plot(
                experimental_data["spins"],
                mus,
                color=alpha_colours[alphas[i]],
            )
        if plot_mu_min:
            ax3.plot(
                experimental_datas[0]["spins"],
                min_mus,
                color="r",
                linestyle=linestyles[1],
            )
            ax3.text(x=8, y=10, s="$\\mu_\\mathrm{min} - \\nu_x$", fontsize=18)
        ax3.plot(
            experimental_datas[0]["spins"],
            [1 for _ in experimental_data["spins"]],
            color="r",
            linestyle=linestyles[2],
        )
        ax3.text(x=10, y=1.5, s="$1~$kHz", fontsize=16)
        savefigure += f"_mu_min"
        ax3.set(xscale="log")
        ax3.set_xticks([10, 20, 30, 40, 50])
        ax3.set(yscale="log")
        # ax3.set_yticks([37.8, 38.4, 39.0])

        ax3.set_xlabel("$N$", fontsize=18)
        ax3.set_ylabel("Detuning $\\mu- \\nu_x~(\\mathrm{kHz}) $", fontsize=18)
    if "time" in subplot:
        ax4 = fig.add_axes([0.66, 0.205, 0.28, 0.26])
        for i, experimental_data in enumerate(experimental_datas):
            ax4.plot(
                experimental_data["spins"],
                [
                    time * experimental_data["optimum_gamma"][i]
                    for i, time in enumerate(experimental_data["time"])
                ],
                linestyle=linestyles[0],
                color=alpha_colours[alphas[i]],
            )
        ax4.plot(
            experimental_data["spins"],
            [sqrt_power_fit(n, 2.24, 0) for n in experimental_data["spins"]],
            linestyle=linestyles[2],
            color="red",
        )
        ax4.text(x=30, y=11, s="$\\sim \\sqrt{N}$", fontsize=16)
        ax4.set_xlabel("$N$", fontsize=18)
        ax4.set_ylabel("$\\gamma T$", fontsize=18)
        # ax4.set_xlabel("$N$")
        # ax4.set_ylabel("$\\gamma T$")

        savefigure += "_time"
    plt.tight_layout()

    save_tag = "_" + save_tag if save_tag else ""
    savefigure += save_tag
    plt.savefig(savefigure + ".pdf")
    plt.show()


def plot_exp_ao_multiple_fidelities_peak_connectivity_mid_end_comparison(
    alphas=[0.5], subplot=[], plot_mu_min=True, save_tag="", use_xy=False
):
    protocol = (
        "experimental/always_on_fast"
        if not use_xy
        else "experimental/always_on_fast_xy"
    )
    experimental_end_datas = [
        read_data(
            protocol,
            "open",
            alpha,
            f"optimum_gammas_end_n_mu_min",
        )
        for alpha in alphas
    ]
    experimental_mid_datas = [
        read_data(
            protocol,
            "open",
            alpha,
            f"optimum_gammas_mid_n_mu_min",
        )
        for alpha in alphas
    ]

    fig, ax = plt.subplots(figsize=figure_size)
    linestyles = ["solid", "dashed", "dotted"]
    colours = [
        "b",
        "g",
        "darkorchid",
        "teal",
        "red",
        "navy",
        "darkorange",
    ]

    for i, experimental_mid_data in enumerate(experimental_mid_datas):
        ax.plot(
            experimental_mid_data["spins"],
            experimental_mid_data["fidelity"],
            linestyle=linestyles[0],
            color=alpha_colours[alphas[i]],
        )
        ax.plot(
            experimental_end_datas[i]["spins"],
            experimental_end_datas[i]["fidelity"],
            linestyle=linestyles[1],
            color=alpha_colours[alphas[i]],
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
        patch = mpatches.Patch(
            color=alpha_colours[alphas[i]],
            label=f"$\\alpha_{{\\mathrm{{target}} }} = {alpha}$",
        )
        patches.append(patch)
    legend_patch = plt.legend(
        handles=patches, bbox_to_anchor=(0.97, 0.75), fontsize=fontsize_legend
    )
    ax.add_artist(legend_patch)

    solid_line = mlines.Line2D([], [], color="black", label="Final site $= N/2$")
    dashed_line = mlines.Line2D(
        [],
        [],
        color="black",
        linestyle=linestyles[1],
        label="Final site $= N$",
    )
    lines = [solid_line, dashed_line]
    legend_line = plt.legend(
        handles=lines, bbox_to_anchor=(0.97, 0.57), fontsize=fontsize_legend
    )
    ax.add_artist(legend_line)

    ax.set_yticks([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])

    savefigure = (
        f"plots/experimental/alphas/plot_ao_low_n_peak_fidelities_final_site_comparison"
    )
    savefigure += "_xy" if use_xy else ""
    if "alpha" in subplot:
        ax2 = fig.add_axes([0.62, 0.205, 0.32, 0.26])
        for i, experimental_end_data in enumerate(experimental_end_datas):
            ax2.plot(
                experimental_end_data["spins"],
                experimental_end_data["alpha"],
                color=alpha_colours[alphas[i]],
            )
        ax2.set_xlabel("$N$", fontsize=18)
        ax2.set_ylabel("$\\alpha$", fontsize=18)
        ax2.set_xticks([10, 20, 30, 40, 50])
        ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        savefigure += "_alpha"
    if "mu" in subplot:
        l = 1e-3
        if "alpha" in subplot:
            ax3 = fig.add_axes([0.2, 0.205, 0.32, 0.26])
        else:
            ax3 = fig.add_axes([0.62, 0.205, 0.32, 0.26])
        for i, experimental_end_data in enumerate(experimental_end_datas):
            mus = [(mu - (2 * np.pi * 6e6)) * l for mu in experimental_end_data["mu"]]
            min_mus = [
                (min_mu - (2 * np.pi * 6e6)) * l
                for min_mu in experimental_end_data["minimum_mu"]
            ]
            ax3.plot(
                experimental_end_data["spins"],
                mus,
                color=alpha_colours[alphas[i]],
            )
        if plot_mu_min:
            ax3.plot(
                experimental_end_data["spins"],
                min_mus,
                color="r",
                linestyle=linestyles[1],
            )
            ax3.text(x=5, y=20, s="$\\mu_\\mathrm{min} - \\nu_x$", fontsize=18)
        ax3.plot(
            experimental_end_datas[0]["spins"],
            [1 for _ in experimental_end_datas[0]["spins"]],
            color="r",
            linestyle=linestyles[2],
        )
        ax3.text(x=10, y=1.5, s="$1~$kHz", fontsize=16)
        savefigure += f"_mu_min"
        ax3.set(xscale="log")
        ax3.set_xticks([10, 20, 30, 40, 50])
        ax3.set(yscale="log")
        # ax3.set_yticks([37.8, 38.4, 39.0])

        ax3.set_xlabel("$N$", fontsize=18)
        ax3.set_ylabel("Detuning $\\mu- \\nu_x~(\\mathrm{kHz}) $", fontsize=18)

    plt.tight_layout()

    save_tag = "_" + save_tag if save_tag else ""
    savefigure += save_tag
    plt.savefig(savefigure + ".pdf")
    plt.show()


def plot_exp_ao_normalised_times(
    alphas=[0.5], subplot="fidelity", save_tag="", final_site="end_n", use_xy=False
):
    protocol = (
        "experimental/always_on_fast"
        if not use_xy
        else "experimental/always_on_fast_xy"
    )
    experimental_datas = [
        read_data(
            protocol,
            "open",
            alpha,
            f"optimum_gammas_{final_site}_mu_min",
        )
        for alpha in alphas
    ]

    fig, ax = plt.subplots(figsize=[6, 6])
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
            [
                time * experimental_data["optimum_gamma"][i]
                for i, time in enumerate(experimental_data["time"])
            ],
            linestyle=linestyles[0],
            color=alpha_colours[alphas[i]],
        )
    ax.plot(
        experimental_data["spins"],
        [sqrt_power_fit(n, 2.24, 0) for n in experimental_data["spins"]],
        linestyle=linestyles[1],
        color="black",
    )
    ax.set_xlabel("$N$", fontsize=fontsize_axis)
    ax.set_ylabel("$\\gamma T$", fontsize=fontsize_axis)
    savefigure = f"plots/experimental/alphas/plot_ao_low_n_peak_times_{final_site}"
    savefigure += "_xy" if use_xy else ""
    ax.legend(
        [f"$\\alpha_\\textrm{{target}} = {alpha}$" for alpha in alphas]
        + ["$\\sim \\sqrt{n}$"],
        fontsize=fontsize_legend,
    )

    if subplot == "fidelity":
        ax2 = fig.add_axes([0.58, 0.25, 0.36, 0.26])
        for i, experimental_data in enumerate(experimental_datas):
            ax2.plot(
                experimental_data["spins"],
                experimental_data["fidelity"],
                color=alpha_colours[alphas[i]],
            )
        ax2.set_xlabel("$N$", fontsize=18)
        ax2.set_ylabel("$F$", fontsize=18)
        ax2.set_xticks([10, 20, 30, 40, 50])
        ax2.set_yticks([0.96, 0.97, 0.98, 0.99, 1.0])
        savefigure += "_fidelity"

    plt.tight_layout()
    save_tag = "_" + save_tag if save_tag else ""
    savefigure += save_tag
    plt.savefig(savefigure + ".pdf")
    plt.show()


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


def plot_exp_ao_specific_n_fidelity_strobe(
    alpha=0.5, spins=4, n_strobe=5, use_xy=False
):
    protocol = (
        "experimental/always_on_fast"
        if not use_xy
        else "experimental/always_on_fast_xy"
    )
    experimental_data = read_data_spin(
        protocol, "open", alpha, "optimum_gammas_end_n", spins
    )
    times, psi_states, final_state, chain = quantum_communication_exp_strobe(
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
        n_strobe=n_strobe,
    )
    qst_fidelity = chain.overlaps_evolution(final_state.subspace_ket, psi_states)

    fig, ax = plt.subplots()
    ax.plot(times, qst_fidelity)
    ax.set(xlabel="$Time~(s)$")
    ax.grid()
    plt.show()


def plot_exp_ao_specific_n_fidelity_strobe_comparison(
    alpha=0.5, spins_list=[4], n_strobes=[2, 3, 4], use_xy=False, show_legend=True
):
    colours = [
        "black",
        "b",
        # "g",
        "green",
        "red",
        # "navy",
        # "darkorange",
        "teal",
    ]
    protocol = (
        "experimental/always_on_fast"
        if not use_xy
        else "experimental/always_on_fast_xy"
    )
    experimental_datas = [
        read_data_spin(protocol, "open", alpha, "optimum_gammas_end_n_mu_min", spins)
        for spins in spins_list
    ]
    times_ = []
    fidelities_ = []
    for i, spins in enumerate(spins_list):
        qst_times = []
        qst_fidelities = []
        times, psi_states, final_state, chain = quantum_communication_exp(
            spins=spins,
            marked_strength=experimental_datas[i]["optimum_gamma"],
            switch_time=np.pi
            * np.sqrt(spins / 2)
            / experimental_datas[i]["analytical_gamma"],
            mu=np.ones(1) * experimental_datas[i]["mu"],
            omega=np.array(
                [
                    2 * np.pi * 6e6,
                    2 * np.pi * 5e6,
                    experimental_datas[i]["z_trap_frequency"],
                ]
            ),
            open_chain=True,
            start_site=1,
            final_site=spins,
            always_on=True,
            dt=1e-7,
            gamma_rescale=False,
            use_xy=use_xy,
        )
        qst_times.append(times)
        qst_fidelities.append(
            chain.overlaps_evolution(final_state.subspace_ket, psi_states)
        )

        for n_strobe in n_strobes:
            times, psi_states, final_state, chain = quantum_communication_exp_strobe(
                spins=spins,
                marked_strength=experimental_datas[i]["optimum_gamma"],
                switch_time=np.pi
                * np.sqrt(spins / 2)
                / experimental_datas[i]["analytical_gamma"],
                mu=np.ones(1) * experimental_datas[i]["mu"],
                omega=2
                * np.pi
                * np.array(
                    [6e6, 5e6, experimental_datas[i]["z_trap_frequency"] / (2 * np.pi)]
                ),
                open_chain=True,
                start_site=1,
                final_site=spins,
                always_on=True,
                dt=1e-7,
                gamma_rescale=False,
                n_strobe=n_strobe,
                use_xy=use_xy,
            )
            qst_times.append(times)
            qst_fidelities.append(
                chain.overlaps_evolution(final_state.subspace_ket, psi_states)
            )
        times_.append(qst_times)
        fidelities_.append(qst_fidelities)

    fig, ax = plt.subplots(figsize=[6, 6])
    fig.suptitle(
        f"$N ={spins_list[0]}$",
        fontsize=fontsize_title,
    )
    for i, qst_fidelities in enumerate(fidelities_):
        for j, y in enumerate(qst_fidelities):
            ax.plot(times_[i][j], y, color=colours[j])
    if show_legend:
        ax.legend(
            ["Non-stroboscopic"] + [f"Strobes $= {s}$" for s in n_strobes],
            # loc="center right",
            fontsize=fontsize_legend,
        )
    ax.set(xlabel="Time~(s)")
    plt.savefig(
        f"plots/experimental/alpha={alpha}/plot_exp_ao_fidelity_strobe_comparison_n={spins_list}{'_xy' if use_xy else ''}.pdf",
        bbox_inches="tight",
    )
    ax.grid()
    plt.show()


# Experimental always-on protocol plots with noise
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_exp_ao_specific_n_fidelity_with_noise(
    alpha=0.5, spins=4, t2_list=[1e-3], samples=100, use_xy=False
):
    protocol = (
        "experimental/always_on_fast"
        if not use_xy
        else "experimental/always_on_fast_xy"
    )
    qst_fidelities = []
    experimental_data = read_data_spin(
        protocol,
        "open",
        alpha,
        f"optimum_gammas_end_n",
        spins,
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
    qst_fidelities.append(
        chain.overlaps_evolution(final_state.subspace_ket, psi_states)
    )
    for t2 in t2_list:
        times, psi_states, final_state, chain = quantum_communication_exp_noise(
            spins=spins,
            marked_strength=experimental_data["optimum_gamma"],
            switch_time=np.pi
            * np.sqrt(spins / 2)
            / experimental_data["analytical_gamma"],
            mu=np.ones(1) * experimental_data["mu"],
            omega=2
            * np.pi
            * np.array([6e6, 5e6, experimental_data["z_trap_frequency"] / (2 * np.pi)]),
            t2=t2,
            samples=samples,
            open_chain=True,
            start_site=1,
            final_site=spins,
            always_on=True,
            dt=1e-8,
            gamma_rescale=False,
        )
        qst_fidelities.append(
            chain.overlaps_noisy_evolution(final_state.subspace_ket, psi_states)
        )

    fig, ax = plt.subplots(figure_size=[8, 8])
    for f in qst_fidelities:
        ax.plot(times, f)
    ax.set(xlabel="$Time~(s)$")
    ax.legend(
        ["No noise"] + [f"$t_2 = {t2}$" for t2 in t2_list],
        # loc="center right",
        fontsize=fontsize_legend,
    )
    ax.grid()
    plt.savefig(
        f"plots/experimental/alpha={alpha}/plot_noise_fidelities_n={spins}{'_xy' if use_xy else ''}.pdf"
    )
    plt.show()


def plot_exp_ao_multiple_t2s(
    alpha=0.5, t2_list=[0.01], subplot=None, plot_title=False, use_xy=False
):
    protocol = (
        "experimental/always_on_fast"
        if not use_xy
        else "experimental/always_on_fast_xy"
    )
    experimental_data_no_noise = read_data(
        protocol, "open", alpha, f"optimum_gammas_end_n" + ("_mu_min" if use_xy else "")
    )

    experimental_datas = [
        read_data(
            "experimental/always_on_fast",
            "open",
            alpha,
            f"optimum_gammas_end_n_t2={t2}_samples=2000",
        )
        for t2 in t2_list
    ]

    fig, ax = plt.subplots(figsize=[6, 6])
    if plot_title:
        fig.suptitle(
            f"$\\alpha_{{\\mathrm{{target}}}}={alpha}$",
            fontsize=fontsize_title,
        )
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
    ax.plot(
        experimental_data_no_noise["spins"],
        experimental_data_no_noise["fidelity"],
        linestyle=linestyles[1],
        color=alpha_colours[alpha],
    )
    for i, experimental_data in enumerate(experimental_datas):
        ax.plot(
            experimental_data["spins"],
            experimental_data["fidelity"],
            linestyle=linestyles[0],
            color=alpha_colours[alpha],
        )
        ax.fill_between(
            experimental_data["spins"],
            experimental_data["fidelity_lower_error"],
            experimental_data["fidelity_upper_error"],
            color=alpha_colours[alpha],
            alpha=0.3,
        )
    ax.set_xlabel("$N$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)

    if subplot == "alpha":
        ax.legend(
            ["No noise"] + [f"$t_2 = {t2}~$s" for t2 in t2_list],
            loc="lower left",
            bbox_to_anchor=(0, 0.42),
            fontsize=fontsize_legend,
        )
    else:
        ax.legend(
            ["No noise"] + [f"$t_2 = {t2}~$s" for t2 in t2_list],
            loc="best",
            fontsize=fontsize_legend,
        )
    # patches = []
    # for i, alpha in enumerate(alphas):
    #     patch = mpatches.Patch(color=colours[i], label=f"$\\alpha = {alpha}$")
    #     patches.append(patch)
    # legend_patch = plt.legend(
    #     handles=patches, bbox_to_anchor=(0.97, 0.63), fontsize=fontsize_legend
    # )
    # ax.add_artist(legend_patch)
    if alpha in [0.1, 0.2]:
        ax.set_yticks([0.990, 0.992, 0.994, 0.996, 0.998, 1.0])
    else:
        ax.set_yticks([0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.0])

    savefigure = f"plots/experimental/alpha={alpha}/plot_ao_low_n_noise_fidelities"
    if subplot == "alpha":
        ax2 = fig.add_axes([0.29, 0.23, 0.28, 0.22])
        for i, experimental_data in enumerate(experimental_datas):
            ax2.plot(
                experimental_data["spins"], experimental_data["alpha"], color=colours[i]
            )

        ax2.set_xlabel("$N$", fontsize=14)
        ax2.set_ylabel("$\\alpha$", fontsize=14)
        ax2.set_xticks([10, 20, 30, 40, 50])
        ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax2.tick_params(axis="both", which="major", labelsize=12)
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
    savefigure += "_xy" if use_xy else ""
    plt.savefig(savefigure + ".pdf")
    plt.show()


def plot_exp_ao_multiple_alphas(
    alpha_list=[0.4],
    t2=0.01,
    samples=2000,
    subplot=None,
    plot_title=False,
    use_xy=False,
):
    protocol = (
        "experimental/always_on_fast"
        if not use_xy
        else "experimental/always_on_fast_xy"
    )
    experimental_datas_no_noise = [
        read_data(
            protocol,
            "open",
            alpha,
            f"optimum_gammas_end_n" + ("_mu_min" if use_xy else ""),
        )
        for alpha in alpha_list
    ]

    experimental_datas = [
        read_data(
            protocol,
            "open",
            alpha,
            f"optimum_gammas_end_n"
            + ("_mu_min" if use_xy else "")
            + f"_t2={t2}_samples={samples}",
        )
        for alpha in alpha_list
    ]

    fig, ax = plt.subplots(figsize=[8, 6])
    if plot_title:
        fig.suptitle(
            f"$t_2={t2}$",
            fontsize=fontsize_title,
        )
    linestyles = ["solid", "dashed", "dotted"]

    for i, alpha in enumerate(alpha_list):
        ax.plot(
            experimental_datas_no_noise[i]["spins"],
            experimental_datas_no_noise[i]["fidelity"],
            linestyle=linestyles[1],
            color=alpha_colours[alpha],
        )
        ax.plot(
            experimental_datas[i]["spins"],
            experimental_datas[i]["fidelity"],
            linestyle=linestyles[0],
            color=alpha_colours[alpha],
        )
        ax.fill_between(
            experimental_datas[i]["spins"],
            experimental_datas[i]["fidelity_lower_error"],
            experimental_datas[i]["fidelity_upper_error"],
            color=alpha_colours[alpha],
            alpha=0.3,
        )

    ax.set_xlabel("$N$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)

    legend_alphas = []
    for i, alpha in enumerate(alpha_list):
        legend_alphas.append(
            mlines.Line2D(
                [],
                [],
                color=alpha_colours[alpha],
                label=f"$\\alpha_\\textrm{{target}} = {alpha}$",
            )
        )

    legend0 = plt.legend(
        handles=legend_alphas,
        loc="lower left",
        fontsize=fontsize_legend - 2,
    )
    ax.add_artist(legend0)

    lines = [
        mlines.Line2D(
            [],
            [],
            color="black",
            linestyle="dashed",
            label="No noise",
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            linestyle="solid",
            label=f"$t_2 = {t2}$",
        ),
    ]

    legend_lines = plt.legend(
        handles=lines,
        loc="lower right",
        # bbox_to_anchor=(1, 0.37),
        fontsize=fontsize_legend - 2,
    )
    ax.add_artist(legend_lines)

    # ax.set_yticks([0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.0])
    ax.set_yticks(
        [
            # 0.87,
            0.88,
            # 0.89,
            0.90,
            # 0.91,
            0.92,
            # 0.93,
            0.94,
            # 0.95,
            0.96,
            # 0.97,
            0.98,
            # 0.99,
            1.0,
        ]
    )

    savefigure = f"plots/experimental/alphas/plot_ao_low_n_noise_fidelities_t2={t2}"

    plt.tight_layout()
    savefigure += "_xy" if use_xy else ""
    plt.savefig(savefigure + ".pdf")
    plt.show()


def plot_exp_broken_spin(
    alpha=0.1, t2=0.01, ratios=[10], subplot="alpha", plot_title=True
):
    experimental_datas = [
        read_data(
            "experimental/always_on_fast",
            "open",
            alpha,
            f"optimum_gammas_end_n_t2={t2}_samples=5000_broken_spin=n_over_2_broken_spin_factor={r}",
        )
        for r in ratios
    ]

    fig, ax = plt.subplots(figsize=[8, 6])
    if plot_title:
        fig.suptitle(
            f"$\\alpha_{{\\mathrm{{target}}}}={alpha}$",
            fontsize=fontsize_title,
        )
    linestyles = ["solid", "dashed", "dotted"]
    colours = ["red", "royalblue", "darkorange", "teal"]

    for i, experimental_data in enumerate(experimental_datas):
        ax.plot(
            experimental_data["spins"],
            experimental_data["fidelity"],
            linestyle=linestyles[0],
            color=colours[i],
        )
        ax.fill_between(
            experimental_data["spins"],
            experimental_data["fidelity_lower_error"],
            experimental_data["fidelity_upper_error"],
            color=colours[i],
            alpha=0.3,
        )
    ax.set_xlabel("$N$", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)

    ax.legend(
        [f"$R = {r}$" for r in ratios],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        fontsize=fontsize_legend,
    )

    # patches = []
    # for i, alpha in enumerate(alphas):
    #     patch = mpatches.Patch(color=colours[i], label=f"$\\alpha = {alpha}$")
    #     patches.append(patch)
    # legend_patch = plt.legend(
    #     handles=patches, bbox_to_anchor=(0.97, 0.63), fontsize=fontsize_legend
    # )
    # ax.add_artist(legend_patch)

    ax.set_yticks(
        [
            0.70,
            0.72,
            0.74,
            0.76,
            0.78,
            0.80,
            0.82,
            0.84,
            0.86,
            0.88,
            0.90,
            0.92,
            0.94,
            0.96,
            0.98,
            1.0,
        ]
    )

    savefigure = (
        f"plots/experimental/alpha={alpha}/plot_ao_low_n_noise_fidelities_broken_spin"
    )
    if subplot == "alpha":
        ax2 = fig.add_axes([0.23, 0.24, 0.28, 0.22])
        for i, experimental_data in enumerate(experimental_datas):
            ax2.plot(
                experimental_data["spins"], experimental_data["alpha"], color="black"
            )

        ax2.set_xlabel("$N$", fontsize=16)
        ax2.set_ylabel("$\\alpha$", fontsize=16)
        ax2.set_xticks([10, 20, 30, 40, 50])
        ax2.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
        ax2.tick_params(axis="both", which="major", labelsize=14)
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


# Spin-boson fidelity and purity plots
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_spin_boson_fidelity_comparisons(n_ions, n_phonons, alphas):
    fidelities = []
    times = []
    for alpha in alphas:
        data = data_handling.read_data_spin(
            protocol="experimental/always_on_fast_xy",
            chain="open",
            alpha=alpha,
            save_tag="optimum_gammas_end_n_mu_min",
            spins=n_ions,
        )
        mu = data["mu"]
        z_trap_frequency = data["z_trap_frequency"]
        states_full = data_handling.read_all_state_data(
            "spin_boson_single_mode",
            "open",
            alpha,
            n_ions,
            mu,
            z_trap_frequency,
            "full_state",
        )
        states_xy = data_handling.read_all_state_data(
            "spin_boson_single_mode",
            "open",
            alpha,
            n_ions,
            mu,
            z_trap_frequency,
            "xy_state",
        )

        # Times
        times_full = states_full["time_microseconds"]
        times_xy = states_xy["time_microseconds"]

        # States
        qu_states_full = [
            quimb.qu(state, qtype="ket") for state in states_full["states"]
        ]
        qu_states_xy = [quimb.qu(state, qtype="ket") for state in states_xy["states"]]

        # Fidelity and time
        qu_states_full_pt = [
            quimb.partial_trace(
                state,
                [n_phonons] + ([2] * n_ions),
                [i for i in range(1, n_ions + 1, 1)],
            )
            for state in qu_states_full
        ]
        fidelities.append(
            [
                quimb.expectation(qu_states_xy[i], state_full)
                for i, state_full in enumerate(qu_states_full_pt)
            ]
        )
        times.append(times_full)

    fig, ax = plt.subplots()
    for i, alpha in enumerate(alphas):
        ax.plot(times[i], fidelities[i])
    ax.legend(
        [f"$\\alpha = {alpha}$" for alpha in alphas],
        fontsize=fontsize_legend,
    )
    ax.set(xlabel="Time $(\\mu s)$")
    ax.grid()
    plt.show()


def plot_spin_boson_fom_for_alpha(
    n_ions,
    alphas,
    fom="fidelity",
    save_fig=False,
    plot_title=False,
    plot_fits=False,
    plot_dephasing=False,
):
    def calculate_leakage(t, omega_eff, omega_single_mode, g_single_mode):
        E = (
            (1 - np.cos((omega_eff - omega_single_mode) * t))
            * np.power(g_single_mode[0], 2)
            / (2 * np.power(omega_eff - omega_single_mode, 2))
        )
        return 1 - E

    foms = []
    times = []
    fits = []
    mus = []
    alphas_ = []

    mu_dict = {}
    z_trap_dict = {}
    for alpha in alphas:
        data = data_handling.read_data_spin(
            protocol="experimental/always_on_fast_xy",
            chain="open",
            alpha=alpha,
            save_tag="optimum_gammas_end_n_mu_min",
            spins=n_ions,
        )
        mu = data["mu"]
        a = data["alpha"]
        z_trap_frequency = data["z_trap_frequency"]

        fom_dict = data_handling.read_data(
            protocol="spin_boson_single_mode",
            chain="open",
            alpha=alpha,
            save_tag=f"{fom}_n={n_ions}_mu={mu}_z_trap={z_trap_frequency}",
        )
        if plot_fits:
            ion_trap = iontrap.IonTrapXY(
                n_ions,
                mu=np.ones(1) * mu,
                omega=np.array(
                    [2 * np.pi * 6e6, 2 * np.pi * 5e6, data["z_trap_frequency"]]
                ),
            )
            (
                g_single_mode,
                omega_single_mode,
                omega_eff,
                delta,
            ) = ion_trap.single_mode_ms()

            leakage = [
                calculate_leakage(t, omega_eff, omega_single_mode, g_single_mode)
                for t in fom_dict["time"]
            ]
            fits.append(leakage)

        foms.append(fom_dict[fom])
        times.append(fom_dict["time"])
        mus.append(mu)
        alphas_.append(a)

        mu_dict[alpha] = mu
        z_trap_dict[alpha] = z_trap_frequency

    alpha_colours = {
        0.2: "royalblue",
        0.4: "green",
        0.8: "red",
    }
    if plot_dephasing:
        fig, ax = plt.subplots(figsize=[10, 8])
    else:
        fig, ax = plt.subplots(figsize=[10, 8])
    if plot_title:
        fig.suptitle(
            f"$N = {n_ions}$",
            fontsize=fontsize_title,
        )
    legend_alphas = []
    for i, alpha in enumerate(alphas):
        ax.plot(times[i], foms[i], color=alpha_colours[alpha])
        legend_alphas.append(
            mlines.Line2D(
                [],
                [],
                color=alpha_colours[alpha],
                label=f"$\\alpha = {alpha}$ with $\\mu = {mus[i]/1000000:.6f}$ MHz",
            )
        )
        if plot_fits:
            ax.plot(times[i], fits[i], color=alpha_colours[alpha], linestyle="dotted")

    legend0 = plt.legend(
        handles=legend_alphas,
        loc="lower left",
        # bbox_to_anchor=(1, 0.5),
        fontsize=fontsize_legend,
    )
    ax.add_artist(legend0)

    if plot_fits:
        fit_line = mlines.Line2D(
            [],
            [],
            color="black",
            linestyle="dotted",
            label="$1 - \\mathcal{E}(t)$",
        )

        # legend1 = plt.legend(handles=[solid_line, dotted_line, x_marker], loc=3, fontsize=fontsize_legend)
        legend_fit_line = plt.legend(
            handles=[fit_line],
            bbox_to_anchor=(0.97, 0.11),
            fontsize=fontsize_legend,
        )
        ax.add_artist(legend_fit_line)

    alpha_dephasing_colours = {
        0.2: "darkslategray",
        0.4: "lawngreen",
        0.8: "red",
    }
    t2_dephasing_colours = {
        0.0038: "darkblue",
        0.0036: "indigo",
        0.0030: "mediumvioletred",
        0.043: "lawngreen",
    }

    if plot_dephasing:
        t2s = {0.2: 0.0035, 0.4: 0.043}
        alphas = {
            0.0030: 0.2,
            0.0036: 0.2,
            0.0038: 0.2,
            0.043: 0.4,
        }
        t2s_line = []
        for t2 in alphas:
            alpha = alphas[t2]
            dephasing_dict = data_handling.read_data(
                protocol="spin_boson_single_mode",
                chain="open",
                alpha=alpha,
                save_tag=f"xy_model_fidelity_n={n_ions}_t2={t2}_mu={mu_dict[alpha]}_z_trap={z_trap_dict[alpha]}",
            )
            ax.plot(
                dephasing_dict["time"],
                dephasing_dict["fidelity"],
                linewidth=2,
                color=t2_dephasing_colours[t2],
                linestyle="dashed",
            )
            t2s_line.append(
                mlines.Line2D(
                    [],
                    [],
                    color=t2_dephasing_colours[t2],
                    linestyle="dashed",
                    label=f"$t_2 = {t2}$ s -- $\\alpha = {alpha}$",
                )
            )

        # legend1 = plt.legend(handles=[solid_line, dotted_line, x_marker], loc=3, fontsize=fontsize_legend)
        legend_t2s_line = plt.legend(
            handles=t2s_line,
            bbox_to_anchor=(0.45, 0.47),
            fontsize=fontsize_legend,
        )
        ax.add_artist(legend_t2s_line)

    ax.set_xlabel("Time (s)", fontsize=fontsize_axis)
    ax.set_ylabel(fom.capitalize(), fontsize=fontsize_axis)
    # ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # ax.set_yticks([0.8, 0.9, 1.0])
    # ax.set_yticks([0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.0])

    ax.grid()
    if save_fig:
        plt.savefig(
            f"plots/experimental/alphas/{fom}_n={n_ions}{'_with_leakage' if plot_fits else ''}{'_with_dephasing' if plot_dephasing else ''}.pdf",
            bbox_inches="tight",
        )
    plt.show()


def plot_spin_boson_leakage_with_fit(
    n_ions,
    alpha,
    save_fig=False,
    plot_title=False,
    plot_label=None,
    omega_eff_r=0,
):
    def calculate_leakage(t, omega_eff, omega_single_mode, g_single_mode):
        E = (
            (1 - np.cos((omega_eff - omega_single_mode) * t))
            * np.power(g_single_mode[0], 2)
            / (2 * np.power(omega_eff - omega_single_mode, 2))
        )
        return E

    data = data_handling.read_data_spin(
        protocol="experimental/always_on_fast_xy",
        chain="open",
        alpha=alpha,
        save_tag="optimum_gammas_end_n_mu_min",
        spins=n_ions,
    )
    mu = data["mu"]
    a = data["alpha"]
    z_trap_frequency = data["z_trap_frequency"]

    leakage_dict = data_handling.read_data(
        protocol="spin_boson_single_mode",
        chain="open",
        alpha=alpha,
        save_tag=f"leakage_n={n_ions}_mu={mu}_z_trap={z_trap_frequency}",
    )

    ion_trap = iontrap.IonTrapXY(
        n_ions,
        mu=np.ones(1) * mu,
        omega=np.array([2 * np.pi * 6e6, 2 * np.pi * 5e6, data["z_trap_frequency"]]),
    )
    (
        g_single_mode,
        omega_single_mode,
        omega_eff,
        delta,
    ) = ion_trap.single_mode_ms()

    fit = [
        calculate_leakage(t, omega_eff, omega_single_mode, g_single_mode)
        for t in leakage_dict["time"]
    ]

    if omega_eff_r:
        omega_eff_fit = [
            calculate_leakage(
                t, omega_eff_r * omega_eff, omega_single_mode, g_single_mode
            )
            for t in leakage_dict["time"]
        ]
    colours = ["royalblue", "green", "red"]

    if plot_label:
        fig, ax = plt.subplots(figsize=[4.4, 7])
    else:
        fig, ax = plt.subplots(figsize=figure_size)

    if plot_title:
        fig.suptitle(
            # f"$\\alpha = {alpha}$ with $\\mu = {mu/1000000:.6f}$ MHz",
            f"$\\alpha = {alpha}$",
            fontsize=fontsize_title,
        )

    alpha_colours = {
        0.2: "royalblue",
        0.4: "green",
        0.8: "red",
    }
    ax.plot(
        leakage_dict["time"][:5000],
        leakage_dict["leakage"][:5000],
        color=alpha_colours[alpha],
    )
    ax.plot(leakage_dict["time"][:5000], fit[:5000], color="black", linestyle="dotted")
    if omega_eff_r:
        ax.plot(
            leakage_dict["time"][:5000],
            omega_eff_fit[:5000],
            color="black",
            linestyle="dashed",
        )

    ax.legend(
        [
            "$E(t)$",
            "$\\|\\mathcal{E}(t)\\|$",
            f"$\\|\\mathcal{{E}}^\\prime (t)\\|$",
        ],
        loc="lower right",
    )

    ax.set_xlabel("Time (s)", fontsize=fontsize_axis)
    # ax.set_yticks([0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005])

    if plot_label:
        ax.text(0.025, 0.90, plot_label, fontsize=26, transform=plt.gcf().transFigure)
    ax.grid()
    if save_fig:
        plt.savefig(
            f"plots/experimental/alphas/leakage_n={n_ions}_alpha={alpha}.pdf",
            bbox_inches="tight",
        )
    plt.show()


def plot_spin_boson_leakage_two_phonon_modes_with_fit(
    n_ions,
    alpha,
    save_fig=False,
    plot_title=False,
    plot_label=None,
    hs_correction=False,
    omega_eff_r=0,
):
    def calculate_leakage(t, ion, omega_eff, omega_modes):
        E = (
            (
                (1 - np.cos((omega_eff - omega_modes[0]) * t))
                * np.power(gs[ion - 1][0], 2)
                / (2 * np.power(omega_eff - omega_modes[0], 2))
            )
            + (
                (
                    1
                    - np.cos((omega_eff - omega_modes[0]) * t)
                    - np.cos((omega_eff - omega_modes[1]) * t)
                    + np.cos(((omega_modes[0]) - omega_modes[1]) * t)
                )
                * gs[ion - 1][0]
                * gs[ion - 1][1]
            )
            / (2 * (omega_eff - omega_modes[0]) * (omega_eff - omega_modes[1]))
            + (
                (1 - np.cos((omega_eff - omega_modes[1]) * t))
                * np.power(gs[ion - 1][1], 2)
                / (2 * np.power(omega_eff - omega_modes[1], 2))
            )
        )
        return E

    data = data_handling.read_data_spin(
        protocol="experimental/always_on_fast_xy",
        chain="open",
        alpha=alpha,
        save_tag="optimum_gammas_end_n_mu_min",
        spins=n_ions,
    )
    mu = data["mu"]
    a = data["alpha"]
    z_trap_frequency = data["z_trap_frequency"]
    hs_correct_save_tag = "hs_correction_" if hs_correction else ""

    leakage_dict = data_handling.read_data(
        protocol="spin_boson_two_mode",
        chain="open",
        alpha=alpha,
        save_tag=f"leakage_{hs_correct_save_tag}n={n_ions}_mu={mu}_z_trap={z_trap_frequency}",
    )

    ion_trap = iontrap.IonTrapXY(
        n_ions,
        mu=np.ones(1) * mu,
        omega=np.array([2 * np.pi * 6e6, 2 * np.pi * 5e6, data["z_trap_frequency"]]),
    )

    gs, omega_modes, omega_eff, deltas = ion_trap.two_mode_ms()

    fit = [
        np.mean(
            [calculate_leakage(t, i, omega_eff, omega_modes) for i in range(n_ions)]
        )
        for t in leakage_dict["time"]
    ]

    if omega_eff_r:
        omega_eff_fit = [
            np.mean(
                [
                    calculate_leakage(t, i, omega_eff_r * omega_eff, omega_modes)
                    for i in range(n_ions)
                ]
            )
            for t in leakage_dict["time"]
        ]

    if plot_label:
        fig, ax = plt.subplots(figsize=[4.4, 7])
    else:
        fig, ax = plt.subplots(figsize=[8, 8])

    if plot_title:
        fig.suptitle(
            f"$\\alpha = {alpha}$",
            # f"$\\alpha = {alpha}$ with $\\mu = {mu/1000000:.6f}$ MHz",
            fontsize=fontsize_title,
        )

    alpha_colours = {
        0.2: "royalblue",
        0.4: "green",
        0.8: "red",
    }
    points = 5000
    ax.plot(
        leakage_dict["time"][:points],
        leakage_dict["leakage"][:points],
        color=alpha_colours[alpha],
    )
    ax.plot(
        leakage_dict["time"][:points], fit[:points], color="black", linestyle="dotted"
    )
    if omega_eff_fit:
        ax.plot(
            leakage_dict["time"][:points],
            omega_eff_fit[:points],
            color="black",
            linestyle="dashed",
        )

    ax.legend(
        [
            "$E(t)$",
            "$\\left[\\|\\mathcal{E}_{2}(t)\\|\\right]_k$",
            "$\\left[\\|\\mathcal{E}_{2}^\\prime (t)\\|\\right]_k$",
        ],
        loc="lower right",
    )

    ax.set_xlabel("Time (s)", fontsize=fontsize_axis)
    ax.set_xticks([0, 0.00005, 0.0001])
    # ax.set_yticks([0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005])

    if plot_label:
        ax.text(0.025, 0.90, plot_label, fontsize=26, transform=plt.gcf().transFigure)
    ax.grid()
    if save_fig:
        plt.savefig(
            f"plots/experimental/alphas/leakage_{hs_correct_save_tag}n={n_ions}_alpha={alpha}_phonon_modes=2.pdf",
            bbox_inches="tight",
        )
    plt.show()


def plot_phonon_occupation_numbers(
    n_ions, n_phonons, alpha, save_fig=False, plot_title=False, plot_label=False
):
    data = data_handling.read_data_spin(
        protocol="experimental/always_on_fast_xy",
        chain="open",
        alpha=alpha,
        save_tag="optimum_gammas_end_n_mu_min",
        spins=n_ions,
    )
    mu = data["mu"]
    a = data["alpha"]
    z_trap_frequency = data["z_trap_frequency"]

    phonons_dict = data_handling.read_data(
        protocol="spin_boson_single_mode",
        chain="open",
        alpha=alpha,
        save_tag=f"phonons_n={n_ions}_n_phonons={n_phonons}_mu={mu}_z_trap={z_trap_frequency}",
    )

    if plot_label:
        fig, axs = plt.subplots(2, figsize=[6, 8])
    else:
        fig, axs = plt.subplots(2, figsize=[6, 6])

    if plot_title:
        fig.suptitle(
            f"$\\alpha = {alpha}$ with $\\mu = {mu/1000000:.6f}$ MHz",
            fontsize=fontsize_title,
        )

    colours = ["royalblue", "green", "red", "purple"]
    # ax.plot(
    #     phonons_dict["time"][:5000],
    #     phonons_dict["leakage"][:5000],
    #     color=alpha_colours[alpha],
    # )
    # ax.plot(phonons_dict["time"][:5000], fit[:5000], color="black", linestyle="dotted")
    for i in range(n_phonons - 1):
        y = [literal_eval(result)[i] for result in phonons_dict["phonons"]]
        axs[1].plot(phonons_dict["time"], y, color=colours[i])
        axs[0].plot(phonons_dict["time"], y, color=colours[i])

    axs[1].set_ylim(0, 0.035)
    axs[0].set_ylim(0.965, 1)

    axs[1].spines["top"].set_visible(False)
    axs[0].spines["bottom"].set_visible(False)

    # axs[1].yaxis.tick_left()
    axs[0].set_xticks([])  # don't put tick labels at the top
    # axs[1].set_xticks([])  # don't put tick labels at the top

    axs[1].legend(
        [f"{i} " + "phonons" if i != 1 else "1 phonon" for i in range(n_phonons - 1)],
        bbox_to_anchor=(0.95, 1.35),
        fontsize=fontsize_legend + 2,
    )

    plt.subplots_adjust(wspace=0.20)
    axs[1].set_xlabel("Time (s)", fontsize=fontsize_axis + 2)
    # ax.set_yticks([0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005])

    if plot_label:
        axs[0].text(
            0.02, 0.93, plot_label, fontsize=26, transform=plt.gcf().transFigure
        )
    # axs.grid()

    d = 0.020  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=axs[1].transAxes, color="k", clip_on=False)
    axs[1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    axs[1].plot((-d, d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    kwargs.update(transform=axs[0].transAxes)  # switch to the bottom axes
    axs[0].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-left diagonal
    axs[0].plot((-d, d), (-d, +d), **kwargs)  # top-right diagonal

    if save_fig:
        plt.savefig(
            f"plots/experimental/alpha={alpha}/phonons_n={n_ions}_n_phonons={n_phonons}_alpha={alpha}.pdf",
            bbox_inches="tight",
        )
    plt.show()


def plot_phonon_number_expectation(
    n_ions, n_phonons, alphas, save_fig=False, plot_title=False, plot_label=False
):
    times = []
    phonon_num = []
    for alpha in alphas:
        data = data_handling.read_data_spin(
            protocol="experimental/always_on_fast_xy",
            chain="open",
            alpha=alpha,
            save_tag="optimum_gammas_end_n_mu_min",
            spins=n_ions,
        )
        mu = data["mu"]
        a = data["alpha"]
        z_trap_frequency = data["z_trap_frequency"]

        n_phonons_dict = data_handling.read_data(
            protocol="spin_boson_single_mode",
            chain="open",
            alpha=alpha,
            save_tag=f"n_phonons_n={n_ions}_n_phonons={n_phonons}_mu={mu}_z_trap={z_trap_frequency}",
        )
        times.append(n_phonons_dict["time"])
        phonon_num.append(n_phonons_dict["n_phonons"])

    if plot_label:
        fig, ax = plt.subplots(figsize=[6, 8])
    else:
        fig, ax = plt.subplots(figsize=figure_size)

    if plot_title:
        fig.suptitle(
            f"$\\alpha = {alpha}$ with $\\mu = {mu/1000000:.6f}$ Hz",
            fontsize=fontsize_title,
        )

    alpha_colours = {
        0.2: "royalblue",
        0.4: "green",
        0.8: "red",
    }
    for i, alpha in enumerate(alphas):
        ax.plot(
            times[i],
            phonon_num[i],
            color=alpha_colours[alpha],
        )

    ax.legend([f"$\\alpha = {alpha}$" for alpha in alphas])

    ax.set_xlabel("Time (s)", fontsize=fontsize_axis)
    # ax.set_yticks([0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005])

    if plot_label:
        ax.text(0.025, 0.90, plot_label, fontsize=26, transform=plt.gcf().transFigure)
    ax.grid()
    if save_fig:
        plt.savefig(
            f"plots/experimental/alphas/n_phonons_n={n_ions}_n_phonons={n_phonons}_alpha={alpha}.pdf",
            bbox_inches="tight",
        )
    plt.show()


def plot_phonon_number_expectation_higher_subspace(
    n_ions,
    n_phonons,
    alpha,
    rs,
    subspace=1,
    save_fig=False,
    plot_title=False,
    plot_label=False,
    points=5000,
):
    if not subspace:
        subspace = n_ions
    data = data_handling.read_data_spin(
        protocol="experimental/always_on_fast_xy",
        chain="open",
        alpha=alpha,
        save_tag="optimum_gammas_end_n_mu_min",
        spins=n_ions,
    )
    mu = data["mu"]
    a = data["alpha"]
    z_trap_frequency = data["z_trap_frequency"]
    ion_trap_xy = iontrap.IonTrapXY(
        n_ions=n_ions,
        mu=np.ones(1) * (mu * 1),
        omega=np.array([2 * np.pi * 6e6, 2 * np.pi * 5e6, z_trap_frequency]),
    )
    gs, omega_single_mode, omega_eff, delta = ion_trap_xy.single_mode_ms()

    n_phonons_dict = data_handling.read_data(
        protocol="spin_boson_single_mode",
        chain="open",
        alpha=alpha,
        save_tag=f"subspace_n_over_{subspace}_n_phonons_n={n_ions}_mu={mu}_z_trap={z_trap_frequency}",
    )
    phonons_dict = data_handling.read_data(
        protocol="spin_boson_single_mode",
        chain="open",
        alpha=alpha,
        save_tag=f"subspace_n_over_{subspace}_phonons_n={n_ions}_mu={mu}_z_trap={z_trap_frequency}",
    )
    times = n_phonons_dict["time"][:points]
    times_micro = [t * 1000000 for t in times]
    phonon_num = n_phonons_dict["n_phonons"][:points]

    phonon_times = phonons_dict["time"][:points]
    phonon_times_micro = [t * 1000000 for t in phonon_times]
    phonons = phonons_dict["phonons"][:points]

    if plot_label:
        fig, ax = plt.subplots(figsize=[6, 8])
    else:
        fig, ax = plt.subplots(figsize=[8, 8])

    if plot_title:
        fig.suptitle(
            f"${n_ions // subspace}$ excitation{'' if n_ions // subspace == 1 else 's'}, $N={n_ions}$, $\\alpha = {alpha}$",
            fontsize=fontsize_title,
        )

    alpha_colours = {
        0.2: "royalblue",
        0.4: "green",
        0.8: "red",
    }

    Es_colours = ["c", "m", "y"]
    Es_linestyles = ["dotted", "dashed"]
    phonon_colours = ["green", "red", "purple"]
    legend = []

    if n_ions != subspace:
        ax.plot(
            times_micro,
            phonon_num,
            color=alpha_colours[alpha],
        )
        legend.append("$\\bar{n}(t)$")

    plot_phonons = n_phonons - 1 if subspace > 2 else n_phonons
    for i in range(1, plot_phonons):
        y = [literal_eval(result)[i] for result in phonons]
        ax.plot(phonon_times_micro, y, color=phonon_colours[i - 1])
        legend.append(f"{i} phonon" + ("" if i == 1 else "s"))

    def calculate_leakage(t, r):
        E = (n_ions / subspace) * (
            (1 - np.cos(((r * omega_eff) - omega_single_mode) * t))
            * np.power(gs[0], 2)
            / (2 * np.power((r * omega_eff) - omega_single_mode, 2))
        )
        return E

    for i, r in enumerate(rs):
        Es = [calculate_leakage(t, r) for t in times]
        # ax.plot(times, Es, linestyle="dashed", color=Es_colours[i])
        ax.plot(times_micro, Es, linestyle=Es_linestyles[i], color="black")
        # legend.append(
        #     f"${n_ions//subspace} \\Vert\\mathcal{{E}}^\\prime (t)\\Vert$ with $\\omega_\\textrm{{eff}}^\\prime = {r}\\omega_\\textrm{{eff}}$"
        # )
        if n_ions // subspace == 1:
            legend.append(
                f"$\\Vert\\mathcal{{E}} (t)\\Vert$"
                if np.isclose(r, 1.0)
                else f"$\\Vert\\mathcal{{E}}^\\prime (t)\\Vert$"
            )
        else:
            legend.append(
                f"${n_ions//subspace} \\Vert\\mathcal{{E}} (t)\\Vert$"
                if np.isclose(r, 1.0)
                else f"${n_ions//subspace}\\Vert\\mathcal{{E}}^\\prime (t)\\Vert$"
            )
    ax.set_xlabel("Time (µs)", fontsize=fontsize_axis)
    # ax.set_yticks([0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005])

    ax.legend(legend, loc="lower right")
    if plot_label:
        ax.text(0.025, 0.90, plot_label, fontsize=26, transform=plt.gcf().transFigure)
    ax.grid()
    if save_fig:
        plt.savefig(
            f"plots/experimental/alphas/subspace_n_over_{subspace}_n_phonons_n={n_ions}_n_phonons={n_phonons}_alpha={alpha}.pdf",
            bbox_inches="tight",
        )
    plt.show()


def plot_full_stroboscopic(
    n_ions,
    alpha,
    strobes,
    save_fig=False,
    plot_title=False,
    plot_label=None,
    plot_xy=False,
):
    experimental_data = data_handling.read_data_spin(
        protocol="experimental/always_on_fast_xy",
        chain="open",
        alpha=alpha,
        save_tag="optimum_gammas_end_n_mu_min",
        spins=n_ions,
    )
    mu = experimental_data["mu"]
    a = experimental_data["alpha"]
    z_trap_frequency = experimental_data["z_trap_frequency"]
    strobe_time = experimental_data["time"] / strobes

    strobes_dict = data_handling.read_data(
        protocol="spin_boson_single_mode",
        chain="open",
        alpha=alpha,
        save_tag=f"transfer_fidelity_full_n={n_ions}_strobes={strobes}_mu={mu}_z_trap={z_trap_frequency}",
    )

    switch_time = strobes_dict["time"][-1]

    if plot_xy:
        strobes_xy_dict = data_handling.read_data(
            protocol="spin_boson_single_mode",
            chain="open",
            alpha=alpha,
            save_tag=f"transfer_fidelity_xy_n={n_ions}_strobes={strobes}_mu={mu}_z_trap={z_trap_frequency}",
        )

    ion_trap = iontrap.IonTrapXY(
        n_ions,
        mu=np.ones(1) * mu,
        omega=np.array([2 * np.pi * 6e6, 2 * np.pi * 5e6, z_trap_frequency]),
    )

    (
        gs,
        omega_single_mode,
        omega_eff,
        delta,
    ) = ion_trap.single_mode_ms()
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
    gamma = 4 * Js[0][0] * n_ions
    print(f"Js = {Js[0][0]}")
    print(f"gamma = {gamma}")

    qst_times, psi_states, final_state, chain = quantum_communication_exp_strobe(
        spins=n_ions,
        marked_strength=gamma,
        switch_time=switch_time,
        strobe_time=strobe_time,
        mu=np.ones(1) * mu,
        omega=2 * np.pi * np.array([6e6, 5e6, z_trap_frequency / (2 * np.pi)]),
        start_site=1,
        final_site=n_ions,
        dt=1e-7,
        use_xy=True,
        single_mode=True,
    )
    qst_fidelities = chain.overlaps_evolution(final_state.subspace_ket, psi_states)

    if plot_label:
        fig, ax = plt.subplots(figsize=[6, 8])
    else:
        fig, ax = plt.subplots(figsize=[8, 8])

    if plot_title:
        fig.suptitle(
            f"$\\alpha = {alpha}$ with $\\mu = {mu:.2f}$ Hz",
            fontsize=fontsize_title,
        )

    alpha_colours = {
        0.2: "royalblue",
        0.4: "green",
        0.8: "red",
    }
    ax.plot(
        strobes_dict["time"],
        strobes_dict["fidelity"],
        color=alpha_colours[alpha],
    )
    ax.plot(qst_times, qst_fidelities, color="black", linestyle="dotted")

    if plot_xy:
        ax.plot(
            strobes_xy_dict["time"],
            strobes_xy_dict["fidelity"],
            color=alpha_colours[alpha],
            linestyle="dashed",
        )
    ax.legend(
        ["Full dynamics", "XY Model"] + (["XY Model (quimb)"] if plot_xy else []),
        # loc="lower right",
        loc="upper right",
    )

    ax.set_xlabel("Time (s)", fontsize=fontsize_axis)
    # ax.set_xticks([0, 0.001, 0.002, 0.003, 0.004, 0.005])
    # ax.set_yticks([0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005])

    if plot_label:
        ax.text(0.025, 0.90, plot_label, fontsize=26, transform=plt.gcf().transFigure)
    ax.grid()
    if save_fig:
        plt.savefig(
            f"plots/experimental/alphas/transfer_fidelity_n={n_ions}_alpha={alpha}.pdf",
            bbox_inches="tight",
        )
    plt.show()


def plot_spin_boson_fidelity_xy_with_r_comparisons(
    n_ions,
    n_phonons,
    alpha,
    rs=[],
    phonon_modes=1,
    plot_title=False,
    plot_label=False,
    save_fig=False,
    hs_correction=False,
    subspace=1,
):
    if phonon_modes == 1:
        protocol = "spin_boson_single_mode"
    elif phonon_modes == 2:
        protocol = "spin_boson_two_mode"

    if subspace > 1:
        subspace_tag = f"subspace_n_over_{subspace}_"
    else:
        subspace_tag = ""
    hs_tag = "hs_correction_" if hs_correction else ""
    fidelities = []
    times = []
    mus = []
    alphas_ = []

    rs = [1] + rs

    for r in rs:
        data = data_handling.read_data_spin(
            protocol="experimental/always_on_fast_xy",
            chain="open",
            alpha=alpha,
            save_tag="optimum_gammas_end_n_mu_min",
            spins=n_ions,
        )
        mu = data["mu"]
        a = data["alpha"]
        z_trap_frequency = data["z_trap_frequency"]

        fidelities_dict = data_handling.read_data(
            protocol=protocol,
            chain="open",
            alpha=alpha,
            save_tag=f"{subspace_tag}fidelity_{hs_tag}n={n_ions}_mu={mu}_z_trap={z_trap_frequency}"
            + (f"_r={r}" if r != 1 else f"_r={r}" if subspace > 1 else ""),
        )

        fidelities.append([abs(complex(f)) for f in fidelities_dict["fidelity"]])
        times.append(fidelities_dict["time"])
        mus.append(mu)
        alphas_.append(a)

    r_colours = ["royalblue", "green", "red", "purple"]
    fig, ax = plt.subplots(figsize=[6, 8])

    if plot_title:
        if subspace > 1:
            fig.suptitle(
                f"{n_ions//subspace} Excitations, $N = {n_ions}$, $\\alpha = {alpha}$",
                fontsize=fontsize_title,
            )
        elif phonon_modes == 2:
            fig.suptitle(
                # f"$N = {n_ions}$, $\\alpha = {alpha}$ with $\\mu = {mu/1000000:.6f}$ MHz",
                f"Two phonon modes, $N = {n_ions}$, $\\alpha = {alpha}$",
                fontsize=fontsize_title,
            )
        else:
            fig.suptitle(
                # f"$N = {n_ions}$, $\\alpha = {alpha}$ with $\\mu = {mu/1000000:.6f}$ MHz",
                f"One phonon mode, $N = {n_ions}$, $\\alpha = {alpha}$",
                fontsize=fontsize_title,
            )
    legend_rs = []
    for i, r in enumerate(rs):
        J_legend_label = (
            (
                f"$J_{{ij}}^\\prime = {r:.3f}J_{{ij}}$"
                if r != 1
                else f"$J_{{ij}}^\\prime = J_{{ij}}$"
            )
            if phonon_modes == 1
            else (
                f"$\\langle J_{{ij}}^\\prime \\rangle = {r:.3f} \\langle J_{{ij}} \\rangle$"
                if r != 1
                else f"$J_{{ij}}^\\prime  =  J_{{ij}} $"
            )
        )
        ax.plot(times[i], fidelities[i], color=r_colours[i])
        legend_rs.append(
            mlines.Line2D(
                [],
                [],
                color=r_colours[i],
                label=J_legend_label,
            )
        )
    legend = plt.legend(
        handles=legend_rs,
        loc="lower left",
        # bbox_to_anchor=(1, 0.5),
        fontsize=fontsize_legend + 4,
    )
    ax.add_artist(legend)

    ax.set_xlabel("Time (s)", fontsize=fontsize_axis)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis)
    # ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # ax.set_yticks([0.8, 0.9, 1.0])
    # ax.set_yticks([0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.0])
    if plot_label:
        ax.text(0.025, 0.90, plot_label, fontsize=26, transform=plt.gcf().transFigure)

    ax.grid()
    if save_fig:
        plt.savefig(
            f"plots/experimental/alphas/{subspace_tag}fidelity_{hs_tag}rs_n={n_ions}_phonon_modes={phonon_modes}.pdf",
            bbox_inches="tight",
        )
    plt.show()


def plot_spin_boson_compare_phonon_modes(
    n_ions,
    n_phonons,
    alpha,
    phonon_modes=[1, 2],
    correct_hs=[],
    time=1000,
    plot_title=False,
    save_fig=False,
):
    fidelities = []
    times = []
    pm_protocol_dict = {1: "spin_boson_single_mode", 2: "spin_boson_two_mode"}
    for pm in phonon_modes:
        correct_hs_tag = "_hs_correction" if pm in correct_hs else ""
        data = data_handling.read_data_spin(
            protocol="experimental/always_on_fast_xy",
            chain="open",
            alpha=alpha,
            save_tag="optimum_gammas_end_n_mu_min",
            spins=n_ions,
        )
        mu = data["mu"]
        a = data["alpha"]
        z_trap_frequency = data["z_trap_frequency"]

        fidelities_dict = data_handling.read_data(
            protocol=pm_protocol_dict[pm],
            chain="open",
            alpha=alpha,
            save_tag=f"fidelity{correct_hs_tag}_n={n_ions}_mu={mu}_z_trap={z_trap_frequency}",
        )

        time_index = [
            i for i, t in enumerate(fidelities_dict["time"]) if t >= time * 1e-6
        ][0]

        fidelities.append(fidelities_dict["fidelity"][:time_index])
        times.append(fidelities_dict["time"][:time_index])

    fig, ax = plt.subplots(figsize=[8, 8])

    if plot_title:
        fig.suptitle(
            f"$N = {n_ions}$, $\\alpha = {alpha}$",
            fontsize=fontsize_title,
        )
    legend = []
    for i, pm in enumerate(phonon_modes):
        ax.plot(times[i], fidelities[i])
        legend.append(f"{pm} phonon mode" + ("s" if pm > 1 else ""))

    ax.legend(legend, loc="lower left", fontsize=fontsize_legend + 4)

    ax.set_xlabel("Time (s)", fontsize=fontsize_axis + 4)
    ax.set_ylabel("Fidelity", fontsize=fontsize_axis + 4)
    # ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # ax.set_yticks([0.8, 0.9, 1.0])
    # ax.set_yticks([0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.0])

    ax.grid()
    if save_fig:
        plt.savefig(
            f"plots/experimental/alphas/fidelity_phonon_modes_n={n_ions}.pdf",
            bbox_inches="tight",
        )
    plt.show()
