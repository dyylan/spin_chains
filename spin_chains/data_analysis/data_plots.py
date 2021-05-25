import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from .data_handling import read_data, update_data

plt.rc("text", usetex=True)
font = {"family": "serif", "size": 16, "serif": ["computer modern roman"]}
plt.rc("font", **font)

fontsize_title = 24
fontsize_axis = 22
fontsize_legend = 18
fontsize_ticks = 18
figure_size = [10, 8]


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
    plt.savefig(f"plots/alpha=1/plot_ao_low_n_fideilties.pdf", bbox_inches="tight")
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
        [
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
        f"plots/alpha=1/plot_ao_low_n_fidelities_simple.pdf", bbox_inches="tight"
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
        [
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
        f"plots/alpha=1/plot_ao_open_low_n_fidelities_noise.pdf", bbox_inches="tight"
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
