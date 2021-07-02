import autograd as ag
from autograd import numpy as agnp
import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
import GPyOpt

from .fits import inverse_power_fit, yukawa_inverse_power_fit

plt.rc("text", usetex=True)
font = {"family": "serif", "size": 16, "serif": ["computer modern roman"]}
plt.rc("font", **font)

fontsize_title = 24
fontsize_axis = 22
fontsize_legend = 18
fontsize_ticks = 18
figure_size = [10, 8]


class IonTrap:

    k = 8.98755179e9  # coulomb constant
    hbar = 1.054571817e-34  # reduced planck constant
    e = 1.602176634e-19  # charge of ion

    def __init__(
        self,
        n_ions,
        mu=np.ones(1) * 2 * np.pi * 6.01e6,
        omega=2 * np.pi * np.array([6e6, 5e6, 1e6]),
        Omega=2 * np.pi * 1e6,
        ion_mass=171 * 1.660539066e-27,
        use_optimal_omega=False,
    ):
        self.l = 1e-6  # length scale
        self.m = ion_mass  # mass of the ion
        self.n = n_ions
        self.mu = mu
        if use_optimal_omega:
            z_trap = 0.9 * (2 * np.log(n_ions) * 6e6) / (3 * n_ions)
            self.omega = 2 * np.pi * np.array([6e6, 5e6, z_trap])
        else:
            self.omega = omega  # x, y and z trap frequencies
        self.Omega = (
            Omega if isinstance(Omega, list) else np.ones((n_ions, 1)) * Omega
        )  # rabi frequency
        self.delta_k = (
            2 * 2 * np.pi / 355e-9 * np.array([1, 0, 0])
        )  # wave vector difference

        self.calculate_Js()

    def calculate_Js(self):
        self.positions = self._calculate_equilibrium_positions()
        self.hessian = self._calculate_equilibrium_hessian()
        (
            self.eigenfrequencies,
            self.eigenmodes,
        ) = self._calculate_normal_mode_eigenpairs()
        self.Js = self._spin_interaction_graph()
        # print("Calculated Js")
        return self.Js

    def update_mu(self, mu):
        self.mu = mu
        self.calculate_Js()

    def update_omega(self, omega):
        self.omega = omega
        self.calculate_Js()

    def update_mu_for_target_alpha(
        self, target_alpha, mu_bounds, steps, verbosity=True
    ):
        def cost(params):
            parameters = params[0].tolist()
            mu_trial = parameters[0]
            self.update_mu(np.ones(1) * mu_trial)
            alpha, _ = self._calculate_alpha(
                x=0,
                verbosity=verbosity,
            )
            alpha_diff = abs(alpha - target_alpha)
            if verbosity:
                print(f"distance to target alpha = {alpha_diff}")
            return alpha_diff

        bounds = [
            {
                "name": "mu",
                "type": "continuous",
                "domain": (mu_bounds[0][0], mu_bounds[1][0]),
            }
        ]
        optimisation = GPyOpt.methods.BayesianOptimization(
            cost,
            domain=bounds,
            model_type="GP",
            acquisition_type="EI",
            normalize_Y=True,
            acquisition_weight=2,
            maximize=False,
        )

        max_iter = steps
        max_time = 180
        optimisation.run_optimization(max_iter, max_time, verbosity=verbosity)

        mu = optimisation.x_opt
        self.update_mu(np.ones(1) * mu)
        alpha, Js = self._calculate_alpha(x=0, verbosity=verbosity)
        return mu, alpha, Js

    def nondimensionalize_decorator(expr):
        def _expr(x, n, m, omega, l):
            return expr(x * l, n, m, omega, l) * l / (IonTrap.k * IonTrap.e ** 2)

        return _expr

    def flatten_decorator(expr):
        def _expr(x, n, m, omega, l):
            return expr(x.reshape(n, 3), n, m, omega, l)

        return _expr

    # nondimensionalized coulomb potential
    @staticmethod
    @nondimensionalize_decorator
    def expr_coul(x, n, m, omega, l):
        combs = np.triu_indices(n, k=1)
        return (
            IonTrap.k
            * IonTrap.e ** 2
            / agnp.linalg.norm(x[combs[0]] - x[combs[1]], axis=-1)
        ).sum()

    # nondimensionalized harmonic potential
    @staticmethod
    @nondimensionalize_decorator
    def expr_har(x, n, m, omega, l):
        return (m * omega ** 2 * x ** 2 / 2).sum()

    # flattened nondimensionalized total potential
    @staticmethod
    @flatten_decorator
    def expr_tot(x, n, m, omega, l):
        return IonTrap.expr_coul(x, n, m, omega, l) + IonTrap.expr_har(
            x, n, m, omega, l
        )

    # hessian of flattened nondimensionalized total potential
    @staticmethod
    def hess_expr_tot(x, n, m, omega, l):
        return ag.hessian(IonTrap.expr_tot, 0)(x, n, m, omega, l)

    # equilibrium position
    def _calculate_equilibrium_positions(self):
        x_ep = (
            opt.minimize(
                IonTrap.expr_tot,
                np.pad(
                    np.linspace(-(self.n - 1) / 2, (self.n - 1) / 2, self.n).reshape(
                        self.n, 1
                    ),
                    pad_width=[[0, 0], [2, 0]],
                ).flatten(),
                args=(self.n, self.m, self.omega, self.l),
                method="SLSQP",
                options={"maxiter": 1000},
                tol=1e-15,
            ).x.reshape(self.n, 3)
            * self.l
        )
        return x_ep

    # hessian evaluated at equilibrium position
    def _calculate_equilibrium_hessian(self):
        return IonTrap.hess_expr_tot(
            self.positions.flatten() / self.l, self.n, self.m, self.omega, self.l
        )

    # normal mode eigenpairs
    def _calculate_normal_mode_eigenpairs(self):
        lamb, b = np.linalg.eigh(self.hessian)
        eigenfrequencies = np.sqrt(
            lamb * IonTrap.k * IonTrap.e ** 2 / (self.l ** 3 * self.m)
        )  # eigenfrequencies
        eigenmodes = b.reshape(self.n, 3, 3 * self.n)  # eigenvectors
        return eigenfrequencies, eigenmodes

    # Calculates spin interaction graph
    def _spin_interaction_graph(self):
        eta = np.einsum(
            "ikm,k,m->im",
            self.eigenmodes,
            self.delta_k,
            np.sqrt(IonTrap.hbar / (2 * self.m * self.eigenfrequencies)),
        )

        zeta = np.einsum("im,in->imn", self.Omega, eta)
        Js = np.einsum(
            "ij,imn,jmn,n,mn->ij",
            1 - np.identity(self.n),
            zeta,
            zeta,
            self.eigenfrequencies,
            1 / np.subtract.outer(self.mu ** 2, self.eigenfrequencies ** 2),
        )
        return Js

    # Returns the alpha value from a specific mu
    def _calculate_alpha(self, x=0, verbosity=True):
        if x:
            ions = [x - i for i in range(x)] + [i - x for i in range(x + 1, self.n)]
            Js = np.delete(self.Js[x], x)
        else:
            ions = [i for i in range(1, self.n)]
            Js = self.Js[x][1:]

        popt, pcov = opt.curve_fit(
            inverse_power_fit, ions, Js, bounds=(0, [250000.0, 2.0, 10.0])
        )
        if verbosity:
            print(
                f"Fit for Js with {self.n} spins and mu = {self.mu[0]}: y = {popt[0]} r^-{popt[1]} + {popt[2]}"
                f"\n\talpha = {popt[1]}"
            )
        alpha = popt[1]
        return alpha, Js

    def plot_Js(self):
        plt.imshow(self.Js)
        plt.show()

    def plot_ion_chain_positions(self):
        # fig = plt.figure(figsize=[12, 12])
        fig = plt.figure(figsize=[10, 8])
        ax = fig.add_subplot(111, projection="3d")
        max_position = 0
        l = 1e6
        for position in self.positions:
            x, y, z = position[2] * l, position[0] * l, position[1] * l
            ax.scatter(x, y, z, color="blue")
            if abs(x) > max_position:
                max_position = abs(x)
        ticks = [-max_position, -max_position / 2, 0, max_position / 2, max_position]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
        ax.set_xlabel("Axial $\\mathrm{\\mu m}$", fontsize=fontsize_axis)
        ax.set_ylabel("$\\mathrm{\\mu m}$", fontsize=fontsize_axis)
        ax.set_zlabel("$\\mathrm{\\mu m}$", fontsize=fontsize_axis)
        ax.tick_params(axis="both", labelsize=fontsize_ticks - 6)
        plt.show()

    def plot_spin_interactions(self, x=0, exp_fit=True):
        if x:
            ions = [x - i for i in range(x)] + [i - x for i in range(x + 1, self.n)]
            Js = np.delete(self.Js[x], x)
        else:
            ions = [i for i in range(1, self.n)]
            Js = self.Js[x][1:]

        popt, pcov = opt.curve_fit(
            inverse_power_fit, ions, Js, bounds=(0, [250000.0, 2.0, 10.0])
        )
        print(
            f"Fit for Js with {self.n} spins and mu = {self.mu}: y = {popt[0]} x^-{popt[1]} + {popt[2]}"
        )

        fig, ax = plt.subplots(figsize=figure_size)
        linestyles = ["", "dashed", "dashed"]
        markers = ["d", "", ""]
        colours = ["b", "r", "g"]

        alpha_fit = inverse_power_fit(ions, popt[0], popt[1], 0)

        ys = [Js, alpha_fit]
        # legend = [f"Numerical $J_{{1 j}}$", f"${popt[0]:.2f} r^{{-{popt[1]:.3f}}}$"]
        legend = [f"Numerical $J_{{1 j}}$", f"$\propto r^{{-{popt[1]:.3f}}}$"]

        if exp_fit:
            popt, pcov = opt.curve_fit(
                yukawa_inverse_power_fit,
                ions,
                Js,
                bounds=(0, [250000.0, 0.1, 2.0, 1.0]),
            )
            print(
                f"Fit for Js with {self.n} spins and mu = {self.mu}: y = {popt[0]} exp(-{popt[1]} x) x^-{popt[2]} + {popt[3]}"
            )
            alpha_yukawa_fit = yukawa_inverse_power_fit(
                ions, popt[0], popt[1], popt[2], popt[3]
            )
            ys.append(alpha_yukawa_fit)
            # legend.append(f"${popt[0]:.2f} e^{{-{popt[1]:.3f} r}} r^{{-{popt[2]:.3f}}}$")
            legend.append(f"$\propto e^{{-{popt[1]:.3f} r}} r^{{-{popt[2]:.3f}}}$")

        for i, y in enumerate(ys):
            ax.plot(
                ions, y, linestyle=linestyles[i], color=colours[i], marker=markers[i]
            )

        ax.legend(legend, fontsize=fontsize_legend)

        ax.set_xlabel(xlabel="$r = j - 1$", fontsize=fontsize_axis)
        ax.set_ylabel(ylabel="Coupling strength $J_{1 j}$", fontsize=fontsize_axis)

        ax.grid()
        # ax.set(xscale="log")
        # ax.set(yscale="log")
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)
        plt.savefig(
            f"plots/experimental_Js/end_spins={self.n}_mu={self.mu}.pdf",
            bbox_inches="tight",
        )
        plt.show()


# Omega = np.ones((N, 1)) * 2 * np.pi * 1e6  # rabi frequency
# mu = np.ones(1) * 2 * np.pi * 6.01e6  # raman beatnote detuning
# delta_k = 2 * 2 * np.pi / 355e-9 * np.array([1, 0, 0])  # wave vector difference

# J = spin_interaction_graph(Omega, mu, delta_k)  # spin interaction graph

# plt.imshow(J)  # plot spin interaction graph
# plt.show()


# def plot_end_spin_interactions(Js, n, mu, exp_fit=True):
#     Js = Js[1:]
#     ions = [i for i in range(1, n)]

#     popt, pcov = opt.curve_fit(
#         inverse_power_fit, ions, Js, bounds=(0, [250000.0, 2.0, 10.0])
#     )
#     print(
#         f"Fit for Js with {n} spins and mu = {mu}: y = {popt[0]} x^-{popt[1]} + {popt[2]}"
#     )

#     fig, ax = plt.subplots(figsize=figure_size)
#     linestyles = ["", "dashed", "dashed"]
#     markers = ["d", "", ""]
#     colours = ["b", "r", "g"]

#     alpha_fit = inverse_power_fit(ions, popt[0], popt[1], 0)

#     ys = [Js, alpha_fit]
#     # legend = [f"Numerical $J_{{1 j}}$", f"${popt[0]:.2f} r^{{-{popt[1]:.3f}}}$"]
#     legend = [f"Numerical $J_{{1 j}}$", f"$\propto r^{{-{popt[1]:.3f}}}$"]

#     if exp_fit:
#         popt, pcov = opt.curve_fit(
#             yukawa_inverse_power_fit,
#             ions,
#             Js,
#             bounds=(0, [250000.0, 0.1, 2.0, 1.0]),
#         )
#         print(
#             f"Fit for Js with {n} spins and mu = {mu}: y = {popt[0]} exp(-{popt[1]} x) x^-{popt[2]} + {popt[3]}"
#         )
#         alpha_yukawa_fit = yukawa_inverse_power_fit(
#             ions, popt[0], popt[1], popt[2], popt[3]
#         )
#         ys.append(alpha_yukawa_fit)
#         # legend.append(f"${popt[0]:.2f} e^{{-{popt[1]:.3f} r}} r^{{-{popt[2]:.3f}}}$")
#         legend.append(f"$\propto e^{{-{popt[1]:.3f} r}} r^{{-{popt[2]:.3f}}}$")

#     for i, y in enumerate(ys):
#         ax.plot(
#             ions, y, linestyle=linestyles[i], color=colours[i], marker=markers[i]
#         )

#     ax.legend(legend, fontsize=fontsize_legend)

#     ax.set_xlabel(xlabel="$r = j - 1$", fontsize=fontsize_axis)
#     ax.set_ylabel(ylabel="Coupling strength $J_{1 j}$", fontsize=fontsize_axis)

#     ax.grid()
#     # ax.set(xscale="log")
#     # ax.set(yscale="log")
#     plt.xticks(fontsize=fontsize_ticks)
#     plt.yticks(fontsize=fontsize_ticks)
#     plt.savefig(
#         f"plots/experimental_Js/end_spins={n}_mu={mu}.pdf",
#         bbox_inches="tight",
#     )
#     plt.show()

# # Returns the alpha value from a specific mu
# def calculate_alpha(n_ions, Omega, mu, delta_k, verbosity=True):
#     x_ep = calculate_equilibrium_positions(n_ions)
#     hessian = calculate_equilibrium_hessian(x_ep, n_ions)
#     w, b = calculate_normal_mode_eigenpairs(hessian, n_ions)
#     Js = spin_interaction_graph(n_ions, Omega, mu, delta_k, w, b)

#     Js_end = Js[0][1:]
#     ions = [i for i in range(1, n_ions)]

#     popt, pcov = opt.curve_fit(
#         inverse_power_fit, ions, Js_end, bounds=(0, [250000.0, 2.0, 10.0])
#     )
#     if verbosity:
#         print(
#             f"Fit for Js with {n_ions} spins and mu = {mu[0]}: y = {popt[0]} r^-{popt[1]} + {popt[2]}"
#             f"\n\talpha = {popt[1]}"
#         )
#     alpha = popt[1]
#     return alpha, Js

# # Find mu for specific alpha
# def mu_for_specific_alpha(target_alpha, mu_range, steps, n_ions, Omega, delta_k):
#     mus = np.linspace(mu_range[0][0], mu_range[1][0], steps)
#     Js = 0
#     mu = 0
#     alpha = 0
#     diff_alpha = 100
#     for trial_mu in mus:
#         t_mu = np.ones(1) * trial_mu
#         alpha_trial, Js_trial = calculate_alpha(n_ions, Omega, t_mu, delta_k)
#         diff_alpha_trial = target_alpha - alpha_trial
#         if abs(diff_alpha_trial) < diff_alpha:
#             Js = Js_trial
#             mu = t_mu
#             alpha = alpha_trial
#             diff_alpha = abs(diff_alpha_trial)
#     return mu, alpha, Js

# # Find mu for specific alpha
# def mu_for_specific_alpha_BO(
#     target_alpha, mu_bounds, steps, n_ions, Omega, delta_k, verbosity=True
# ):
#     def cost(params):
#         parameters = params[0].tolist()
#         mu_trial = parameters[0]
#         alpha, _ = calculate_alpha(
#             n_ions, Omega, np.ones(1) * mu_trial, delta_k, verbosity=verbosity
#         )
#         alpha_diff = abs(alpha - target_alpha)
#         if verbosity:
#             print(f"distance to target alpha = {alpha_diff}")
#         return alpha_diff

#     bounds = [
#         {
#             "name": "mu",
#             "type": "continuous",
#             "domain": (mu_bounds[0][0], mu_bounds[1][0]),
#         }
#     ]
#     optimisation = GPyOpt.methods.BayesianOptimization(
#         cost,
#         domain=bounds,
#         model_type="GP",
#         acquisition_type="EI",
#         normalize_Y=True,
#         acquisition_weight=2,
#         maximize=False,
#     )

#     max_iter = steps
#     max_time = 180
#     optimisation.run_optimization(max_iter, max_time, verbosity=True)

#     mu = optimisation.x_opt
#     alpha, Js = calculate_alpha(n_ions, Omega, np.ones(1) * mu, delta_k)
#     return mu, alpha, Js
