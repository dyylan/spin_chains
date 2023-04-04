import autograd as ag
from autograd import numpy as agnp
import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import GPyOpt

from ..functions.fits import inverse_power_fit, yukawa_inverse_power_fit

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
        calculate_alpha=False,
    ):
        self.laser_stability_limit = 1000  # in Hz
        self.l = 1e-6  # length scale
        self.m = ion_mass  # mass of the ion
        self.n_ions = n_ions
        self.mu = mu
        self.Omega = (
            Omega if isinstance(Omega, list) else np.ones((n_ions, 1)) * Omega / n_ions
        )  # rabi frequency
        self.delta_k = (
            2 * 2 * np.pi / 355e-9 * np.array([1, 0, 0])
        )  # wave vector difference
        if use_optimal_omega:
            # z_trap = 0.75 * (2 * np.log(n_ions) * 6e6) / (3 * n_ions)
            self.omega = 2 * np.pi * np.array([6e6, 5e6, 1e6])
            self.optimise_z_trap_frequency(initial_z=0.8, z_step=0.005, verbosity=True)
        else:
            self.omega = omega  # x, y and z trap frequencies
            self.calculate_Js()
        if calculate_alpha:
            self.calculate_alpha()

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

    def calculate_minimum_mu(self, verbosity=True):
        eta = np.einsum(
            "ikm,k,m->im",
            self.eigenmodes,
            self.delta_k,
            np.sqrt(IonTrap.hbar / (2 * self.m * self.eigenfrequencies)),
        )
        delta = self.eigenfrequencies - self.omega.max()
        index = np.argmin(abs(delta))  # eigenfrequency of the highest trap mode
        mu_min_minus = abs(
            (3 * eta[0][index] * self.Omega[0]) - self.eigenfrequencies[index]
        )
        mu_min_plus = (3 * eta[0][index] * self.Omega[0]) + self.eigenfrequencies[index]
        mu_min = mu_min_minus if mu_min_minus > mu_min_plus else mu_min_plus
        if verbosity:
            print(f"Calculated minimum mu = {mu_min[0]}")
        return mu_min

    def single_mode_g(self):
        eta = np.einsum(
            "ikm,k,m->im",
            self.eigenmodes,
            self.delta_k,
            np.sqrt(IonTrap.hbar / (2 * self.m * self.eigenfrequencies)),
        )
        delta = self.eigenfrequencies - self.omega.max()
        index = np.argmin(abs(delta))  # index of single phonon mode
        self.g_single_mode = [
            -self.Omega[i][0] * eta[i][index] for i in range(self.n_ions)
        ]
        self.delta = self.mu[0] - self.eigenfrequencies[index]
        return self.g_single_mode, self.delta

    # Returns the alpha value from a specific mu
    def calculate_alpha(self, x=0, verbosity=True):
        if x:
            ions = [x - i for i in range(x)] + [
                i - x for i in range(x + 1, self.n_ions)
            ]
            Js = np.delete(self.Js[x], x)
        else:
            ions = [i for i in range(1, self.n_ions)]
            Js = self.Js[x][1:]

        popt, pcov = opt.curve_fit(
            inverse_power_fit, ions, Js, bounds=(0, [3e7, 2.0, 10.0])
        )
        if verbosity:
            print(
                f"Fit for Js with {self.n_ions} spins and mu = {self.mu[0]}: y = {popt[0]} r^-{popt[1]} + {popt[2]}"
                f"\n\talpha = {popt[1]}"
            )
        self.alpha = popt[1]
        return self.alpha, Js

    def distance_between_ions(self):
        z_positions = [position[2] for position in self.positions]
        central_distance = (
            z_positions[self.n_ions // 2] - z_positions[self.n_ions // 2 - 1]
        )
        end_distance = z_positions[self.n_ions - 1] - z_positions[self.n_ions - 2]
        return central_distance, end_distance

    def check_trap_is_linear(self, tolerance=1e5):
        is_linear = True
        max_z = abs(self.positions[0][2])
        for position in self.positions:
            x, y = position[0], position[1]
            if (abs(x) * tolerance > max_z) or (abs(y) * tolerance > max_z):
                is_linear = False
        return is_linear

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
            alpha, _ = self.calculate_alpha(
                x=0,
                verbosity=verbosity,
            )
            alpha_diff = abs(alpha - target_alpha)
            if verbosity:
                print(f"\tdistance to target alpha = {alpha_diff}")
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
        max_time = 300
        optimisation.run_optimization(max_iter, max_time, verbosity=verbosity)

        mu = optimisation.x_opt
        self.update_mu(np.ones(1) * mu)
        self.alpha, Js = self.calculate_alpha(x=0, verbosity=verbosity)
        return mu, self.alpha, Js

    def optimise_z_trap_frequency(self, initial_z=0.85, z_step=0.01, verbosity=True):
        z_factor = (2 * np.log(self.n_ions) * 6e6) / (3 * self.n_ions)
        z = initial_z
        self.omega = np.array([self.omega[0], self.omega[1], 2 * np.pi * z * z_factor])
        self.calculate_Js()
        if np.isnan(self.Js).any():
            raise ValueError("Initial z trap frequency is too high.")
        while (not np.isnan(self.Js).any()) & self.check_trap_is_linear():
            z += z_step
            self.omega = self.omega = np.array(
                [self.omega[0], self.omega[1], 2 * np.pi * z * z_factor]
            )
            self.calculate_Js()
        z -= z_step
        self.omega = self.omega = np.array(
            [self.omega[0], self.omega[1], 2 * np.pi * z * z_factor]
        )
        self.calculate_Js()
        if verbosity:
            print(f"Calculated z trap frequency with z prefactor = {z}")
        return self.omega

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
                    np.linspace(
                        -(self.n_ions - 1) / 2, (self.n_ions - 1) / 2, self.n_ions
                    ).reshape(self.n_ions, 1),
                    pad_width=[[0, 0], [2, 0]],
                ).flatten(),
                args=(self.n_ions, self.m, self.omega, self.l),
                method="SLSQP",
                options={"maxiter": 1000},
                tol=1e-15,
            ).x.reshape(self.n_ions, 3)
            * self.l
        )
        return x_ep

    # hessian evaluated at equilibrium position
    def _calculate_equilibrium_hessian(self):
        return IonTrap.hess_expr_tot(
            self.positions.flatten() / self.l, self.n_ions, self.m, self.omega, self.l
        )

    # normal mode eigenpairs
    def _calculate_normal_mode_eigenpairs(self):
        lamb, b = np.linalg.eigh(self.hessian)
        eigenfrequencies = np.sqrt(
            lamb * IonTrap.k * IonTrap.e ** 2 / (self.l ** 3 * self.m)
        )  # eigenfrequencies
        eigenmodes = b.reshape(self.n_ions, 3, 3 * self.n_ions)  # eigenvectors
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
            1 - np.identity(self.n_ions),
            zeta,
            zeta,
            self.eigenfrequencies,
            1 / np.subtract.outer(self.mu ** 2, self.eigenfrequencies ** 2),
        )
        return Js

    def plot_Js(self):
        plt.imshow(self.Js)
        plt.show()

    def plot_ion_chain_positions(self, savefig=False):
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
        ax.set_xlabel("Axial $z~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        ax.set_ylabel("$x~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        ax.set_zlabel("$y~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        ax.tick_params(axis="both", labelsize=fontsize_ticks - 6)
        if savefig:
            plt.savefig(
                f"plots/experimental/chain_positions/plot_ion_chain_positions_n={self.n_ions}_mu={self.mu[0]:.2f}.pdf"
            )
        plt.show()

    def plot_spin_interactions(
        self, x=0, exp_fit=True, savefig=False, include_ion_chain=False, plot_label=None
    ):
        if x:
            ions = [x - i for i in range(x)] + [
                i - x for i in range(x + 1, self.n_ions)
            ]
            Js = np.delete(self.Js[x], x)
        else:
            ions = [i for i in range(1, self.n_ions)]
            Js = self.Js[x][1:]

        popt, pcov = opt.curve_fit(
            inverse_power_fit, ions, Js, bounds=(0, [3e7, 2.0, 10.0])
        )
        print(
            f"Fit for Js with {self.n_ions} spins and mu = {self.mu}: y = {popt[0]} x^-{popt[1]} + {popt[2]}"
        )
        if plot_label:
            fig, ax = plt.subplots(figsize=[10, 10])
        else:
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
                bounds=(0, [3e7, 1.0, 2.0, 1.0]),
            )
            print(
                f"Fit for Js with {self.n_ions} spins and mu = {self.mu}: y = {popt[0]} exp(-{popt[1]} x) x^-{popt[2]} + {popt[3]}"
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
        ax.set_ylabel(ylabel="Coupling strength $J_{1 j}$ (Hz)", fontsize=fontsize_axis)

        ax.grid()
        # ax.set(xscale="log")
        # ax.set(yscale="log")
        plt.xticks(fontsize=fontsize_ticks)
        plt.yticks(fontsize=fontsize_ticks)

        if plot_label:
            ax.text(
                0.025, 0.90, plot_label, fontsize=22, transform=plt.gcf().transFigure
            )

        if include_ion_chain:
            ax2 = fig.add_axes([0.48, 0.38, 0.45, 0.42], projection="3d")
            max_position = 0
            l = 1e6
            for position in self.positions:
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

        plt.tight_layout()
        if savefig:
            plt.savefig(
                f"plots/experimental/spin_interactions/plot_ion_chain_positions_n={self.n_ions}_mu={self.mu[0]:.2f}_omega_z={self.omega[2]:.2f}.pdf"
            )
        plt.show()


class IonTrapXY:

    k = 8.98755179e9  # coulomb constant
    hbar = 1.054571817e-34  # reduced planck constant
    e = 1.602176634e-19  # charge of ion
    omega_0 = 12.642821e9  # qubit frequency

    def __init__(
        self,
        n_ions,
        mu=np.ones(1) * 2 * np.pi * 6.01e6,
        omega=2 * np.pi * np.array([6e6, 5e6, 0.5e6]),
        Omega=2 * np.pi * 1e6,
        ion_mass=171 * 1.660539066e-27,
        use_optimal_omega=False,
        calculate_alpha=False,
    ):
        self.laser_stability_limit = 1000  # in Hz
        self.l = 1e-6  # length scale
        self.m = ion_mass  # mass of the ion
        self.n_ions = n_ions
        self.mu = mu
        self.Omega = (
            Omega if isinstance(Omega, list) else np.ones(n_ions) * Omega / n_ions
        )  # rabi frequency
        self.delta_k = (
            2 * 2 * np.pi / 355e-9 * np.array([1, 0, 0])
        )  # wave vector difference
        if use_optimal_omega:
            # z_trap = 0.75 * (2 * np.log(n_ions) * 6e6) / (3 * n_ions)
            self.omega = 2 * np.pi * np.array([6e6, 5e6, 1e6])
            self.optimise_z_trap_frequency(initial_z=0.8, z_step=0.005, verbosity=True)
        else:
            self.omega = omega  # x, y and z trap frequencies
            self.calculate_Js()
        if calculate_alpha:
            self.calculate_alpha()

    def calculate_Js(self):
        self.positions = self._calculate_equilibrium_positions()
        self.hessian = self._calculate_equilibrium_hessian()
        (
            self.eigenfrequencies,
            self.eigenmodes,
        ) = self._calculate_normal_mode_eigenpairs()
        self.single_mode_ms()
        self.Js = self._spin_interaction_graph()
        # print("Calculated Js")
        return self.Js

    # def calculate_minimum_mu(self, verbosity=True):
    #     eta = np.einsum(
    #         "ikm,k,m->im",
    #         self.eigenmodes,
    #         self.delta_k,
    #         np.sqrt(IonTrapXY.hbar / (2 * self.m * self.eigenfrequencies)),
    #     )
    #     delta = self.eigenfrequencies - self.omega.max()
    #     index = np.argmin(abs(delta))  # eigenfrequency of the highest trap mode
    #     mu_min_minus = abs(
    #         (3 * eta[0][index] * self.Omega[0]) - self.eigenfrequencies[index]
    #     )
    #     mu_min_plus = (3 * eta[0][index] * self.Omega[0]) + self.eigenfrequencies[index]
    #     mu_min = mu_min_minus if mu_min_minus > mu_min_plus else mu_min_plus
    #     mu_min = np.ones(1) * mu_min
    #     if verbosity:
    #         print(f"Calculated minimum mu = {mu_min[0]}")
    #     return mu_min

    def calculate_minimum_mu(self, verbosity=True):
        eta = np.einsum(
            "ikm,k,m->im",
            self.eigenmodes,
            self.delta_k,
            np.sqrt(IonTrapXY.hbar / (2 * self.m * self.eigenfrequencies)),
        )
        delta = self.eigenfrequencies - self.omega.max()
        index = np.argmin(abs(delta))  # eigenfrequency of the highest trap mode
        mu_min_minus = np.sqrt(
            ((3 * eta[0][index] * self.Omega[0]) - self.eigenfrequencies[index]) ** 2
            - (self.Omega[0] ** 2)
        )
        mu_min_plus = np.sqrt(
            ((3 * eta[0][index] * self.Omega[0]) + self.eigenfrequencies[index]) ** 2
            - (self.Omega[0] ** 2)
        )
        mu_min = mu_min_minus if mu_min_minus > mu_min_plus else mu_min_plus
        mu_min = np.ones(1) * mu_min
        if verbosity:
            print(f"Calculated minimum mu = {mu_min[0]}")
        return mu_min

    def single_mode_g(self):
        eta = np.einsum(
            "ikm,k,m->im",
            self.eigenmodes,
            self.delta_k,
            np.sqrt(IonTrapXY.hbar / (2 * self.m * self.eigenfrequencies)),
        )
        delta = self.eigenfrequencies - self.omega.max()
        index = np.argmin(abs(delta))  # index of single phonon mode
        self.g_single_mode = [
            self.Omega[i][0] * eta[i][index] for i in range(self.n_ions)
        ]
        self.delta = self.mu[0] - self.eigenfrequencies[index]
        return self.g_single_mode, self.delta

    def single_mode_ms(self):
        eta = np.einsum(
            "ikm,k,m->im",
            self.eigenmodes,
            self.delta_k,
            np.sqrt(IonTrapXY.hbar / (2 * self.m * self.eigenfrequencies)),
        )
        delta_vec = self.eigenfrequencies - self.omega.max()
        index = np.argmin(abs(delta_vec))  # index of single phonon mode
        self.g_single_mode = [self.Omega[i] * eta[i][index] for i in range(self.n_ions)]
        self.omega_single_mode = self.eigenfrequencies[index]
        delta = self.mu[0] - self.eigenfrequencies[index]
        # self.omega_eff = np.sqrt(
        #     ((IonTrapXY.omega_0 + self.omega_single_mode + delta) ** 2)
        #     + (self.Omega[0] ** 2)
        # )
        # self.omega_eff = np.sqrt(
        #     ((self.omega_single_mode + delta) ** 2) + (self.Omega[0] ** 2)
        # )
        self.omega_eff = np.sqrt((self.mu[0] ** 2) + (self.Omega[0] ** 2))
        return self.g_single_mode, self.omega_single_mode, self.omega_eff, delta

    def two_mode_ms(self):
        eta = np.einsum(
            "ikm,k,m->im",
            self.eigenmodes,
            self.delta_k,
            np.sqrt(IonTrapXY.hbar / (2 * self.m * self.eigenfrequencies)),
        )
        delta_vec = self.eigenfrequencies - self.omega.max()
        # max_ = np.min(abs(delta_vec))
        if self.n_ions == 10:
            indices = np.array([27, 25])
        else:
            indices = np.argpartition(abs(delta_vec), 2)[
                :2
            ]  # index of both phonon modes
        self.g_single_mode = [
            self.Omega[i] * eta[i][indices[0]] for i in range(self.n_ions)
        ]
        self.omega_single_mode = self.eigenfrequencies[indices[0]]
        self.g_modes = list(
            zip(
                self.g_single_mode,
                [self.Omega[i] * eta[i][indices[1]] for i in range(self.n_ions)],
            )
        )
        self.omega_modes = [
            self.omega_single_mode,
            self.eigenfrequencies[indices[1]],
        ]
        deltas = [
            self.mu[0] - self.eigenfrequencies[indices[0]],
            self.mu[0] - self.eigenfrequencies[indices[1]],
        ]
        self.omega_eff = np.sqrt((self.mu[0] ** 2) + (self.Omega[0] ** 2))
        return self.g_modes, self.omega_modes, self.omega_eff, deltas

    def calculate_t_0(self):
        self.single_mode_ms()
        self.t_0 = 2 * np.pi / (self.omega_eff - self.omega_single_mode)
        return self.t_0

    # Returns the alpha value from a specific mu
    def calculate_alpha(self, x=0, all_interactions=False, verbosity=True):
        if all_interactions:
            ions = []
            Js = []
            for ion in range(self.n_ions):
                ions.extend(
                    [ion - i for i in range(ion)]
                    + [i - ion for i in range(ion + 1, self.n_ions)]
                )
                Js.extend(np.delete(self.Js[ion], ion))
        else:
            if x:
                ions = [x - i for i in range(x)] + [
                    i - x for i in range(x + 1, self.n_ions)
                ]
                Js = np.delete(self.Js[x], x)
            else:
                ions = [i for i in range(1, self.n_ions)]
                Js = self.Js[x][1:]

        popt, pcov = opt.curve_fit(
            inverse_power_fit, ions, Js, bounds=(0, [3e7, 2.0, 10.0])
        )
        if verbosity:
            print(
                f"Fit for Js with {self.n_ions} spins and mu = {self.mu[0]}: y = {popt[0]} r^-{popt[1]} + {popt[2]}"
                f"\n\talpha = {popt[1]}"
            )
        self.alpha = popt[1]
        return self.alpha, Js

    def distance_between_ions(self):
        z_positions = [position[2] for position in self.positions]
        central_distance = (
            z_positions[self.n_ions // 2] - z_positions[self.n_ions // 2 - 1]
        )
        end_distance = z_positions[self.n_ions - 1] - z_positions[self.n_ions - 2]
        return central_distance, end_distance

    def check_trap_is_linear(self, tolerance=1e5):
        is_linear = True
        max_z = abs(self.positions[0][2])
        for position in self.positions:
            x, y = position[0], position[1]
            if (abs(x) * tolerance > max_z) or (abs(y) * tolerance > max_z):
                is_linear = False
        return is_linear

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
            alpha, _ = self.calculate_alpha(
                x=0,
                verbosity=verbosity,
            )
            alpha_diff = abs(alpha - target_alpha)
            if verbosity:
                print(f"\tdistance to target alpha = {alpha_diff}")
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
        max_time = 300
        optimisation.run_optimization(max_iter, max_time, verbosity=verbosity)

        mu = optimisation.x_opt
        self.update_mu(np.ones(1) * mu)
        self.alpha, Js = self.calculate_alpha(x=0, verbosity=verbosity)
        return mu, self.alpha, Js

    def optimise_z_trap_frequency(self, initial_z=0.85, z_step=0.01, verbosity=True):
        z_factor = (2 * np.log(self.n_ions) * 6e6) / (3 * self.n_ions)
        z = initial_z
        self.omega = np.array([self.omega[0], self.omega[1], 2 * np.pi * z * z_factor])
        self.calculate_Js()
        if np.isnan(self.Js).any():
            raise ValueError("Initial z trap frequency is too high.")
        while (not np.isnan(self.Js).any()) & self.check_trap_is_linear():
            z += z_step
            self.omega = self.omega = np.array(
                [self.omega[0], self.omega[1], 2 * np.pi * z * z_factor]
            )
            self.calculate_Js()
        z -= z_step
        self.omega = self.omega = np.array(
            [self.omega[0], self.omega[1], 2 * np.pi * z * z_factor]
        )
        self.calculate_Js()
        if verbosity:
            print(f"Calculated z trap frequency with z prefactor = {z}")
        return self.omega

    def nondimensionalize_decorator(expr):
        def _expr(x, n, m, omega, l):
            return expr(x * l, n, m, omega, l) * l / (IonTrapXY.k * IonTrapXY.e ** 2)

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
            IonTrapXY.k
            * IonTrapXY.e ** 2
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
        return IonTrapXY.expr_coul(x, n, m, omega, l) + IonTrapXY.expr_har(
            x, n, m, omega, l
        )

    # hessian of flattened nondimensionalized total potential
    @staticmethod
    def hess_expr_tot(x, n, m, omega, l):
        return ag.hessian(IonTrapXY.expr_tot, 0)(x, n, m, omega, l)

    # equilibrium position
    def _calculate_equilibrium_positions(self):
        x_ep = (
            opt.minimize(
                IonTrapXY.expr_tot,
                np.pad(
                    np.linspace(
                        -(self.n_ions - 1) / 2, (self.n_ions - 1) / 2, self.n_ions
                    ).reshape(self.n_ions, 1),
                    pad_width=[[0, 0], [2, 0]],
                ).flatten(),
                args=(self.n_ions, self.m, self.omega, self.l),
                method="SLSQP",
                options={"maxiter": 1000},
                tol=1e-15,
            ).x.reshape(self.n_ions, 3)
            * self.l
        )
        return x_ep

    # hessian evaluated at equilibrium position
    def _calculate_equilibrium_hessian(self):
        return IonTrapXY.hess_expr_tot(
            self.positions.flatten() / self.l, self.n_ions, self.m, self.omega, self.l
        )

    # normal mode eigenpairs
    def _calculate_normal_mode_eigenpairs(self):
        lamb, b = np.linalg.eigh(self.hessian)
        eigenfrequencies = np.sqrt(
            lamb * IonTrapXY.k * IonTrapXY.e ** 2 / (self.l ** 3 * self.m)
        )  # eigenfrequencies
        eigenmodes = b.reshape(self.n_ions, 3, 3 * self.n_ions)  # eigenvectors
        return eigenfrequencies, eigenmodes

    # Calculates spin interaction graph
    def _spin_interaction_graph(self):
        eta = np.einsum(
            "ikm,k,m->im",
            self.eigenmodes,
            self.delta_k,
            np.sqrt(IonTrapXY.hbar / (2 * self.m * self.eigenfrequencies)),
        )
        # Omega = self.Omega[0]
        # Js = np.einsum(
        #     "ij,i,j,im,jm,m,m->ij",
        #     1 - np.identity(self.n_ions),
        #     self.Omega,
        #     self.Omega,
        #     eta,
        #     eta,
        #     self.mu - self.eigenfrequencies,
        #     1 / (8 * (((self.mu - self.eigenfrequencies) ** 2) - (self.B ** 2))),
        # )
        Js = np.einsum(
            "ij,i,j,im,jm,m,m->ij",
            1 - np.identity(self.n_ions),
            self.Omega,
            self.Omega,
            eta,
            eta,
            self.eigenfrequencies,
            1 / (8 * ((self.omega_eff ** 2) - (self.eigenfrequencies ** 2))),
        )
        # Js = np.einsum(
        #     "ij,i,j,im,jm,m->ij",
        #     1 - np.identity(self.n_ions),
        #     self.Omega,
        #     self.Omega,
        #     eta,
        #     eta,
        #     1 / (8 * (self.omega_eff - self.eigenfrequencies)),
        # )
        return Js

    def plot_Js(self):
        plt.imshow(self.Js)
        plt.show()

    def plot_ion_chain_positions(self, savefig=False):
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
        ax.set_xlabel("Axial $z~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        ax.set_ylabel("$x~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        ax.set_zlabel("$y~(\\mathrm{\\mu m})$", fontsize=fontsize_axis)
        ax.tick_params(axis="both", labelsize=fontsize_ticks - 6)
        if savefig:
            plt.savefig(
                f"plots/experimental/chain_positions/plot_ion_chain_positions_n={self.n_ions}_mu={self.mu[0]:.2f}.pdf"
            )
        plt.show()

    def plot_spin_interactions(
        self,
        x=0,
        exp_fit=True,
        savefig=False,
        include_ion_chain=False,
        all_interactions=False,
        use_z_position=False,
        fit_end_and_mid=False,
        plot_label=None,
    ):
        def compute_ions(ion):
            if use_z_position:
                ions = [
                    abs(self.positions[ion][2] - self.positions[i][2])
                    for i in range(ion)
                ] + [
                    abs(self.positions[i][2] - self.positions[ion][2])
                    for i in range(ion + 1, self.n_ions)
                ]
            else:
                ions = [ion - i for i in range(ion)] + [
                    i - ion for i in range(ion + 1, self.n_ions)
                ]
            return ions

        if all_interactions:
            ions = []
            Js = []
            for ion in range(self.n_ions):
                ions.extend(compute_ions(ion))
                Js.extend(np.delete(self.Js[ion], ion))
        else:
            if x:
                ions = compute_ions(x)
                Js = np.delete(self.Js[x], x)
            else:
                ions = [i for i in range(1, self.n_ions)]
                Js = self.Js[x][1:]
        if fit_end_and_mid:
            ions_end = compute_ions(0)
            Js_end = self.Js[0][1:]

            ion_mid = self.n_ions // 2
            ions_mid = compute_ions(ion_mid)
            Js_mid = np.delete(self.Js[ion_mid], ion_mid)
            if use_z_position:
                popt_end, pcov = opt.curve_fit(
                    inverse_power_fit, ions_end, Js_end, bounds=(0, [100, 2.0, 10.0])
                )
                popt_mid, pcov = opt.curve_fit(
                    inverse_power_fit, ions_mid, Js_mid, bounds=(0, [100, 2.0, 10.0])
                )
            else:
                popt_end, pcov = opt.curve_fit(
                    inverse_power_fit, ions_end, Js_end, bounds=(0, [3e7, 2.0, 10.0])
                )
                popt_mid, pcov = opt.curve_fit(
                    inverse_power_fit, ions_mid, Js_mid, bounds=(0, [3e7, 2.0, 10.0])
                )
            print(
                f"Fit for end Js with {self.n_ions} spins and mu = {self.mu}: y = {popt_end[0]} x^-{popt_end[1]} + {popt_end[2]}"
                f"\nFit for mid Js with {self.n_ions} spins and mu = {self.mu}: y = {popt_mid[0]} x^-{popt_mid[1]} + {popt_mid[2]}"
            )
        if use_z_position:
            popt, pcov = opt.curve_fit(
                inverse_power_fit, ions, Js, bounds=(0, [100, 2.0, 10.0])
            )
        else:
            popt, pcov = opt.curve_fit(
                inverse_power_fit, ions, Js, bounds=(0, [3e7, 2.0, 10.0])
            )
        print(
            f"Fit for Js with {self.n_ions} spins and mu = {self.mu}: y = {popt[0]} x^-{popt[1]} + {popt[2]}"
        )
        if plot_label:
            fig, ax = plt.subplots(figsize=[8, 8])
        else:
            fig, ax = plt.subplots(figsize=figure_size)

        linestyles = ["", "dashed", "dashed", "dashed", "dashed"]
        markers = ["d", "", "", "", "", ""]
        colours = ["b", "r", "g", "purple", "teal"]

        ions_list = (
            np.linspace(0, max(ions), 50)
            if use_z_position
            else list(range(1, self.n_ions))
        )

        alpha_fit = inverse_power_fit(ions_list, popt[0], popt[1], 0)
        ys = [alpha_fit]
        ax.plot(ions, Js, linestyle=linestyles[0], color=colours[0], marker=markers[0])
        # legend = [f"Numerical $J_{{1 j}}$", f"${popt[0]:.2f} r^{{-{popt[1]:.3f}}}$"]
        legend = [
            f"Numerical $J_{{{'i j' if all_interactions else '1 j'}}}$",
            f"$\propto r^{{-{popt[1]:.3f}}}$",
        ]

        if exp_fit:
            popt, pcov = opt.curve_fit(
                yukawa_inverse_power_fit,
                ions,
                Js,
                bounds=(0, [3e7, 1.0, 2.0, 1.0]),
            )
            print(
                f"Fit for Js with {self.n_ions} spins and mu = {self.mu}: y = {popt[0]} exp(-{popt[1]} x) x^-{popt[2]} + {popt[3]}"
            )
            alpha_yukawa_fit = yukawa_inverse_power_fit(
                ions_list, popt[0], popt[1], popt[2], popt[3]
            )
            ys.append(alpha_yukawa_fit)
            # legend.append(f"${popt[0]:.2f} e^{{-{popt[1]:.3f} r}} r^{{-{popt[2]:.3f}}}$")
            legend.append(f"$\propto e^{{-{popt[1]:.3f} r}} r^{{-{popt[2]:.3f}}}$")
        if fit_end_and_mid:
            alpha_mid_fit = inverse_power_fit(ions_list, popt_mid[0], popt_mid[1], 0)
            alpha_end_fit = inverse_power_fit(ions_list, popt_end[0], popt_end[1], 0)
            ys.append(alpha_mid_fit)
            ys.append(alpha_end_fit)
            legend.append(f"$\propto r^{{-{popt_mid[1]:.3f}}}$")
            legend.append(f"$\propto r^{{-{popt_end[1]:.3f}}}$")

        for i, y in enumerate(ys):
            ax.plot(
                ions_list,
                y,
                linestyle=linestyles[i + 1],
                color=colours[i + 1],
                marker=markers[i + 1],
            )

        ax.legend(legend, fontsize=fontsize_legend + 6)

        if all_interactions:
            if use_z_position:
                ax.set_xlabel(xlabel="$| z_j - z_i |$ (m)", fontsize=fontsize_axis + 4)
            else:
                ax.set_xlabel(xlabel="$r = j - i$", fontsize=fontsize_axis + 4)
            ax.set_ylabel(
                ylabel="Coupling strength $J_{i j}$ (Hz)", fontsize=fontsize_axis + 4
            )
        else:
            if use_z_position:
                ax.set_xlabel(xlabel="$| z_j - z_1 |$ (m)", fontsize=fontsize_axis + 4)
            else:
                ax.set_xlabel(xlabel="$r = j - 1$", fontsize=fontsize_axis + 4)
            ax.set_ylabel(
                ylabel="Coupling strength $J_{1 j}$ (Hz)", fontsize=fontsize_axis + 4
            )

        ax.grid()
        # ax.set(xscale="log")
        # ax.set(yscale="log")
        plt.xticks(fontsize=fontsize_ticks + 4)
        plt.yticks(fontsize=fontsize_ticks + 4)

        if plot_label:
            ax.text(
                0.025, 0.90, plot_label, fontsize=26, transform=plt.gcf().transFigure
            )

        if include_ion_chain:
            ax2 = fig.add_axes([0.45, 0.275, 0.43, 0.40], projection="3d")
            max_position = 0
            l = 1e6
            for position in self.positions:
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
            ax2.tick_params(axis="both", labelsize=fontsize_ticks - 6)

        plt.tight_layout()
        if savefig:
            z_position_tag = "_z_positions" if use_z_position else ""
            plt.savefig(
                f"plots/experimental/spin_interactions/plot_ion_chain_positions_n={self.n_ions}_mu={self.mu[0]:.2f}_omega_z={self.omega[2]:.2f}{z_position_tag}.pdf"
            )
        plt.show()
