import numpy as np
import GPyOpt


def optimise_gamma_time_AO(cost, bounds, steps, verbosity=True):
    def cost(params):
        parameters = params[0].tolist()
        gamma_trial = parameters[0]
        self.update_mu(np.ones(1) * mu_trial)
        alpha, _ = self.calculate_alpha(
            x=0,
            verbosity=verbosity,
        )
        alpha_diff = abs(alpha - target_alpha)
        if verbosity:
            print(f"distance to target alpha = {alpha_diff}")
        return alpha_diff

    bounds = [
        {
            "name": "gamma",
            "type": "continuous",
            "domain": (gamma_bounds[0], gamma_bounds[1]),
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
    self.alpha, Js = self.calculate_alpha(x=0, verbosity=verbosity)
    return mu, self.alpha, Js
