from chains import Chain1dSubspace2particles
from states import TwoParticleHubbbardState
import numpy as np
from scipy.signal import find_peaks

np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import GPyOpt

spins = 16
state_samples = 100


def cost(params):
    # es_engineered = 0
    es_engineered = params[0].tolist()

    js_engineered = [
        1 * (np.sqrt((spin + 1) * (spins - spin - 1)) / (spins / 2))
        for spin in range(spins)
    ]

    us = 20
    vs = 10

    chain = Chain1dSubspace2particles(
        spins=spins,
        dt=0.01,
        js=js_engineered,
        us=us,
        es=es_engineered,
        vs=vs,
        open_chain=True,
    )
    fidelity = calculate_fidelity(chain, 5000)
    return fidelity


def calculate_fidelity(chain, time):
    init_state = TwoParticleHubbbardState(spins, sites=[1, 1])
    target_state = TwoParticleHubbbardState(spins, sites=[spins, spins])
    chain.initialise(init_state, subspace_evolution=True)
    times, states = chain.time_evolution(time=time)
    overlaps = chain.overlaps_evolution(target_state.subspace_ket, states)
    peaks, _ = find_peaks(overlaps, height=(0, 1.05))
    fidelity = overlaps[peaks[0]]
    print(f"fidelity = {fidelity}")
    return fidelity


if __name__ == "__main__":
    rest = {"type": "continuous", "domain": (-20, 20)}

    bounds = [{"name": f"e_{e+1}", **rest} for e in range(spins)]

    optimisation = GPyOpt.methods.BayesianOptimization(
        cost,
        domain=bounds,
        model_type="GP",
        acquisition_type="EI",
        normalize_Y=True,
        acquisition_weight=2,
        maximize=True,
    )

    max_iter = 400  # maximum iterations
    max_time = 1200  # maximum time seconds

    optimisation.run_optimization(max_iter, max_time, verbosity=True)

    es_print = f"epsilons = {optimisation.x_opt}"

    print(f"epsilons = {optimisation.x_opt}")
    print(f"Optimum fidelity: {-optimisation.fx_opt}")

    optimisation.plot_convergence()
