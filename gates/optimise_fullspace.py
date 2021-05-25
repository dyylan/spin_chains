from chains import Chain1d
from states import RandomState, SingleState
import numpy as np

np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import GPyOpt

spins = 3
n = 2 ** spins
state_samples = 100
U = (1 / np.sqrt(n)) * np.array(
    [[np.exp(1j * 2 * np.pi * x * k / n) for x in range(n)] for k in range(n)]
)


def cost(params):
    params = params[0].tolist()
    jx = params[:spins]
    jy = params[spins : 2 * spins]
    jz = params[2 * spins : 3 * spins]
    hx = params[3 * spins : 4 * spins]
    hy = params[4 * spins : 5 * spins]
    hz = params[5 * spins : 6 * spins]
    chain = Chain1d(spins=spins, dt=1, jx=jx, jy=jy, jz=jz)
    for spin in range(spins):
        chain.add_marked_site(spin + 1, hx[spin], "x")
        chain.add_marked_site(spin + 1, hy[spin], "y")
        chain.add_marked_site(spin + 1, hz[spin], "z")
    chain.hamiltonian.initialise_evolution(subspace_evolution=False)
    # cost = np.linalg.norm(U - chain.hamiltonian.U)
    fidelity = calculate_fidelity(chain.hamiltonian.U)
    return fidelity


def calculate_fidelity(chain_U):
    random_states = [RandomState(spins, False) for _ in range(state_samples)]
    # random_states = [SingleState(spins, j+1) for j in range(2**spins)]
    fidelities = []
    for state in random_states:
        chain_state = np.matmul(chain_U, state.ket)
        qft_state = np.matmul(U, state.ket)
        overlap = np.vdot(qft_state, chain_state)
        fidel = np.multiply(np.conj(overlap), overlap)
        fidelities.append(np.abs(fidel))
    fidelity = np.mean(fidelities)
    print(f"fidelity = {fidelity}")
    return fidelity


if __name__ == "__main__":
    rest = {"type": "continuous", "domain": (-20, 20)}

    bounds = [{"name": f"jx_{jx+1}", **rest} for jx in range(spins)]
    bounds += [{"name": f"jy_{jy+1}", **rest} for jy in range(spins)]
    bounds += [{"name": f"jz_{jz+1}", **rest} for jz in range(spins)]
    bounds += [{"name": f"hx_{hx+1}", **rest} for hx in range(spins)]
    bounds += [{"name": f"hy_{hy+1}", **rest} for hy in range(spins)]
    bounds += [{"name": f"hz_{hz+1}", **rest} for hz in range(spins)]

    optimisation = GPyOpt.methods.BayesianOptimization(
        cost,
        domain=bounds,
        model_type="GP",
        acquisition_type="EI",
        normalize_Y=True,
        acquisition_weight=2,
        maximize=True,
    )

    max_iter = 200  # maximum time 40 iterations
    max_time = 600  # maximum time 60 seconds

    optimisation.run_optimization(max_iter, max_time, verbosity=True)

    jx_print = "\tjx = ["
    jy_print = "\tjy = ["
    jz_print = "\tjz = ["
    hx_print = "\thx = ["
    hy_print = "\thy = ["
    hz_print = "\thz = ["
    for j in range(spins):
        jx_print += f"{optimisation.x_opt[j]},"
        jy_print += f"{optimisation.x_opt[spins+j]},"
        jz_print += f"{optimisation.x_opt[2*spins+j]},"
        hx_print += f"{optimisation.x_opt[3*spins+j]},"
        hy_print += f"{optimisation.x_opt[4*spins+j]},"
        hz_print += f"{optimisation.x_opt[5*spins+j]},"
    jx_print = jx_print[:-1] + "]"
    jy_print = jy_print[:-1] + "]"
    jz_print = jz_print[:-1] + "]"
    hx_print = hx_print[:-1] + "]"
    hy_print = hy_print[:-1] + "]"
    hz_print = hz_print[:-1] + "]"

    print(
        f"Parameters that optimise fidelity:\n"
        f"{jx_print}\n"
        f"{jy_print}\n"
        f"{jz_print}\n"
        f"{hx_print}\n"
        f"{hy_print}\n"
        f"{hz_print}\n"
    )
    print(f"Optimum fidelity: {-optimisation.fx_opt}")

    optimisation.plot_convergence()
