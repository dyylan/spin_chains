from chains import Chain1d
from states import RandomState, SingleState
import numpy as np

np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import GPyOpt

spins = 3
n = 2 ** spins
state_samples = 1000
# U = (1/np.sqrt(n))*np.array([[np.exp(1j*2*np.pi*x*k/n) for x in range(n)]
# for k in range(n)])
U = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ]
)


def cost(params):
    params = params[0].tolist()
    couplings = couplings_map(params)
    jx = couplings["jx"]
    jy = couplings["jy"]
    jz = couplings["jz"]
    hx = couplings["hx"]
    hy = couplings["hy"]
    hz = couplings["hz"]
    chain = Chain1d(spins=spins, dt=1, jx=jx, jy=jy, jz=jz)
    for spin in range(spins):
        chain.add_marked_site(spin + 1, hx[spin], "x")
        chain.add_marked_site(spin + 1, hz[spin], "z")
    chain.hamiltonian.initialise_evolution(subspace_evolution=False)
    # cost = np.linalg.norm(U - chain.hamiltonian.U)
    fidelity = calculate_fidelity(chain.hamiltonian.U)
    return fidelity


def couplings_map(params):
    couplings = {
        "jx": [params[3], params[4], params[5]],
        "jy": [params[3], 0, 0],
        "jz": [params[0], params[1], params[2]],
        "hx": [params[5], params[4], params[8]],
        "hy": [0, 0, 0],
        "hz": [params[6], params[7], params[1] + params[2]],
    }
    return couplings


def calculate_fidelity(chain_U):
    random_states = [RandomState(spins, False) for _ in range(state_samples)]
    # random_states = [SingleState(spins, j+1) for j in range(2**spins)]
    fidelities = []
    for state in random_states:
        chain_state = np.matmul(chain_U, state.ket)
        toffoli_state = np.matmul(U, state.ket)
        overlap = np.vdot(toffoli_state, chain_state)
        fidel = np.multiply(np.conj(overlap), overlap)
        fidelities.append(np.abs(fidel))
    fidelity = np.mean(fidelities)
    print(f"fidelity = {fidelity}")
    return fidelity


if __name__ == "__main__":
    rest = {"type": "continuous", "domain": (-70, 70)}

    bounds = [
        {"name": f"jzz_12", **rest},
        {"name": f"jzz_23", **rest},
        {"name": f"jzz_13", **rest},
        {"name": f"jyy_12", **rest},
        {"name": f"jxx_23", **rest},
        {"name": f"jxx_13", **rest},
        {"name": f"hz_1", **rest},
        {"name": f"hz_2", **rest},
        {"name": f"hx_3", **rest},
    ]

    optimisation = GPyOpt.methods.BayesianOptimization(
        cost,
        domain=bounds,
        model_type="GP",
        acquisition_type="MPI",
        normalize_Y=True,
        acquisition_optimizer_type="lbfgs",
        Initial_design_numdata=50,
        maximize=True,
    )

    max_iter = 2000  # maximum time 40 iterations
    max_time = 7200  # maximum time 60 seconds

    optimisation.run_optimization(max_iter, max_time, verbosity=True)

    print(f"Parameters that optimise fidelity:\n" f"{optimisation.x_opt}")
    print(f"Optimum fidelity: {-optimisation.fx_opt}")

    optimisation.plot_convergence()
