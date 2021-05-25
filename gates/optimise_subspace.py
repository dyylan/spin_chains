from chains import Chain1dSubspace
from states import RandomState
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import GPyOpt

spins = 16
state_samples = 100
U = (1/np.sqrt(spins))*np.array([[np.exp(1j*2*np.pi*x*k/spins) for x in range(spins)] 
                                                                for k in range(spins)])

def cost(params):
    params = params[0].tolist()
    js = params[:spins]
    hs = params[spins:]
    chain = Chain1dSubspace(spins=spins, dt=1, js=js)
    for spin, h in enumerate(hs):
        chain.add_marked_site(spin+1, h)
    chain.hamiltonian.initialise_evolution()
    # cost = np.linalg.norm(U - chain.hamiltonian.U)
    fidelity = calculate_fidelity(chain.hamiltonian.U)
    return fidelity

def calculate_fidelity(chain_U):
    random_states = [RandomState(spins, True) for _ in range(state_samples)]
    fidelities = []
    for state in random_states:
        chain_state = np.matmul(chain_U, state.subspace_ket)
        qft_state = np.matmul(U, state.subspace_ket)
        overlap = np.vdot(qft_state, chain_state)
        fidel = np.multiply(np.conj(overlap), overlap)
        fidelities.append(np.abs(fidel))
    fidelity = np.mean(fidelities)
    print(f'fidelity = {fidelity}')
    return fidelity

if __name__ == "__main__":
    js = list(np.random.uniform(0,2,spins))
    hs = list(np.random.uniform(0,2,spins))
    # parameters = js + hs

    rest = {'type': 'continuous', 'domain': (-20,20)}

    bounds = [{'name': f'j_{j}', **rest} for j in range(spins)]
    bounds += [{'name': f'h_{h}', **rest} for h in range(spins)]

    optimisation = GPyOpt.methods.BayesianOptimization(cost,
                                                        domain=bounds,
                                                        model_type = 'GP',
                                                        acquisition_type='EI',  
                                                        normalize_Y = True,
                                                        acquisition_weight = 2,
                                                        maximize=True)

    max_iter = 600  # maximum time 40 iterations
    max_time = 3000  # maximum time 60 seconds

    optimisation.run_optimization(max_iter, max_time, verbosity=True)

    print(f'Paramters optimise fidelity: {optimisation.x_opt}')
    print(f'Optimum fidelity: {-optimisation.fx_opt}')

    optimisation.plot_convergence()