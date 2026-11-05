import numpy as np
from functools import partial
from tqdm import tqdm


class model:
    """
    Object for representing classical SIR model,
    plus optional parameters for other model variants

    ------
    INPUTS
    ------
    |sigma -> number of contacts between individuals per unit of time
    |T -> probability of disease transmission per contact
    |N -> initial number of population
    |gamma -> vaccinations per unit of time
    |alpha -> recoveries per unit of time
    |model_function -> model function selected

    ---------------
    OPTIONAL INPUTS
    ---------------
    |Alpha -> birth rate per unit of time
    |Omega_S -> death rate of susceptible population per unit of time
    |Omega_I -> death rate of infected population per unit of time
    |Omega_R -> death rate of recovered population per unit of time

    """


    def __init__(self,
                 sigma:float,
                 T:float,
                 N:int,
                 gamma:float,
                 alpha:float,
                 initial_status:list,
                 model_function:str="basic_SIR",
                 Alpha:float=0.0,
                 Omega_S:float=0.0,
                 Omega_I:float=0.0,
                 Omega_R:float=0.0):

        #Basic model parameters
        self._N = N
        self._sigma = sigma
        self._T = T
        self._beta = sigma*T
        self._gamma = gamma
        self._alpha = alpha

        #Optional parameters
        self._Alpha = Alpha
        self._Omega_S = Omega_S
        self._Omega_I = Omega_I
        self._Omega_R = Omega_R

        #Initial state
        self._initial_status = np.array(initial_status)[np.newaxis,:]
        if sum(initial_status) != N:
            raise ValueError(f"Provided initial status must sum up to the initial population value")
        
        #Total parameters
        self.total_parameters = {
            "initial_status":self._initial_status,
            "N":self._N,
            "beta":self._beta,
            "gamma":self._gamma,
            "alpha":self._alpha,
            "Alpha":self._Alpha,
            "Omega_S":self._Omega_S,
            "Omega_I":self._Omega_I,
            "Omega_R":self._Omega_R
        }

        #ODE system function
        self._set_model_function(model_function)


    def _set_model_function(self,model_function: str):
        self.model = model_function
        self._model_function = partial(model_functions[model_function],
                                        params=self.total_parameters)
        
    def euler_solver(self,delta_t:float,T_max:float,initial_status:np.array=None):
        """
        Given an initial [S,I,R] (or other lenght array) values, solve the system according 
        to Euler's method
        ------
        INPUTS
        ------
        delta_t -> infinitesimal time step value (in unit time)
        T_max -> total simulation time (in unit time)
        initial_status -> [S,I,R] initial values

        """
        if initial_status is None:
            initial_status = self._initial_status
        f = self._model_function
        status = np.array(initial_status,dtype=float)
        time = np.arange(start=0,stop=T_max,step=delta_t)
        status_array = [status.copy()]
        for t in tqdm(time[:-1]):
            status += delta_t*f(status=status,t=t)
            status_array.append(status.copy())

        return time,np.array(status_array)


    def RK4_solver(self,delta_t:float,T_max:float,initial_status:np.array=None):
        """
        Given an initial [S,I,R] (or other lenght array) values, solve the system according 
        to Runge-Kutta with 4 steps
        ------
        INPUTS
        ------
        delta_t -> infinitesimal time step value (in unit time)
        T_max -> total simulation time (in unit time)
        initial_status -> [S,I,R] initial values

        """
        if initial_status is None:
            initial_status = self._initial_status
        f = self._model_function
        status = np.array(initial_status,dtype=float)
        time = np.arange(start=0,stop=T_max,step=delta_t)
        status_array = [status.copy()]

        c1 = delta_t
        c2 = delta_t/2
        c3 = delta_t/6

        for t in tqdm(time[:-1]):
            k1 = f(status=status, t=t)
            k2 = f(status=status+c2*k1, t=t+c2)
            k3 = f(status=status+c2*k2, t=t+c2)
            k4 = f(status=status+c1*k3, t=t+c1)

            status += c3*(k1+2*k2+2*k3+k4)
            status_array.append(status.copy())

        return time,np.array(status_array)


    def estimate_numerical_errors(self, delta_t: float, T_max: float):
        """
        Solves the system using both Euler and RK4, then estimates 
        truncation and round-off errors.
        """
        # 1. Run both solvers and squeeze out any extra dimensions (e.g., shape (n, 1, 3) -> (n, 3))
        time_euler, sol_euler = self.euler_solver(delta_t=delta_t, T_max=T_max)
        time_rk4, sol_rk4 = self.RK4_solver(delta_t=delta_t, T_max=T_max)
        
        sol_euler = np.squeeze(sol_euler)
        sol_rk4 = np.squeeze(sol_rk4)
        
        # 2. Truncation Error Estimation (Euler vs RK4 Reference)
        abs_trunc_error = np.abs(sol_euler - sol_rk4)  # Shape: (n_timesteps, 3)
        max_trunc_error = np.max(abs_trunc_error, axis=0)  # Max error per category [S, I, R]
        
        # Relative truncation error
        rel_trunc_error = abs_trunc_error / (np.abs(sol_rk4) + 1e-15)
        max_rel_trunc_error = np.max(rel_trunc_error, axis=0)
        
        # Root Mean Square Error (RMSE) for each category
        rmse = np.sqrt(np.mean(abs_trunc_error**2, axis=0))
        
        # 3. Round-off Error Estimation
        mu = 2**(-52)  # 64-bit double precision machine epsilon (~2.22e-16)
        n_steps = len(time_euler)
        accumulated_roundoff = np.sqrt(n_steps) * mu
        
        # 4. Print Summary Report Safely using .item()
        print(f"==================================================")
        print(f" NUMERICAL ERROR ANALYSIS (delta_t = {delta_t}, Steps = {n_steps})")
        print(f"==================================================")
        print(f"Machine Epsilon (μ): {mu:.2e}")
        print(f"Estimated Accumulated Round-off Error: ~{accumulated_roundoff:.2e}\n")
        
        categories = ['Susceptible (S)', 'Infected (I)', 'Recovered (R)']
        for i, cat in enumerate(categories):
            print(f"[{cat}]")
            print(f"  Max Absolute Truncation Error: {max_trunc_error[i].item():.6e}")
            print(f"  Max Relative Truncation Error: {max_rel_trunc_error[i].item():.4%}")
            print(f"  RMSE:                          {rmse[i].item():.6e}")
            print("-" * 50)
            
        return {
            'abs_trunc_error': abs_trunc_error,
            'max_trunc_error': max_trunc_error,
            'rmse': rmse,
            'roundoff_estimate': accumulated_roundoff
        }


## MODEL SYSTEM FUNCTIONS
def SIR(status,t,params):
    """
    Normal SIR model
    """

    S,I,R = status.T

    N = params["N"]
    beta = params["beta"]
    gamma = params["gamma"]
    alpha = params["alpha"]

    fs = -beta*S*I/N - gamma*S
    fi =  beta*S*I/N - alpha*I
    fr = gamma*S + alpha*I

    # if np.any(np.abs(S+I+R-N)) > tol:
    #     raise ValueError("SIR Model discrepancy between fixed population and population status"+
    #                      "\n\nCheck for correct initial values")
    
    return np.column_stack((fs,fi,fr))


def SIRD(status,t,params):
    """
    SIR model with natality and de[ath ]functions
    """
    S, I, R, D = status.T

    if t==0 and np.any(D > 0.0):
        raise ValueError(f"SIRD Model should not start with {D} infection deaths")

    N = S+I+R
    beta = params["beta"]
    gamma = params["gamma"]
    alpha = params["alpha"]

    Alpha = params["Alpha"]
    Omega_S = params["Omega_S"]
    Omega_I = params["Omega_I"]
    Omega_R = params["Omega_R"]

    fs = -beta*S*I/N - gamma*S + Alpha - Omega_S*S
    fi =  beta*S*I/N - alpha*I - Omega_I*I
    fr = gamma*S + alpha*I - Omega_R*R
    fd = Omega_I*I
    
    return np.column_stack((fs,fi,fr,fd))


def SIRM(status,t,params):
    """
    SIR model with permanent mobility between multiregions

    W matrix represents the mobility matrix:
    elemnt ij is the ratio of population moving from j to i per unit of time
    """
    S,I,R = status.T

    W = params["W"]
    O = W.sum(axis=0) #Outflow
    N = S+I+R
    beta = params["beta"]
    gamma = params["gamma"]
    alpha = params["alpha"]

    fs = -beta*S*I/N - gamma*S + W @ S - O*S
    fi =  beta*S*I/N - alpha*I + W @ I - O*I
    fr = gamma*S + alpha*I + W @ R - O*R


    return np.column_stack((fs,fi,fr))

model_functions = { "basic_SIR": SIR,
                    "mortality_SIR": SIRD,
                    "mobility_SIR":SIRM}

tol = 1e-6