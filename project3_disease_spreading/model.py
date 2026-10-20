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
    |initial_value -> [S,I,R] initial value
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
                     model_function:str,
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

        #Total parameters
        self.total_parameters = {
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
        self.model = model_function
        self._model_function = partial(model_functions[model_function],
                                       params=self.total_parameters)


    def euler_solver(self,delta_t:float,T_max:float,initial_status=np.array):
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
        f = self._model_function
        status = np.array(initial_status,dtype=float)
        time = np.arange(start=0,stop=T_max,step=delta_t)
        status_array = [status.copy()]
        for t in tqdm(time[:-1]):
            status += delta_t*f(status=status,t=t)
            status_array.append(status.copy())

        return time,np.array(status_array)


    def RK4_solver(self,delta_t:float,T_max:float,initial_status:np.array):
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



## MODEL SYSTEM FUNCTIONS
def SIR(status,t,params):
    """
    Normal SIR model
    """
    S = status[0]
    I = status[1]
    R = status[2]

    N = params["N"]
    beta = params["beta"]
    gamma = params["gamma"]
    alpha = params["alpha"]

    fs = -beta*S*I/N - gamma*S
    fi =  beta*S*I/N - alpha*I
    fr = gamma*S + alpha*I

    if abs(S+I+R-N) > tol:
        raise ValueError(f"SIR Model discrepancy between\n\nFixed population\nN:{N}\n\n"+
                         f"Population status\nS:{S}\nI:{I}\nR:{R}\n\nCheck for correct initial values")
    
    return np.array([fs,fi,fr])


def SIRD(status,t,params):
    """
    SIR model with natality and defunctions
    """
    S = status[0]
    I = status[1]
    R = status[2]
    D = status[3]

    if t==0 and D > 0.0:
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
    
    return np.array([fs,fi,fr,fd])


model_functions = { "basic_SIR": SIR,
                    "mortality_SIR": SIRD}

tol = 1e-6