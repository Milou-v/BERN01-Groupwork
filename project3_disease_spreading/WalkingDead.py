import numpy as np
from functools import partial
from tqdm import tqdm

class Zombie_Apocalypse:
    '''
    Class object for representing a society during a zombie apocalypse.
    S = Susceptible/Human
    I = Infected --> not yet turned
    Z = Zombie
    R = Removed/Dead

    ------
    ATTRIBUTES
    ------
    | initial status -> initial values of S, I, Z, R
    | beta -> bite rate / attack rate: [S -> I]
    | Omega_s -> natural deaths (no TBI): [S -> I]
    | Omega_tbi -> Suicides or head trauma deaths: [S -> R]
    | k -> reanimation after incubation (turn rate ~0.3/day): [I -> Z]
    | delta -> Mercy kills on dying (intervention rate): [I -> R]
    | alpha -> Combat kills on zombies: [Z -> R]
    | birth -> birth rate
    '''

    def __init__(self,
                 initial_status: list,
                 beta: float=0.02,
                 Omega_s: float=0.000015,
                 Omega_tbi: float=0.000015,
                 k: float=3.0,
                 delta: float=0.00001, 
                 alpha: float=0.0004,
                 birth: float=0.00007,
                 model_function: str='latent_zombies'
                 ):

       
        # Basic Model Parameters
        self._initial_status = np.array(initial_status)[np.newaxis,:]
        self._beta = beta
        self._Omega_s = Omega_s
        self._Omega_tbi = Omega_tbi
        self._k = k
        self._delta = delta
        self._alpha = alpha
        self._II = birth

        self.all_parameters = {
                    'initial_status': self._initial_status,
                    'beta': self._beta,
                    'Omega_s': self._Omega_s,
                    'Omega_tbi': self._Omega_tbi,
                    'k': self._k,
                    'delta': self._delta,
                    'alpha': self._alpha,
                    'II': self._II
                }

        #ODE system function
        self._set_model_function(model_function)

    def _set_model_function(self,model_function: str):
        print(self.all_parameters)
        self.model = model_function
        self._model_function = partial(functions[model_function],
                                        params=self.all_parameters)

# _____________________________________________________________________________________________________

    def euler_solver(self, delta_t: float, T_max: float, initial_status: np.array = None):
            if initial_status is None:
                initial_status = self._initial_status
            f = self._model_function
            status = np.array(initial_status, dtype=float)
            time = np.arange(start=0, stop=T_max, step=delta_t)
            status_array = [status.copy()]
            
            for t in tqdm(time[:-1]):
                # 1. Calculate the step update
                status += delta_t * f(status=status, t=t)
                
                # 2. GUARDRAIL: Instantly clamp everything at 0 so it can never go negative
                status = np.maximum(0.0, status)
                
                status_array.append(status.copy())

            return time, np.array(status_array)
    
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

# _____________________________________________________________________________________________________


def SIZR(status, t, params):
    """
    Lore-Accurate Zombie Model with Density-Dependent Transmission (S - I - Z - R)
    S: Susceptible (Living)
    I: Infected / Dying (Incubating, waiting to reanimate)
    Z: Zombies (Active walkers)
    R: Removed (Brains destroyed permanently)
    """
    S, I, Z, R = status.T

# Guardrail: Prevent negative values
    S = np.maximum(0.0, S)
    I = np.maximum(0.0, I)
    Z = np.maximum(0.0, Z)
    R = np.maximum(0.0, R)

    # Dynamic Total Population
    N = S + I + Z + R
    N = np.where(N == 0, 1.0, N) 

    # Parameters (with balanced defaults)
    birth = params["II"]                     # Realistic daily per-capita birth rate
    beta = params["beta"]                     # Bite rate (density-dependent)
    omega_s = params["Omega_s"]              # Natural death, intact brain
    omega_tbi = params["Omega_tbi"]          # Suicide / TBI death
    k = params["k"]                         # Incubation turn rate (8 hours = ~3.0/day)
    delta = params["delta"]                  # Mercy kills on dying
    alpha = params["alpha"]                 # Zombie combat elimination rate (proportional to Z)

    # Flow rates
    # Note: Birth is per capita (birth * S) or flat. Let's use per capita to tie to S.
    birth_flux = birth * S
    infection_flux = (beta * S * Z / N)
    natural_death_flux = omega_s * S
    tbi_death_flux = omega_tbi * S
    incubation_flux = k * I
    mercy_flux = delta * I
    combat_flux = S/N * alpha * Z  # Fixed: scales with active zombies, not S * Z

    # Differential equations
    dS = birth_flux - infection_flux - natural_death_flux - tbi_death_flux
    dI = infection_flux + natural_death_flux - incubation_flux - mercy_flux
    dZ = incubation_flux - combat_flux
    dR = tbi_death_flux + mercy_flux + combat_flux

    return np.column_stack((dS, dI, dZ, dR))


# Function dictionary:
functions = {'latent_zombies': SIZR}