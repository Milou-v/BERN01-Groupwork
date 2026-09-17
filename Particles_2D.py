import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm
import pickle
import os
from pathlib import Path
from numba import njit




class Particles_2D():

    def __init__(self,N=100,delta_step=0.4,density=0.1,temperature=0.1,R=None):

        #constant parameters
        self.sigma_0 = 1    # Particle hard core diameter
        self.sigma_1 = 2.5  # Particle soft core diameter


        #system parameters
        self.N = N                                         # Number of particles                       
        self.density = density                             # Reduced density defined as N*(sigma_0^2)/L^2
        self.temperature = temperature                     # Reduced temperature defined as kT/epsilon
        self.beta = 1/self.temperature                     # Reduced beta defines as epsilon/kT
        self.LENGTH = self.sigma_0*np.sqrt(N/self.density) # Side length of boundary box


        #system variables
        if R is not None:
            self.R = R      # Particle positions vector (randomly initialize if no input provided)
        else:
            self.R = np.random.uniform(0, self.LENGTH, size=(N, 2))

        self.overlaps = self.calc_total_overlaps()  # Number of particles with overlapping hard cores    
        self.energy = self.calc_total_U()           # Total system energy obtained by the sum of potentials
        self.accepted_movements = 0                 # Number of accepted particle movements in current Monte-Carlo loop
        self.mc_iterations = 0                      # Number of Monte-Carlo iterations for whole system
        self.production_iterations = 0              # Number of Monte-Carlo iterations for production phase
        self.energy_mean = 0                        # Energy average for current iteration
        self.energy_M2 = 0                          # Sum of squares of the difference between each computed energy state and the average
        self.equilibrium_array = []                 # Array of energies used in equilibrium phase
        self.max_iter = 0                           # Maximum number of iterations in current Monte-Carlo run
        self.delta_step = delta_step                # Delta parameter for random particle stepping


        # Radial distribution attributes
        self.bin_centers = None                     # Store an array after calling the gr function
        self.gr = None                              # Store gr values         "          "


    def calc_total_U(self):
        """
        Calculation of the system total energy by iterating through
        each particle.

        If any pair of particles hard cores are overlapping, it returns np.inf,
        in any other case, it returns the finite energy (unitless).

        INPUT
        self :: particle system (class)

        OUTPUT
        Energy (np.inf/int)
        """
        total_U = 0
        for i in range(0, len(self.R) - 1):

            # x and y differences from particle i
            # to all particles after it
            dx = self.R[i, 0] - self.R[i + 1:, 0]
            dy = self.R[i, 1] - self.R[i + 1:, 1]

            # PBC condition
            dx = dx - self.LENGTH * np.round(dx / self.LENGTH)
            dy = dy - self.LENGTH * np.round(dy / self.LENGTH)

            # Distances squared
            r2 = dx**2 + dy**2

            # Particle overlapping condition
            if np.any(r2 < self.sigma_0**2):
                return np.inf

            # Total energy contribution of particle i
            total_U += np.count_nonzero(r2 < self.sigma_1**2)

        return total_U
    

    def calc_new_pos(self,p):
        """
        Randomly move particle position by the equation
        p = p + ∆(ξ - 0.5)
        Where ξ is a random vector ~~ [Uniform[0,1],Uniform[0,1]]
        """
        ξ = np.random.sample(2)
        p_new = p + self.delta_step*(ξ - 0.5)

        #PBC condition 
        return p_new % self.LENGTH


    def calc_total_overlaps(self):
        """
        Calculation of the system total overlaps

        INPUT
        self :: particle system (class)

        OUTPUT
        nº of overlaps (int)
        """
        total_overlaps = 0
        for i in range(0, len(self.R) - 1):

            # x and y differences from particle i
            # to all particles after it
            dx = self.R[i, 0] - self.R[i + 1:, 0]
            dy = self.R[i, 1] - self.R[i + 1:, 1]

            # PBC condition
            dx = dx - self.LENGTH * np.round(dx / self.LENGTH)
            dy = dy - self.LENGTH * np.round(dy / self.LENGTH)

            # Distances squared
            r2 = dx**2 + dy**2

            # Overlaps contributed from particle i
            total_overlaps += np.count_nonzero(r2 < self.sigma_0**2)
            
        return total_overlaps


    def calc_individual_overlaps(self,p_idx,p):
        """
        Calculate the number of overlaps in a single particle.

        INPUT
        self :: particles system (class) 
        p_idx :: particle index (int)
        p :: particle coordinates (np.array)

        OUTPUT
        nº overlaps (int)
        """
        # coordinates differences from particle p to every other one
        dx = p[0] - self.R[np.arange(self.N) != p_idx,0]
        dy = p[1] - self.R[np.arange(self.N) != p_idx,1]

        #PBC condition
        dx = dx - self.LENGTH * np.round(dx / self.LENGTH)
        dy = dy - self.LENGTH * np.round(dy / self.LENGTH)

        # Distances squared
        r2 = dx**2 + dy**2

        return np.count_nonzero(r2 < self.sigma_0**2)


    def update_estimators(self):
        """
        Welford's online algorithm for estimating the variance of energy
        """
        current_mean = self.energy_mean
        delta = self.energy - current_mean
        self.energy_mean += delta / self.production_iterations
        self.energy_M2 += delta * (self.energy - self.energy_mean)


    def run_mc(self,max_iter=2*(10**7),hotstart=False,production=False,equilibrium=False,delta_optim=False):
        """
        Monte-Carlo simulation of the system
        """
        self.energy = self.calc_total_U()
        self.max_iter = max_iter
        accept_loc = 0
        for i in tqdm(range(max_iter),miniters=max(1, max_iter//100)):
            self.mc_iterations += 1

            # Randomly select a particle
            p_idx = np.random.choice(self.N)
            p = self.R[p_idx]

            # Make a trial displacement
            p_new = self.calc_new_pos(p)
            # Calculate the change of energy in the system if we were accept the particle displacement
            U_m = calc_individual_U(self.R,p_idx,p,self.LENGTH,self.sigma_0,self.sigma_1)       #Energy contribution of current particle
            U_n = calc_individual_U(self.R,p_idx,p_new,self.LENGTH,self.sigma_0,self.sigma_1)   #Energy contribution of trial particle

            # Acceptance conditions for state transition
            # If both states contribute infinite energy, don´t accept
            # If the sugested state has a finite energy and current is infinite, accept and recalculate overlaps (if in hotstart)
            # If the curent state has a finite energy, accept only if:
            # ξ < e−β∆U , where ξ is a random number (0 < ξ < 1) and β is the reduced temperature in our model
            if np.isinf(U_m):
                if np.isinf(U_n):
                    pass            
                else:
                    if hotstart:                
                        self.overlaps -= self.calc_individual_overlaps(p_idx,p)
                    self.R[p_idx] = p_new
                    self.accepted_movements += 1
                    accept_loc += 1
                   
            else:
                ξ = np.random.random()
                delta_U = U_n - U_m
                if ξ < np.exp(-self.beta*delta_U) and not np.isinf(U_n):
                    self.R[p_idx] = p_new
                    self.accepted_movements += 1
                    accept_loc += 1
                    self.energy += delta_U

            # If the simulation is running on hotstart, then it stops when reaching a state with no overlaps
            if hotstart and self.overlaps==0:
                print(f"Cold start reached in {self.mc_iterations}")
                break

            # System checks
            if (i % (max_iter//10) == 0) and not hotstart:

                # Verifies if every particle remains inside the box
                for idx,p in enumerate(self.R):
                    if p[0]<0 or p[1]<0 or p[0]>self.LENGTH or p[1]>self.LENGTH:
                        raise ValueError(f"Particle nº{idx} with coordinates {p} is outside of the box")

                # Recalculate energy and overlaps; checks for discrepancies
                current_energy = self.energy
                self.energy = self.calc_total_U()
                if abs((current_energy-self.energy)/self.energy) > 10**(-11):
                    raise ValueError(f"Energy discrepancy between\nCurrent energy:{current_energy}\nRecalculated energy:{self.energy}")

                self.overlaps = self.calc_total_overlaps()
                if self.overlaps != 0:
                    raise ValueError(f"{self.overlaps} overlaps encountered")

            # Equilibrium plot
            if (i % (max_iter//1000) == 0) and equilibrium:
                self.equilibrium_array.append(-self.beta*self.energy)

            # Dynamical change of delta in equilibrium phase:
            if delta_optim and i % 500 == 0 and i > 0:
                ratio = accept_loc / 500
                if ratio > 0.7:
                    self.delta_step *= 1.1

                elif ratio < 0.25:
                    self.delta_step *= 0.9
                self.delta_step = min(self.delta_step, self.LENGTH)
                accept_loc = 0


            # System update of statistical estimators
            if production:
                self.production_iterations += 1
                self.update_estimators()


    def state_snapshot(self,directory=r"snapshots_drop_temp"):    

        fig, ax = plt.subplots()
        ax.scatter(self.R[:, 0], self.R[:, 1], color="#189536cc", alpha=0.67)
        
        # Plot configuration
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        
        # Save snapshot in specified directory
        rute = os.path.join(directory, f"d_{self.density}_t_{self.temperature}.jpg")
        plt.savefig(rute, bbox_inches='tight', dpi=300)
        plt.close(fig)


    def scatter(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.LENGTH)
        ax.set_ylim(0, self.LENGTH)
        ax.set_aspect("equal")
        
        # Soft cores
        for x, y in self.R:
            ax.add_patch(
                plt.Circle((x, y), self.sigma_1/2, color="#189536cc", alpha=0.09)
            )

        # Hard cores
        for x, y in self.R:
            ax.add_patch(
                plt.Circle((x, y), self.sigma_0/2, color="#189536cc", alpha=0.5)
            )

        ax.set_aspect("equal")
        # Plot configuration
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.title(fr"$\rho={self.density}$")
        plt.show()


    def print_status(self):
        """
        Print relevant class attributes
        """
        m = (f"Overlaps: {self.overlaps}" 
        f"\nEnergy: {self.energy}"
        f"\nAccepted movements: {self.accepted_movements}"
        f"\nCompleted iterations: {self.mc_iterations}"
        f"\nAcceptance ratio:{np.round(self.accepted_movements/max(1,self.mc_iterations),2)}"
        f"\n\n\nSystem Parameters"
        f"\nNumber of particles:{self.N}"
        f"\nDensity*:{self.density}"
        f"\nTemperature*:{self.temperature}"
        f"\nDelta Step:{self.delta_step}")
        print(m)


    def reset_stats(self):
        """
        Function that resets statistical attributes in the system class
        """
        self.accepted_movements = 0
        self.mc_iterations = 0
        self.production_iterations = 0
        self.energy_mean = 0
        self.energy_M2 = 0
        self.equilibrium_array = []


    def radial_distribution(self, dr=0.05):
        """
        Radial distribution function that plots the average density at a distance dr from a particle 
        
        Arguments:
        dr : float
            Width of distance bins.
            
        Returns:
        bin_centers : np.ndarray, Radius values (r).
        gr : np.ndarray, Radial distribution values g(r).
        """
        n = len(self.R)
        max_r = self.LENGTH / 2.0  # Only measure up to half the box size
        bins = np.arange(0, max_r + dr, dr)
        
        # 1. Compute all pairwise distances with Periodic Boundary Conditions (PBC)
        distances = []
        for i in range(n - 1):
            dx = self.R[i, 0] - self.R[i+1:, 0]
            dy = self.R[i, 1] - self.R[i+1:, 1]
            
            dx = dx - self.LENGTH * np.round(dx / self.LENGTH)
            dy = dy - self.LENGTH * np.round(dy / self.LENGTH)
            
            r = np.sqrt(dx**2 + dy**2)
            distances.extend(r)
        distances = np.array(distances)
        
        # 2. Histogram particle distances
        counts, _ = np.histogram(distances, bins=bins)
        
        # Multiply by 2 because pair (i,j) means i sees j AND j sees i
        counts = counts * 2.0
        
        # 3. Normalize to get g(r)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0
        shell_areas = np.pi * (bins[1:]**2 - bins[:-1]**2)  # Area of circular ring
        bulk_density = n / (self.LENGTH**2)                      # Average overall density
        
        # Normalization equation: local density / bulk density
        self.gr = (counts / n) / (shell_areas * bulk_density)

        plt.figure(figsize=(7, 4))
        plt.plot(self.bin_centers, self.gr, lw=2)
        plt.axhline(1.0, color="gray", linestyle="--")  # Ideal gas bulk limit
        plt.xlabel("Distance r")
        plt.ylabel("g(r)")
        plt.title("Radial Distribution Function")
        plt.grid(True)
        plt.show()



def coldstart_finder(N=100,max_iter=2*(10**5)):
    """
    Run simulations for predefined densities to find coldstart positions.
    If found, store a pickled class into the local folder "coldstart".
    """
    densities_delta = {0.1:2
                       ,0.15:2
                       ,0.227:2
                       ,0.291:2
                       ,0.38:2}

    for density,delta in densities_delta.items():
        test_system = Particles_2D(N=N,delta_step=delta,density=density,temperature=10)
        test_system.run_mc(max_iter=max_iter,hotstart=True)
        if test_system.overlaps != 0:
            test_system.print_status()
            test_system.scatter()
            raise ValueError(f"No coldstart found after max_iter({max_iter})\n"
                             f"at density:{density} with delta:{delta}")

        with open(f"coldstart_pickles/d_{density}_N_{N}.pkl", "wb") as f:
            pickle.dump(test_system, f)



@njit
def calc_individual_U(R,p_idx,p,LENGTH,sigma_0,sigma_1):
    """
    Calculate the energy contribution in the system 
    atributed to a single particle.

    INPUT
    self :: particles system (class) 
    p_idx :: particle index (int)
    p :: particle coordinates (np.array)

    OUTPUT
    Energy (np.Inf/int)
    """
    energy = 0
    for i in range(len(R)):
        if i == p_idx:
            continue
        # coordinates differences from particle p to every other one
        dx = p[0] - R[i,0]
        dy = p[1] - R[i,1]

        #PBC condition
        dx = dx - LENGTH * np.round(dx / LENGTH)
        dy = dy - LENGTH * np.round(dy / LENGTH)

        # Distances squared
        r2 = dx**2 + dy**2

        # Particle overlapping condition
        if r2 < sigma_0**2:
            return np.inf

        if r2 < sigma_1**2:
            energy += 1

    return energy


def temperature_drop_production(coldstart,max_temp=0.3,min_temp=0.15,drop_step=0.01,density=0.291):
    """
    Monte-Carlo simulations of a temperature drop in the system
    """
    N = len(coldstart)
    temperatures = []
    T = max_temp

    while T >= 0.20 - 1e-12:
        temperatures.append(round(T, 3))
        T -= 0.025

    T = 0.19
    while T >= min_temp - 1e-12:
        temperatures.append(round(T, 3))
        T -= 0.01
    temperatures = list(dict.fromkeys(temperatures))

    system = Particles_2D(N=N,temperature=temperatures[0],density=density,R=np.array(coldstart, copy=True))
    for step, temperature in enumerate(temperatures):

        print("\n")
        print("#" * 70)
        print(
            f"T* = {temperature:.3f} "
            f"({step + 1}/{len(temperatures)})"
        )
        print("#" * 70)

        system.temperature = temperature
        system.beta = 1.0 / temperature


        system.reset_stats()
        system.run_mc(max_iter=(10**7),equilibrium=True,production=False)
        equilibrium_path = ("runs/equilibrium/"f"equ_{density}_{np.round(temperature,2)}_{N}.pkl")
        with open(equilibrium_path, "wb") as f:
            pickle.dump(system, f)


        if temperature <= 0.2:
            system.reset_stats()
            system.run_mc(max_iter=2*(10**7),production=True,equilibrium=False)
            production_path = ("runs/production/"f"prod_{density}_{np.round(temperature,2)}_{N}.pkl")
            with open(production_path, "wb") as f:
                pickle.dump(system, f)

        system.print_status()


