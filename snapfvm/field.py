import numpy as np

class Field:
    def __init__(self, grid):
        """
        Stores the fluid state (Conservative and Primitive variables).
        """
        self.grid = grid
        N = grid.num_cells
        
        # --- 1. Conservative Variables (The Solver's "Truth") ---
        # These are the variables we actually time-step:
        # rho  : Density [kg/m^3]
        # rhou : Momentum X [kg/(m^2 s)]
        # rhov : Momentum Y [kg/(m^2 s)]
        # rhoE : Total Energy per unit volume [J/m^3]
        
        self.rho  = np.zeros(N)
        self.rhou = np.zeros(N)
        self.rhov = np.zeros(N)
        self.rhoE = np.zeros(N)
        
        # --- 2. Primitive Variables (Helper Views) ---
        # We calculate these from the conservatives whenever needed.
        # u, v : Velocity [m/s]
        # p    : Pressure [Pa]
        self.u = np.zeros(N)
        self.v = np.zeros(N)
        self.p = np.zeros(N)
        
        # --- 3. Gas Constants (Air) ---
        self.gamma = 1.4
        self.R_gas = 287.05  # J/(kg K)

    def primitives_from_conservatives(self):
        """
        Updates u, v, p based on the current rho, rhou, rhov, rhoE.
        Must be called at the start of every time step.
        """
        # 1. Velocity (u = rhou / rho)
        # Add a tiny epsilon to avoid division by zero if rho is empty
        inv_rho = 1.0 / (self.rho + 1e-12)
        
        self.u = self.rhou * inv_rho
        self.v = self.rhov * inv_rho
        
        # 2. Pressure
        # Energy Equation: E = p/(gamma-1) + 0.5*rho*(u^2 + v^2)
        # Solve for p: p = (gamma-1) * (E - Kinetic_Energy)
        
        kinetic_energy = 0.5 * self.rho * (self.u**2 + self.v**2)
        internal_energy = self.rhoE - kinetic_energy
        
        self.p = (self.gamma - 1.0) * internal_energy
        
        # Safety clamp for pressure (prevents negative pressure crashes)
        self.p = np.maximum(self.p, 1e-6)

    def set_uniform_condition(self, rho, u, v, p):
        """
        Sets the entire field to a uniform initial state.
        Useful for initializing the simulation.
        """
        self.rho[:] = rho
        self.u[:]   = u
        self.v[:]   = v
        self.p[:]   = p
        
        # Calculate Conservatives
        self.rhou[:] = rho * u
        self.rhov[:] = rho * v
        self.rhoE[:] = (p / (self.gamma - 1.0)) + 0.5 * rho * (u**2 + v**2)