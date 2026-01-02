"""
snapfvm/physics/base.py
"""
from abc import ABC, abstractmethod
import numpy as np

class PhysicsModel(ABC):
    def __init__(self):
        self.boundary_values = {}

    @property
    @abstractmethod
    def num_variables(self):
        pass

    @property
    def has_viscous_terms(self):
        return False

    def set_boundary_value(self, name, values):
        self.boundary_values[name] = np.array(values, dtype=float)

    @abstractmethod
    def compute_flux(self, q_L, q_R, normal):
        pass
    
    # UPDATED SIGNATURE
    @abstractmethod
    def compute_boundary_flux(self, q_L, normal, boundary_name, distance=1.0):
        pass

    def compute_viscous_flux(self, q_L, q_R, g_L, g_R, normal):
        return np.zeros(self.num_variables)
    
    @abstractmethod
    def q_to_primitives(self, q):
        pass