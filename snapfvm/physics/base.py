"""
snapfvm/physics/base.py
-----------------------
Abstract Base Class.
UPDATED: Adds storage for Dirichlet Boundary Values.
"""
from abc import ABC, abstractmethod
import numpy as np

class PhysicsModel(ABC):
    def __init__(self):
        # Stores target values for named boundaries (e.g. {'inlet': [1.0, 100.0, ...]})
        self.boundary_values = {}

    @property
    @abstractmethod
    def num_variables(self):
        pass

    @property
    def has_viscous_terms(self):
        return False

    def set_boundary_value(self, name, values):
        """ Store a fixed state vector for a boundary tag. """
        self.boundary_values[name] = np.array(values, dtype=float)

    @abstractmethod
    def compute_flux(self, q_L, q_R, normal):
        pass
    
    @abstractmethod
    def compute_boundary_flux(self, q_L, normal, boundary_name):
        pass

    def compute_viscous_flux(self, q_L, q_R, g_L, g_R, normal):
        return np.zeros(self.num_variables)
    
    @abstractmethod
    def q_to_primitives(self, q):
        pass