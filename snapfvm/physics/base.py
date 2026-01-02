"""
snapfvm/physics/base.py
-----------------------
Abstract Base Class.
UPDATED: Implements Face-Flux Boundary Interface.
"""
from abc import ABC, abstractmethod
import numpy as np

class PhysicsModel(ABC):
    @property
    @abstractmethod
    def num_variables(self):
        pass

    @property
    def has_viscous_terms(self):
        return False

    @abstractmethod
    def compute_flux(self, q_L, q_R, normal):
        """ Internal Face Flux. """
        pass
    
    @abstractmethod
    def compute_boundary_flux(self, q_L, normal, boundary_name):
        """ 
        Boundary Face Flux.
        Args:
            q_L: State in the cell adjacent to boundary.
            normal: Outward pointing normal vector (with area magnitude).
            boundary_name: String tag (e.g. 'inlet', 'wall').
        Returns:
            flux: Vector [num_variables] LEAVING the domain.
        """
        pass

    def compute_viscous_flux(self, q_L, q_R, g_L, g_R, normal):
        return np.zeros(self.num_variables)
    
    @abstractmethod
    def q_to_primitives(self, q):
        pass