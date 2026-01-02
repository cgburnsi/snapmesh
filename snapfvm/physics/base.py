"""
snapfvm/physics/base.py
-----------------------
The Abstract Base Class for Physics Kernels.
Defines the strict interface for Conservation Laws.
"""
from abc import ABC, abstractmethod
import numpy as np

class PhysicsModel(ABC):
    @property
    @abstractmethod
    def num_variables(self):
        """ Number of equations (e.g., 4 for 2D Euler). """
        pass

    @property
    def has_viscous_terms(self):
        """ 
        If True, Solver computes gradients (Green-Gauss) 
        and calls compute_viscous_flux. 
        """
        return False

    @abstractmethod
    def compute_flux(self, q_L, q_R, normal):
        """ 
        Computes Convective Flux F(Q) projected onto normal.
        Must return vector of size [num_variables].
        """
        pass
    
    def compute_viscous_flux(self, q_L, q_R, grad_q_L, grad_q_R, normal):
        """ Computes Diffusive Flux (Heat/Shear). Optional. """
        return np.zeros(self.num_variables)
    
    @abstractmethod
    def q_to_primitives(self, q):
        """ Decodes Conservative Q -> Primitive Vars for checking. """
        pass