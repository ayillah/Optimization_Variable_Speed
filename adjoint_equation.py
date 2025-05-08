import numpy as np
from PDEModel1DWithExactSoln import PDEModel1DWithExactSoln

class AdjointEquation(PDEModel1DWithExactSoln):
    """
    Adjoint equation lambda(x, t)_t + (c(x) * lambda(x, t))_x
    = -rho(x, t) * [u(x, t) - u_star(x, t)]
    with variable speed c(x) and constant function rho(x, t)
    """

    def __init__(self):
        super().__init__(1)
        """Initialize constants."""
        self.c0 = 1.0
        self.alpha = 0.25
        self.a = 0.1
        self.b = 0.9
        self.L = 1.0
        self.Omega = 8 * np.pi

    # ----------------------------------------------------------------------
    # Methods of PDEModel1D 

    def c(self, x):
        """Compute the speed c(x)"""
        if x < self.a:
            return self.c0
        elif self.a <= x and x < self.b:
            return self.c0*np.exp(self.alpha*(x-self.a))
        else:
            return self.c0*np.exp(self.alpha*(self.b-self.a))
    
    def rho(self, x):

        return self.c0


    def f(self, x):
        return super().f(x)

    def applyLeftBC(self, x, t, T, dx, dt, u):
        """Non-reflecting"""

        #left = (1 - x**2 / self.L**2) * np.sin(self.Omega * (t - T)) / self.c(x)

        #return left
    
        return super().applyLeftBC(self, x, t, T, dx, dt, u)
    

    def applyRightBC(self, x, t, dx, dt, u):
        return super().applyRightBC(self, x, t, dx, dt, u)


    # --------------------------------------------------------------------
    # Method for PDEModel1DWithExactSoln
    def exact_solution(self, X, t, T):
        """
        Exact solution lambda(x, t) of our PDE 
        lambda(x, t)_t + (c(x) * lambda(x, t))_x = -rho(x, t) * [u(x, t) - u_star(x, t)]
        
        """
        U = np.zeros_like(X)

        for ix, x in enumerate(X):

            U[ix] = (1 - x**2 / self.L**2) * np.sin(self.Omega * (t - T)) / self.c(x)
        
        return U