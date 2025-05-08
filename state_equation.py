import numpy as np
from PDEModel1DWithExactSoln import PDEModel1DWithExactSoln

class StateEquation(PDEModel1DWithExactSoln):
    """
    Exact solution of variable speed problem. Boundary conditions are inflow
    Q_{in}(t) = cos(t) and no refelction at x = L.
    """

    def __init__(self):
        super().__init__(1)
        """Initialize constants."""
        self.c0 = 1.0
        self.alpha = 1.5
        self.a = 0.1
        self.b = 0.9
        self.L = 1.0

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


    def f(self, x):
        """Return the integral of 1/c(x) from 0 to x"""
        
        if x < self.a:
            return x / self.c0
                
        elif self.a <= x < self.b:
           
           T1 = self.a/self.c0
           c_at_x = self.c(x) # = c0*exp(alpha*(x-a))
           T2 = 1/self.alpha * (1.0/self.c0 - 1/c_at_x)
           return T1 + T2
                
        else:
            T1 = self.a/self.c0
            c_at_b = self.c(self.b) # = c0*exp(alpha*(x-a))
            T2 = 1/self.alpha * (1.0/self.c0 - 1/c_at_b)
            T3 = (x-self.b)/c_at_b

            return T1 + T2 + T3

    def applyLeftBC(self, x, t, dx, dt, u):
        """Inflow at the left boundary"""

        left = np.cos(8 * np.pi * t)

        return left
    

    def applyRightBC(self, x, t, dx, dt, u):
        """Setup the right boundary conditons"""

        # Get u at the final and penultimate point
        u_pen = u[:, -2]
        u_ult = u[:, -1]

        # Advect w2 right to left
        right = u_ult - self.c(x) * (dt / dx) * (u_ult - u_pen)

        return right


    # --------------------------------------------------------------------
    # Method for PDEModel1DWithExactSoln
    def exact_solution(self, X, t):
        """
        Exact solution u(x, t) of our PDE 
        u_t + c(x) * u_x = 0
        
        """
        U = np.zeros_like(X)

        for ix, x in enumerate(X):

            U[ix] = np.cos(8 * np.pi * (t - self.f(x)))
        
        return U