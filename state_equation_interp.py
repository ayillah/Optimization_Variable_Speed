import numpy as np
from PDEModel1DWithExactSoln import PDEModel1DWithExactSoln
from SpeedFunc import ThreeZoneSpeedFunction, ConstantSpeedFunction


class StateEquationInterp(PDEModel1DWithExactSoln):
    """
    Exact solution of variable speed problem. Boundary conditions are inflow
    Q_{in}(t) = cos(t) and no refelction at x = L.
    """

    def __init__(self, nx=64, speed_func=ConstantSpeedFunction(1.0)):
        super().__init__(1)
        """Initialize constants."""
        
        self.L = 1.0
        self.nx = nx
        self.speed_func = speed_func

    def c(self, x):
        return self.speed_func.c(x)

    # ----------------------------------------------------------------------
    # Methods of PDEModel1D

    def f(self, x):
        """Compute the characteristic curve using interpolated speeds."""
        return self.speed_func.f(x)

    def applyLeftBC(self, x, t, dx, dt, u):
        """Inflow at the left boundary"""

        left = np.cos(8 * np.pi * t)

        return left

    def applyRightBC(self, x, t, dx, dt, u):
        """Setup the right boundary conditions using second-order approximation."""
        u_pen = u[:, -2]
        u_ult = u[:, -1]

        # Get the appropriate speed for x > b using current nx
        speeds = self.speed_func.c(x)

        # Second-order characteristic-based boundary condition
        right = u_ult - speeds * (dt/dx) * (u_ult - u_pen)
        
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