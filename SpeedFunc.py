from abc import ABC, abstractmethod
from Grid1D import Grid1D
from scipy.interpolate import interp1d
import numpy as np

class SpeedFunction(ABC):
    """This class provides an abstract interface that returns the wave speed
    in a 1D hyperbolic PDE"""

    def __init__(self):
        """Constructor for a speed function. The base class constructor doesn't
        do anything for now.ß"""
        pass

    @abstractmethod
    def c(self, x):
        """Return the speed. A derived class should implement this function"""
        pass

    @abstractmethod
    def f(self, x):
        """Compute f(x), the integral of 1/c(x) from 0 to x"""
        pass


class ConstantSpeedFunction(SpeedFunction):
    """Class for a constant (i.e., space-independent) speed"""
    def __init__(self, cVal):
        super().__init__() # Call base class ctor
        self.cVal = cVal

    def c(self, x):
        return self.cVal
    
    def f(self, x):
        return x/self.cVal

class ThreeZoneSpeedFunction(SpeedFunction):
    """Class for a three-zone speed function in which
    the speed is constant in zones I and III, exponential in the middle
    zone II."""
    def __init__(self, c0=1.0, a=0.1, b=0.9, alpha=0.25):
        super().__init__() # Call base class ctor
        self.c0 = c0
        self.a = a
        self.b = b
        self.alpha = alpha

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


    
class InterpolatedSpeedFunction(SpeedFunction):

    def __init__(self, nx=64, L=1.0, speed_to_interpolate=None):
        super().__init__()
        assert(speed_to_interpolate!=None)
        self.grid = Grid1D(xMin=0, xMax=L, nx=nx)
        self.cVec = np.zeros_like(self.grid.X)
        self.fVec = np.zeros_like(self.grid.X)

        for i,x in enumerate(self.grid.X):
            self.cVec[i] = speed_to_interpolate.c(x)
            self.fVec[i] = speed_to_interpolate.f(x)

        self.c_interp = interp1d(self.grid.X, self.cVec, kind="linear", 
                                 fill_value="extrapolate")
        
        self.f_interp = interp1d(self.grid.X, self.fVec, kind="linear", 
                                 fill_value="extrapolate")
        
    def c(self, x):
        return self.c_interp(x)
    

    def f(self, x):
        return self.f_interp(x)
    


        


