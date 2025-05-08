from abc import abstractmethod # ABC = "Abstract base class"
from PDEModel1D import PDEModel1D

class PDEModel1DWithExactSoln(PDEModel1D):

  def __init__(self, numVars):
    super().__init__(numVars)
  
  @abstractmethod
  def exact_solution(self, X, t):
    '''
    Evaluate the exact solution at every point in the vector X and at time t.
    Returns the solution as a numVars by nx+1 array. The i-th row contains
    the values for the i-th variable at all grid points. The j-th column
    contains all variables at grid point j.
    '''
    pass