from abc import ABC, abstractmethod # ABC = "Abstract base class"

class PDEModel1D(ABC):
  def __init__(self, numVars): 
    self.numVars = numVars


  def numVars(self):
    return self.numVars


  @abstractmethod
  def c(self, x):
    '''Return the speed at position x'''
    pass


  @abstractmethod
  def f(self, x):
    
    pass


  @abstractmethod
  def applyLeftBC(self, x, t, dx, dt, u_prev):
    '''
    Do whatever is needed to obtain u_cur at the left boundary point.
    
    Arguments are:
      *) x -- location of left boundary point
      *) t -- current time (i.e., time at end of step)
      *) dx -- grid spacing dx = x_1 - x_0
      *) dt -- timestep size
      *) u_prev -- solution at previous timestep 
    '''
    pass

  
  @abstractmethod
  def applyRightBC(self, x, t, dx, dt, u_prev):
    '''
    Do whatever is needed to obtain u_cur at the right boundary point.
    
    Arguments are:
      *) x -- location of right boundary point
      *) t -- current time (i.e., time at end of step)
      *) dx -- grid spacing dx = x_{nx} - x_{nx-1}
      *) dt -- timestep size
      *) u_prev -- solution at previous timestep 
    '''
    pass

  
  

  

  
