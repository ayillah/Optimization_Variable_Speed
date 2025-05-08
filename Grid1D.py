import numpy as np

class Grid1D:
  def __init__(self, nx=128, xMin=0.0, xMax=1.0):
    self.nx = nx
    self.xMin = xMin
    self.xMax = xMax
    self.X = np.linspace(self.xMin, self.xMax, self.nx+1)
    self.dx = (self.xMax - self.xMin)/self.nx





