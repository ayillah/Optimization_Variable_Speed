import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from copy import deepcopy
from OutputHandler import OutputHandler
from Grid1D import Grid1D



class MacCormackStepper:
    '''
    MacCormackStepper drives a run of MacCormack's method. The results 
    are stored in a list "history", each entry in which is a tuple 
    (time, solution). 

    Attributes of this class are:
    *) grid -- a Grid1D object that stores spatial discretization information
    *) model -- an object that evaluates the Jacobian and applies 
       boundary conditions for the PDE being solved
    *) epsilon -- safety factor for CFL condition on timestep. The timestep 
       will be dt = epsilon * dx / max(lambda), where max(lambda) is the max
       eigenvalue of the Jacobians at all grid points. Defaults to 0.25.
    *) history -- list of (time, solution) tuples filled in during the call to
       run(). 
    '''
    def __init__(self, grid=Grid1D(), epsilon=0.25, model=None):
      '''
      Constructor. Takes the grid, model, and epsilon as keyword arguments. 
      Call as, e.g., 
          stepper = MacCormackStepper(grid=myGrid, epsilon=0.25, model=myModel)
      '''
      assert model != None, 'please supply a model to the stepper'

      # The only thing that needs doing is to record the grid, model, and 
      # epsilon, and then to create an empty history list.
      self.grid = grid
      self.model = model
      self.epsilon = epsilon
      self.history = []

    def run(self, t_init, t_final, u_init): 
      '''
      Run MacCormack's method on the model contained in self.model, from 
      initial time t_init to final time t_final, with initial values u_init.
      '''
    
      # Allocate storage for the previous solution u_prev, the intermediate 
      # (predicted by MacCormack's predictor step) solution u_pred, and the 
      # solution at the end of the current timestep, u_cur.
      #
      # Each of these will be numVars by nx+1 arrays. The i-th row contains
      # the values for the i-th variable at all grid points. The j-th column
      # contains all variables at grid point j.

      u_prev = np.zeros((self.model.numVars, self.grid.nx+1))
      u_pred = np.zeros((self.model.numVars, self.grid.nx+1))
      u_cur = np.zeros((self.model.numVars, self.grid.nx+1))

  

      # Copy the initial value into the current value array. Use 
      # np.copyto(destination, source) to do the copy, thereby avoiding an 
      # extra array allocation. Also, set the current time to t_init.
      # TODO: safety check to ensure u_init.shape is [numVars, nx+1] before
      # trying to run anything. 

      np.copyto(u_cur, u_init)
      t = t_init

      # For convenience, dereference a few commonly used variables to save 
      # typing self.grid. all the time. 
      nx = self.grid.nx
      dx = self.grid.dx
      X = self.grid.X
      model = self.model

      # -------------------------------------------------------------------
      # Prepare for the run: initialize n_steps to zero, clear the history
      # list, and deep copy the current (initial) (time, soln) tuple into the
      # history list.  

      n_steps = 0
      self.history.clear()
      self.history.append((t, deepcopy(u_cur)))
      
      # -------------------------------------------------------------------
      #        Main MacCormack stepping loop
      # -------------------------------------------------------------------
      while t < t_final:

         n_steps = n_steps + 1

         # Deep copy the current solution into the previous solution array
         np.copyto(u_prev, u_cur)  
          
         # Find the CFL compliant stepsize
         # divide by the maximum of the absolute value of c
         c_max = 0.0
         for x in (X):
            speed = model.c(x)
            abs_of_speeds = np.fabs(speed)
            local_speed_max = np.amax(abs_of_speeds)
            c_max = max(c_max, local_speed_max)

         dt = self.epsilon * (dx / c_max)

         # Adjust the time step in case we are about to hit the final time 
         if t_final - t <= dt:
            dt = t_final - t
          
         # Update the time t_step to the end of the time step
         t_step = t + dt

         # Predictor step
         for i in range(1, nx):
            dudx = (u_prev[:, i+1] - u_prev[:, i])
            c = model.c(X[i])
            u_pred[:,i] = u_prev[:,i] - (dt / dx) * c * dudx
               
         # Apply left boundary condition
         u_cur[:, 0] = np.transpose(model.applyLeftBC(X[0], t_step, dx, dt, u_prev))
          
         np.copyto(u_pred[:,0],u_cur[:, 0])

         # Corrector step
         for i in range(1, nx):
            dudx = (u_pred[:, i] - u_pred[:, i-1])
            c = self.model.c(X[i])
            uMid = 0.5*(u_pred[:,i] + u_prev[:,i])
            u_cur[:,i] = uMid - 0.5 * (dt / dx) * c * dudx

         # Apply right boundary condition 
         u_cur[:, -1] = np.transpose(model.applyRightBC(X[-1], t_step, dx, dt, u_prev))

         # Update the time
         t = t_step

         # Store the results
         self.history.append((t, deepcopy(u_cur)))

      # -------------- End main MacCormack loop -------------------
   
  
          
if __name__=='__main__':
  from state_equation import StateEquation
  from error_rate import ErrorRate
  from FrameWritingOutputHandler import FrameWritingOutputHandler

  errors = []
  grid_sizes = []

  # Loop over grid size
  for nx in (16, 32, 64, 128, 256, 512, 1024):
   grid = Grid1D(nx=nx)

   model = StateEquation()

   stepper = MacCormackStepper(grid=grid, model=model)

   # Set initial value to that of the known solution 
   t_init = 0.0
   uInit = model.exact_solution(grid.X, t_init)

   # Run the simulation from t_init up to t_final
   t_final = 1.0
   stepper.run(t_init, t_final, uInit)
 
   print('done run for nx=', nx)

   doPlots = False # Change this to True to dump plots for all timesteps. That 
   # will be slow!

   # Now we'll loop over the history list, obtaining the error at each
   # timestep and plotting both the solution and the error. We also find
   # the maximum of the errors taken over all timesteps; that should vary as
   # max_err = M dx^2, where M is some constant. 

   max_err = 0.0
   for i,v in enumerate(stepper.history):
      t = v[0]
      W = v[1]
      Wex = model.exact_solution(grid.X, t)
      err = Wex - W
      err_norm = np.linalg.norm(err, 1)/nx/2
      max_err = max(err_norm, max_err)

      if doPlots:
         fig, ax = plt.subplots()
         ax.plot(grid.X, err[0,:], 'b-')
         ax.set(ylim=[-3,3])
         plt.savefig('err-{}-{}.pdf'.format(nx,i))
         plt.close()

         fig, ax = plt.subplots()
         ax.plot(grid.X, W[0,:], 'b-', label='u')
         ax.plot(grid.X, Wex, 'r-', label='u_extact')
         ax.set(ylim=[-4,4])
         ax.legend()
         plt.savefig('soln-{}-{}.pdf'.format(nx,i))
         plt.close()

   print('nx={}, max_err={}'.format(nx, max_err))

   errors.append(max_err)
   grid_sizes.append(nx)

  # Compute the error rate
  error_rate = ErrorRate(grid_sizes, errors)
  p = error_rate.error_rate()

  print("Error rate = {:6f}".format(p))

  plt.loglog(grid_sizes, errors, "r-o")
  plt.xlabel("log(Nx)")
  plt.ylabel("log(error_norms)")
  plt.grid()
  plt.show()
