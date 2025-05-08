import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from copy import deepcopy
from OutputHandler import OutputHandler
from Grid1D import Grid1D

class ModifiedMacCormackStepper:
    '''
    ModifiedMacCormackStepper drives a run of MacCormack's method. The results 
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

        self.grid = grid
        self.model = model
        self.epsilon = epsilon
        self.history = []
        
    def run(self, t_init, T, lambda_init): 
        '''
        Run MacCormack's method on the model contained in self.model, from 
        initial time t_init to final time t_final, with initial values u_init.
        '''

        # Allocate storage for the previous solution u_prev, the intermediate 
        # (predicted by MacCormack's predictor step) solution u_pred, and the 
        # solution at the end of the current timestep, u_cur.
        lambda_prev = np.zeros((self.model.numVars, self.grid.nx+1))
        lambda_pred = np.zeros((self.model.numVars, self.grid.nx+1))
        lambda_cur = np.zeros((self.model.numVars, self.grid.nx+1))

        # Load interpolated u and u_star
        u = np.load(f"aligned_interpolated_solutions_nx_{self.grid.nx}.npy")    # shape: (Nt, vars, nx+1)
        u_star = np.load(f"aligned_interpolated_interp_solutions_star_nx_{self.grid.nx}.npy")    # shape: (Nt, vars, nx+1)

        # Check shape compatibility
        assert u.shape == u_star.shape, "Mismatch between u and u_star"
        Nt = u.shape[0]

        # Initialize the solution at time t_init (final time T)
        np.copyto(lambda_cur, lambda_init)
        t = t_init
        s = T - t

        # For convenience, dereference a few commonly used variables
        nx = self.grid.nx
        dx = self.grid.dx
        X = self.grid.X
        model = self.model

        # Initialize lists to store data for all timesteps
        all_data = []

        # Prepare for the run: initialize n_steps to zero, clear history, and append initial state
        n_steps = 0
        self.history.clear()
        self.history.append(((T - s), deepcopy(lambda_cur)))

        time_index = 0

        # Main MacCormack stepping loop
        while (T - s) < T and time_index < Nt:
            # Increment the time step index
            n_steps = n_steps + 1

            # Deep copy the current solution into the previous solution array
            np.copyto(lambda_prev, lambda_cur)  

            # Find the CFL compliant stepsize
            c_max = 0.0
            for x in (X):
                speed = model.c(x)
                abs_of_speeds = np.fabs(speed)
                local_speed_max = np.amax(abs_of_speeds)
                c_max = max(c_max, local_speed_max)

            dt = self.epsilon * (dx / c_max)
            ds = -dt

            # Adjust the time step in case we are about to hit the final time 
            if s <= -ds:
                ds = -s

            # Update the time t_step to the end of the time step
            t_step = (T - s) - ds

            # Define the explicit and inplicit blending parameter lambda
            lambda_blending = max(0.5 * (1 + (dx / ds)), 0.0)

            # Time-sliced values for current time step
            u_current = u[time_index]
            u_star_current = u_star[time_index]

            # Construct coefficient matrix A for predictor step
            A = np.zeros((nx+1, nx+1))
            b_pred = np.zeros((self.model.numVars, nx + 1))

            for i in range(nx-1, 0, -1):
                c1 = model.c(X[i])
                c2 = model.c(X[i+1])

                A[i, i] = 1.0 - lambda_blending * (ds / dx) * c1   
                A[i, i+1] = lambda_blending * (ds / dx) * c2 

                dlam_dx = c2 * lambda_prev[:, i+1] - c1 * lambda_prev[:, i]
                dlam = u_current[:, i] - u_star_current[:, i]
                #print(f"dlam (i={i}): max={np.max(dlam)}, min={np.min(dlam)}")

                rho = -model.rho(X[i])

                b_pred[:, i] = lambda_prev[:, i] + (1 + lambda_blending) * (ds / dx) * dlam_dx - ds * rho * dlam

            # Apply left boundary condition (non-reflecting boundary condition at x = 0)
            A[0, 0] = 1.0
            b_pred[:, 0] = (lambda_prev[:, 0] + (ds/dx) * (3 * model.c(X[0]) * lambda_prev[:, 0]
                    - 4 * model.c(X[1]) * lambda_prev[:, 1] + model.c(X[2]) * lambda_prev[:, 2])/2
                     + ds * model.rho(X[0]) * (u_current[:, 0] - u_star_current[:, 0]))
            A[-1, -1] = 1.0
            b_pred[:, -1] = lambda_prev[:, -1]
         
            # Solve for lambda_pred
            lambda_pred = la.solve(A, b_pred.T).T

            # Construct coefficient matrix B for corrector step
            B = np.eye(nx + 1)
            b_cur = np.zeros((self.model.numVars, nx + 1))

            for i in range(1, nx):
                c3 = model.c(X[i])
                c4 = model.c(X[i-1])

                B[i, i] = 1.0 - lambda_blending * (ds / dx) * c3
                B[i, i-1] = 0.5 * lambda_blending * (ds / dx) * c4

                dlam_dx3 = c3 * (lambda_blending * (lambda_pred[:, i] + lambda_prev[:, i]) - lambda_pred[:, i])
                dlam_dx4 = c4 * (lambda_pred[:, i-1] - lambda_blending * lambda_prev[:, i-1])
                dlam = u_current[:, i] - u_star_current[:, i]
                #print(f"dlam (i={i}): max={np.max(dlam)}, min={np.min(dlam)}")

                rho = -model.rho(X[i])
                lambda_mid = 0.5 * (lambda_pred[:,i] + lambda_prev[:,i])

                b_cur[:, i] = lambda_mid - 0.5 * (ds / dx) * (dlam_dx3 + dlam_dx4) - 0.5 * ds * rho * dlam

            # Apply right boundary condition v(L, t) = 0 at x = L
            B[-1, -1] = 1.0
            b_cur[:, 0] = lambda_pred[:, 0]
            B[-1, -1] = 1.0
            b_cur[:, -1] = 0.0
          
            # Solve for lambda_cur
            lambda_cur = np.linalg.solve(B, b_cur.T).T

            # Decrease time step index
            s = T - t_step

            # Store the data for the current timestep
            all_data.append(deepcopy(lambda_cur))

            # Store the data
            self.history.append(((T - s), deepcopy(lambda_cur)))

        # Convert list of (numVars, nx + 1) arrays to a 3D Numpy array: (Nt, numVars, nx + 1)
        all_data_array = np.stack(all_data, axis=0)

        # Save the adjoint solution data
        np.save(f'adjoint_solutions_nx_{nx}.npy', all_data_array)

        # End main MacCormack loop
          
if __name__=='__main__':
    from adjoint_equation import AdjointEquation
    from FrameWritingOutputHandler import FrameWritingOutputHandler

    # Loop over grid size
    for nx in (16, 32, 64, 128, 256, 512, 1024):
        grid = Grid1D(nx=nx)

        model = AdjointEquation()

        stepper = ModifiedMacCormackStepper(grid=grid, model=model)

        # Set the initial time (start at the final time t = T)
        t_init = 0.0

        # Set the final time and the initial condition
        T = 1.0

        # Set the final condition as the initial condition, lambda(x, T) = 0
        lambdaInit = np.zeros((model.numVars, grid.nx + 1))  # lambda(x, T) = 0 at the initial time

        # Run the simulation backward from t_final to t_init
        stepper.run(t_init, T, lambdaInit)

        print('done run for nx=', nx)

        doPlots = False  # Change this to True to dump plots for all timesteps. That will be slow!

        # Now we'll loop over the history list, obtaining and plotting 
        # the solution at each time step
        for i, v in enumerate(stepper.history):
            t = v[0]
            s = T - t
            lambda_soln = v[1]
            
            if doPlots:
                fig, ax = plt.subplots()
                ax.plot(grid.X, lambda_soln[0,:], 'b-', label='lambda')
                ax.set(ylim=[-4, 4])
                ax.legend()
                plt.savefig(f'soln-{nx}-{i}.pdf')
                plt.close()

        #print(f'nx={nx}, lambda={lambda_soln[0,:]}')
