import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from Grid1D import Grid1D
from SpeedFunc import (ConstantSpeedFunction, ThreeZoneSpeedFunction, 
    InterpolatedSpeedFunction)
from macCormack_stepper_StateEquationInterp import StateEquationInterp


class MacCormackStepperInterp:
    """
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
    """

    def __init__(self, grid=Grid1D(), epsilon=0.25, model=None):
        """Initialize with optimized parameters for second-order convergence"""
        assert model != None, "please supply a model to the stepper"
        self.grid = grid
        self.model = model
        self.epsilon = epsilon
        self.history = []

    def run(self, t_init, t_final, u_init):
        """
        Run MacCormack's method on the model contained in self.model, from
        initial time t_init to final time t_final, with initial values u_init.
        """

        # Allocate storage for the previous solution u_prev, the intermediate
        # (predicted by MacCormack's predictor step) solution u_pred, and the
        # solution at the end of the current timestep, u_cur.
        #
        # Each of these will be numVars by nx+1 arrays. The i-th row contains
        # the values for the i-th variable at all grid points. The j-th column
        # contains all variables at grid point j.

        u_prev = np.zeros((self.model.numVars, self.grid.nx + 1))
        u_pred = np.zeros((self.model.numVars, self.grid.nx + 1))
        u_cur_interp = np.zeros((self.model.numVars, self.grid.nx + 1))

        # Copy the initial value into the current value array. Use
        # np.copyto(destination, source) to do the copy, thereby avoiding an
        # extra array allocation. Also, set the current time to t_init.
        # TODO: safety check to ensure u_init.shape is [numVars, nx+1] before
        # trying to run anything.

        np.copyto(u_cur_interp, u_init)
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
        self.history.append((t, deepcopy(u_cur_interp)))
        
        # -------------------------------------------------------------------
        #        Main MacCormack stepping loop
        # -------------------------------------------------------------------
        while t < t_final:
            n_steps = n_steps + 1

            # Deep copy the current solution into the previous solution array
            np.copyto(u_prev, u_cur_interp)

            # Find the CFL compliant stepsize
            # divide by the maximum of the absolute value of c
            c_max = 0.0
            for x in (X):
                speeds = self.model.c(x)
                abs_of_speeds = np.fabs(speeds)
                local_speed_max = np.amax(abs_of_speeds)
                c_max = max(c_max, local_speed_max)
        
            dt = self.epsilon * (dx / c_max)

            # Adjust the time step in case we are about to hit the final time 
            if t_final - t <= dt:
                dt = t_final - t
            
            # Update the time t_step to the end of the time step
            t_step = t + dt

            #Predictor step
            for i in range(1, nx):
                dudx = (u_prev[:, i+1] - u_prev[:, i])
                c = self.model.c(X[i])
                u_pred[:, i] = u_prev[:, i] - (dt / dx) * c * dudx 

            # Apply left boundary condition
            u_cur_interp[:, 0] = np.transpose(model.applyLeftBC(X[0], t_step, dx, dt, u_prev))
            
            np.copyto(u_pred[:, 0], u_cur_interp[:, 0])

            # Corrector step
            for i in range(1, nx):
                dudx = (u_pred[:, i] - u_pred[:, i-1])
                c = self.model.c(X[i])
                uMid = 0.5*(u_pred[:,i] + u_prev[:,i])
                u_cur_interp[:,i] = uMid - 0.5 * (dt / dx) * c * dudx

            # Apply right boundary condition with improved extrapolation
            u_cur_interp[:, -1] = np.transpose(model.applyRightBC(X[-1], t_step, dx, dt, u_prev))
           
            # Update time and store results
            t = t_step

            self.history.append((t, deepcopy(u_cur_interp)))

        # -------------- End main MacCormack loop -------------------


if __name__ == "__main__":
    # save_u_star_history.py
    import numpy as np
    from Grid1D import Grid1D
    from macCormack_stepper_StateEquationInterp import MacCormackStepperInterp
    from state_equation_interp import StateEquationInterp
    from SpeedFunc import ConstantSpeedFunction, InterpolatedSpeedFunction, ThreeZoneSpeedFunction

    t_init = 0.0
    t_final = 3.0
    base_speed_func = ThreeZoneSpeedFunction()

    for nx in (16, 32, 64, 128, 256, 512, 1024):
        grid = Grid1D(nx=nx)
        speed_func = InterpolatedSpeedFunction(nx=nx, speed_to_interpolate=base_speed_func)
        model = StateEquationInterp(nx=nx, speed_func=speed_func)

        stepper = MacCormackStepperInterp(grid=grid, model=model)
        uInit = model.exact_solution(grid.X, t_init)

        stepper.run(t_init, t_final, uInit)
        print(f"Done run for nx = {nx}")

        # Save in compatible format
        times = np.array([t for t, _ in stepper.history])
        solutions = np.array([u[np.newaxis, :] if u.ndim == 1 else u for _, u in stepper.history])
        
        np.savez(f"u_star_history_nx_{nx}.npz", times=times, solutions=solutions)


