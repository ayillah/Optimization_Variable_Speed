import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from Grid1D import Grid1D
from state_equation import StateEquation
from adjoint_equation import AdjointEquation
from gradients import GradientCalculator
from macCormack_stepper_StateEquation import MacCormackStepper
from modified_macCormack_stepper_AdjointEquation import ModifiedMacCormackStepper

class PDEOptimizer:
    """
    PDE-constrained optimization solver that finds the optimal wave speed c(x)
    to minimize the cost functional:
    
    F(u,c) = 0.5 * integral integral rho(x,t)*[u(x,t)-u*(x,t)]^2 dxdt + 0.5*Xi*T*integral c_x^2 dx
    
    subject to the PDE constraint:
    u_t + c(x)u_x = 0
    """
    def __init__(self, nx, T=1.0, Xi=1.0, max_iterations=150, tol=1e-6, verbose=True):
        """
        Initialize the PDE optimizer.
        
        Parameters:
        -----------
        nx : int
            Number of spatial grid points (excluding endpoints)
        T : float
            Final time
        Xi : float
            Regularization parameter for the wave speed gradient term
        max_iterations : int
            Maximum number of optimization iterations
        tol : float
            Convergence tolerance for gradient norm
        verbose : bool
            Whether to print progress information
        """
        # Store parameters
        self.nx = nx
        self.T = T
        self.Xi = Xi
        self.max_iterations = max_iterations
        self.tol = tol
        self.verbose = verbose
        
        # Initialize the spatial grid
        self.grid = Grid1D(nx=nx)
        
        # Set the original wave speed c(x)
        self.model = StateEquation()
        self.original_c = np.array([self.model.c(x) for x in self.grid.X])

        # Set initial wave speed c(x)
        self.initial_c = 1 - 0.5 * np.cos(np.pi * self.grid.X)  
        
        # Initialize optimization history trackers
        self.objective_history = []
        self.gradient_norm_history = []
        self.c_history = []
        self.iteration = 0
        self.current_objective = None
        self.current_gradient_norm = None
        
        # Create model instances for state and adjoint equations
        self.state_model = StateEquation()
        self.adjoint_model = AdjointEquation()
        
        # Load precomputed target solution
        try:
            target_data = np.load(f"u_star_history_nx_{nx}.npz")
            self.target_times = target_data['times']
            self.target_solutions = target_data['solutions']
            if self.verbose:
                print(f"Successfully loaded precomputed target solution with shape {self.target_solutions.shape}")
        except Exception as e:
            if self.verbose:
                print(f"Error loading target solution: {e}")
                print("Will fall back to generating target solution if needed")
            self.target_times = None
            self.target_solutions = None
    
    def create_wave_speed_interpolator(self, c_vec):
        """Create a smooth interpolator for the wave speed function"""
        c_safe = np.maximum(c_vec, 1e-4)  # Ensure positive wave speed with safer minimum
        return interp1d(
            self.grid.X, 
            c_safe, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(c_safe[0], c_safe[-1])
        )
        
    def solve_state_equation(self, c):
        """
        Solve the state equation u_t + c(x)u_x = 0
        
        Parameters:
        -----------
        c : ndarray
            Wave speed parameter vector c(x)
            
        Returns:
        --------
        ndarray : State solution u(x,t)
        ndarray : Time points
        """
        print('Solving state equation')
        # Create a temporary StateEquation model with the provided c(x)
        class TemporaryStateModel(StateEquation):
            def __init__(self, c_interp):
                super().__init__()
                self.c_interp = c_interp
                
            def c(self, x):
                return float(self.c_interp(x))
        
        # Create interpolator for wave speed
        c_interp = self.create_wave_speed_interpolator(c)
        temp_model = TemporaryStateModel(c_interp)
        
        # Create MacCormack stepper
        stepper = MacCormackStepper(grid=self.grid, model=temp_model)
        
        # Set initial condition and solve
        t_init = 0.0
        u_init = temp_model.exact_solution(self.grid.X, t_init)
        stepper.run(t_init, self.T, u_init)
        
        # Extract solution history
        Nt = len(stepper.history)
        u = np.zeros((Nt, temp_model.numVars, self.grid.nx + 1))
        times = np.zeros(Nt)
        
        for i, (t, soln) in enumerate(stepper.history):
            times[i] = t
            u[i] = soln
            
        return u, times
    
    def get_target_solution(self, u, times_u):
        """
        Get precomputed target solution u* or generate it if not available
        
        Parameters:
        -----------
        u : ndarray
            State solution (used as fallback if precomputed not available)
        times_u : ndarray
            Time points for u
            
        Returns:
        --------
        ndarray : Target solution u* interpolated to match u's time grid
        """
        if self.target_times is not None and self.target_solutions is not None:
            # Interpolate precomputed target solution to match u's time grid
            u_star = np.zeros_like(u)
            
            for var_idx in range(u.shape[1]):
                for x_idx in range(u.shape[2]):
                    f = interp1d(self.target_times, self.target_solutions[:, var_idx, x_idx], 
                                kind='linear', bounds_error=False, fill_value="extrapolate")
                    u_star[:, var_idx, x_idx] = f(times_u)
            
            return u_star
        else:
            # If precomputed solution is not available, use the original wave speed to generate it
            if self.verbose:
                print("Generating target solution using original wave speed")
            u_star, _ = self.solve_state_equation(self.original_c)
            return u_star
    
    def solve_adjoint_equation(self, u, u_star, c, times_u):
        """
        Solve the adjoint equation for lambda
        
        Parameters:
        -----------
        u : ndarray
            State solution
        u_star : ndarray
            Target solution
        c : ndarray
            Wave speed parameter vector
        times_u : ndarray
            Time points for u solution
            
        Returns:
        --------
        ndarray : Adjoint solution lambda(x,t)
        ndarray : Time points for lambda solution
        """
        print("Solving adjoint equation")
        # Create a temporary AdjointEquation model
        class TemporaryAdjointModel(AdjointEquation):
            def __init__(self, c_interp):
                super().__init__()
                self.c_interp = c_interp
                
            def c(self, x):
                return float(self.c_interp(x))
        
        # Create interpolator for wave speed
        c_interp = self.create_wave_speed_interpolator(c)
        temp_model = TemporaryAdjointModel(c_interp)
        
        # Save solutions for adjoint solver (required by the interface)
        np.save(f"aligned_interpolated_solutions_nx_{self.nx}.npy", u)
        np.save(f"aligned_interpolated_interp_solutions_star_nx_{self.nx}.npy", u_star)
        
        # Create stepper for adjoint equation
        stepper = ModifiedMacCormackStepper(grid=self.grid, model=temp_model)
        
        # Set final condition (adjoint equation runs backward in time)
        t_init = 0.0  # Starting time for backward simulation
        lambda_init = np.zeros((temp_model.numVars, self.grid.nx + 1))
        
        # Run the adjoint simulation backward in time
        stepper.run(t_init, self.T, lambda_init)
        
        # Extract solution history
        Nt_lam = len(stepper.history)
        lam = np.zeros((Nt_lam, temp_model.numVars, self.grid.nx + 1))
        times_lam = np.zeros(Nt_lam)
        
        for i, (t, soln) in enumerate(stepper.history):
            times_lam[i] = t
            lam[i] = soln
            
        return lam, times_lam

    def interpolate_to_common_grid(self, u, u_star, lam, times_u, times_lam):
        """
        Ensure all solutions are on the same time grid for accurate calculations
        """
        # Define common time grid
        Nt_common = max(len(times_u), len(times_lam))
        times_common = np.linspace(0, self.T, Nt_common)
        
        # Initialize interpolated arrays
        u_interp = np.zeros((Nt_common, u.shape[1], u.shape[2]))
        u_star_interp = np.zeros_like(u_interp)
        lam_interp = np.zeros((Nt_common, lam.shape[1], lam.shape[2]))
        
        # Interpolate u to common time grid
        for i in range(u.shape[1]):
            for j in range(u.shape[2]):
                f = interp1d(times_u, u[:, i, j], kind='linear', fill_value="extrapolate")
                u_interp[:, i, j] = f(times_common)
        
        # Interpolate u_star to common time grid
        for i in range(u_star.shape[1]):
            for j in range(u_star.shape[2]):
                f = interp1d(times_u, u_star[:, i, j], kind='linear', fill_value="extrapolate")
                u_star_interp[:, i, j] = f(times_common)
        
        # Interpolate λ to common time grid
        for i in range(lam.shape[1]):
            for j in range(lam.shape[2]):
                f = interp1d(times_lam, lam[:, i, j], kind='linear', fill_value="extrapolate")
                lam_interp[:, i, j] = f(times_common)
            
        return u_interp, u_star_interp, lam_interp, times_common
        
    def build_derivative_matrix(self, N, h):
        """
        Build a finite difference matrix for the spatial derivative d/dx
        """
        D = np.zeros((N, N))
        
        # Interior points: central difference
        for j in range(1, N - 1):
            D[j, j - 1] = -1 / (2 * h)
            D[j, j + 1] = 1 / (2 * h)

        # Left boundary: second-order forward difference
        D[0, 0] = -3 / (2 * h)
        D[0, 1] = 4 / (2 * h)
        D[0, 2] = -1 / (2 * h)

        # Right boundary: second-order backward difference
        D[-1, -3] = 1 / (2 * h)
        D[-1, -2] = -4 / (2 * h)
        D[-1, -1] = 3 / (2 * h)

        return D
        
    def compute_objective(self, c):
        """
        Compute the objective function value that includes both:
        1. Data misfit between computed and target solutions
        2. Regularization term for smoothness of c(x)
        """
        print("computing objective")
        # Enforce minimum values for stability
        c = np.maximum(c, 1e-4)
        
        try:
            # Get state solution for current c
            u, times_u = self.solve_state_equation(c)
            
            # Get target solution (from precomputed data)
            u_star = self.get_target_solution(u, times_u)
            
            # Get adjoint solution for current state and target
            lam, times_lam = self.solve_adjoint_equation(u, u_star, c, times_u)
            
            # Ensure all solutions are on a common time grid
            u_interp, u_star_interp, _, times_common = self.interpolate_to_common_grid(
                u, u_star, lam, times_u, times_lam)
            
            # Set up quadrature weights for time and space integration
            Nt = u_interp.shape[0]
            dt = self.T / (Nt - 1)
            q_t = np.ones(Nt)
            q_t[0] = q_t[-1] = 0.5  # Trapezoidal rule endpoints
            q_t *= dt
            
            q_x = np.ones(self.nx + 1)
            q_x[0] = q_x[-1] = 0.5  # Trapezoidal rule endpoints
            q_x *= self.grid.dx
            
            # Compute data misfit term (assuming rho(x,t)=1)
            diff_sq = np.sum((u_interp - u_star_interp) ** 2, axis=1)
            misfit = 0.5 * np.sum(q_t[:, np.newaxis] * diff_sq * q_x[np.newaxis, :])
            
            # Compute regularization term (0.5*Xi*T*integral c_x^2 dx)
            D = self.build_derivative_matrix(self.nx + 1, self.grid.dx)
            c_x = np.dot(D, c)
            reg = 0.5 * self.Xi * self.T * np.sum(q_x * c_x ** 2)
            
            # Total objective
            objective = misfit + reg 
            
            # Store current value and update history
            self.current_objective = objective
            self.objective_history.append(objective)
            
            return objective
            
        except Exception as e:
            if self.verbose:
                print(f"Error in objective computation: {e}")
            # Return a high value to steer optimizer away from problematic areas
            return 1.0e10
        
    def compute_gradient(self, c):
        """
        Compute the gradient of the objective function with respect to c(x)
        """
        print("Computing gradient")
        # Enforce minimum values for stability
        c = np.maximum(c, 1e-4)
        
        # Store c in history
        self.c_history.append(c.copy())
        
        try:
            # Get solutions for current c
            u, times_u = self.solve_state_equation(c)
            u_star = self.get_target_solution(u, times_u)
            lam, times_lam = self.solve_adjoint_equation(u, u_star, c, times_u)
            
            # Interpolate to common time grid
            u_interp, u_star_interp, lambda_interp, times_common = self.interpolate_to_common_grid(
                u, u_star, lam, times_u, times_lam)
            
            # Use the GradientCalculator to compute gradient
            calculator = GradientCalculator(
                u=u_interp, 
                lam=lambda_interp, 
                c=c,
                times_u=times_common, 
                times_lam=times_common,
                T=self.T, 
                Xi=self.Xi, 
                grid=self.grid
            )
            
            # Calculate gradient at each point
            gradient = np.zeros(self.nx + 1)
            for j in range(self.nx + 1):
                grad_j = calculator.compute_gradient(j)
                gradient[j] = grad_j[0]  # Take first component if multi-variable
            
            # Add gradient component for deviation from original wave speed
            q_x = np.ones(self.nx + 1)
            q_x[0] = q_x[-1] = 0.5  # Trapezoidal rule endpoints
            q_x *= self.grid.dx
            
            # Add small regularization to gradient to improve stability
            D = self.build_derivative_matrix(self.nx + 1, self.grid.dx)
            D_T = D.T
            gradient += 1e-6 * np.dot(D_T, np.dot(D, c))
            
            # Apply gradient scaling to improve convergence
            gradient_scale = max(1.0, np.linalg.norm(gradient))
            gradient = gradient / gradient_scale
            
            # Store gradient norm and update history
            self.current_gradient_norm = np.linalg.norm(gradient)
            self.gradient_norm_history.append(self.current_gradient_norm)
            
            return gradient
            
        except Exception as e:
            if self.verbose:
                print(f"Error in gradient computation: {e}")
            # Return zero gradient to avoid crashes
            zero_grad = np.zeros_like(c)
            self.current_gradient_norm = 0.0
            self.gradient_norm_history.append(0.0)
            return zero_grad
    
    def callback(self, c_current):
        """Callback function for the optimizer to track progress"""
        # Print progress information
        if self.verbose and (self.iteration % 5 == 0 or self.iteration == self.max_iterations - 1):
            print(f"Iteration {self.iteration}: Objective = {self.current_objective:.6e}, "
                  f"Gradient norm = {self.current_gradient_norm:.6e}")
            
            # Calculate and print current error between optimized and original wave speed
            c_error = np.linalg.norm(c_current - self.original_c) / np.linalg.norm(self.original_c)
            print(f"Relative error in wave speed: {c_error:.6e}")
        
        # Check for abnormal values that might cause optimization failures
        if np.isnan(self.current_objective) or np.isinf(self.current_objective) or \
           np.isnan(self.current_gradient_norm) or np.isinf(self.current_gradient_norm):
            if self.verbose:
                print("Warning: NaN or Inf detected in optimization values")
            return True  # Stop optimization if NaN or Inf values are detected
            
        self.iteration += 1

        return False
    
    def optimize(self):
        """
        Run the L-BFGS-B optimization to find the optimal c(x)
        """
        # Reset counters and history
        self.iteration = 0
        self.objective_history = []
        self.gradient_norm_history = []
        self.c_history = []
        
        # Set up bounds for c (must be positive for physical meaning)
        lower_bounds = np.ones(self.nx + 1) * 1e-3  # Increased minimum bound for stability
        upper_bounds = np.ones(self.nx + 1) * 5.0    # Reduced maximum bound for stability
        bounds = list(zip(lower_bounds, upper_bounds))
        
        if self.verbose:
            print(f"Starting optimization with nx = {self.nx}, Xi = {self.Xi}")
            #print(f"Speed penalty weight = {self.speed_penalty_weight}")
        
        try:
            # Run L-BFGS-B optimization with improved parameters
            result = minimize(
                fun=self.compute_objective,
                x0=self.initial_c,
                method='L-BFGS-B',
                jac=self.compute_gradient,
                bounds=bounds,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tol,  
                    'gtol': self.tol,  
                    'maxfun': 1000,
                    'maxls': 50,                # Increased max line search steps
                    'maxcor': min(10, self.nx), # Increase memory size for BFGS approximation
                    'disp': self.verbose
                },
                callback=self.callback
            )
        except Exception as e:
            if self.verbose:
                print(f"Optimization failed with error: {e}")
            # Create a minimal result object with the best available data
            if len(self.c_history) > 0:
                # Use the best solution found before error
                best_idx = np.argmin(self.objective_history) if self.objective_history else -1
                c_optimal = self.c_history[best_idx] if best_idx >= 0 else self.initial_c
            else:
                c_optimal = self.initial_c
                
            result = type('obj', (object,), {
                'x': c_optimal,
                'success': False,
                'message': f"Optimization failed with error: {e}",
                'nfev': len(self.objective_history),
                'njev': len(self.gradient_norm_history)
            })
        
        # Extract optimal solution
        c_optimal = result.x
        
        # Ensure all values are physically meaningful
        c_optimal = np.maximum(c_optimal, 1e-3)
        
        # Save the result
        np.save(f"optimized_c_lbfgsb_nx_{self.nx}_fixed.npy", c_optimal)
        
        # Collect results for analysis
        results = {
            "c_optimal": c_optimal,
            "objective_history": self.objective_history,
            "gradient_norm_history": self.gradient_norm_history,
            "c_history": self.c_history,
            "iterations": len(self.objective_history),
            "converged": result.success,
            "message": result.message,
            "function_evaluations": result.nfev,
            "gradient_evaluations": getattr(result, 'njev', len(self.gradient_norm_history))
        }
        
        return c_optimal, results
    
    def plot_results(self, c_optimal, results):
        """Visualize optimization results"""
        if len(results["objective_history"]) <= 1:
            if self.verbose:
                print("Not enough history data to create meaningful plots")
            return
            
        # Add grid size information to plots
        grid_size_text = f"Grid Size: nx = {self.nx}, Xi = {self.Xi}"
        
        # 1. Optimization convergence plots
        fig1 = plt.figure(figsize=(12, 8))
        fig1.suptitle(grid_size_text, fontsize=16, y=0.98)
            
        # Plot objective function history
        plt.subplot(2, 2, 1)
        plt.semilogy(results["objective_history"])
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function")
        plt.title("Objective Function History")
            
        # Plot gradient norm history
        plt.subplot(2, 2, 2)
        plt.semilogy(results["gradient_norm_history"])
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Norm History")
            
        # Plot initial vs optimized vs original wave speed
        plt.subplot(2, 2, 3)
        plt.plot(self.grid.X, self.initial_c, 'b-', label='Initial c(x)')
        plt.plot(self.grid.X, c_optimal, 'ro', label='Optimized c(x)')
        plt.plot(self.grid.X, self.original_c, 'g--', label='Original c(x)')
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("c(x)")
        plt.title("Wave Speed Parameter Comparison")
        plt.legend()
            
        # Plot evolution of c(x) during optimization
        plt.subplot(2, 2, 4)
        iterations_to_plot = min(10, len(self.c_history))
        indices = np.linspace(0, len(self.c_history)-1, iterations_to_plot).astype(int)
            
        for i, idx in enumerate(indices):
            alpha = 0.3 + 0.7 * i / (iterations_to_plot - 1)
            plt.plot(self.grid.X, self.c_history[idx], alpha=alpha, 
                     label=f'Iter {idx}' if i % 3 == 0 else None)
        
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("c(x)")
        plt.title("Evolution of c(x) During Optimization")
        plt.legend()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(f"optimization_results_lbfgsb_nx_{self.nx}_fixed.pdf", dpi=150)
        plt.show()
        
        # 2. Solution comparison plots at multiple time points
        # Get solutions with optimized c
        u_opt, times_u_opt = self.solve_state_equation(c_optimal)
        u_star_opt = self.get_target_solution(u_opt, times_u_opt)
        
        # Create figure for solution comparison at multiple time points
        fig2 = plt.figure(figsize=(15, 10))
        fig2.suptitle(f"Solution Comparison at Various Times - {grid_size_text}", fontsize=16, y=0.98)
        
        # Time points to plot
        time_points = [0.0, 0.75, 1.5, 3.0]
        
        for i, target_time in enumerate(time_points):
            # Find closest time point in solution
            t_idx = np.argmin(np.abs(times_u_opt - target_time))
            actual_time = times_u_opt[t_idx]
            
            # Plot state vs target at this time
            plt.subplot(2, 2, i+1)
            plt.plot(self.grid.X, u_opt[t_idx, 0, :], 'b--', label='State u')
            plt.plot(self.grid.X, u_star_opt[t_idx, 0, :], 'r-', label='Target u*')
            plt.plot(self.grid.X, u_opt[t_idx, 0, :] - u_star_opt[t_idx, 0, :], 'g-.', label='Difference')
            plt.grid(True)
            plt.xlabel("x")
            plt.ylabel("Solution")
            plt.title(f"Solution at t={actual_time:.3f}")
            plt.legend()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(f"solution_comparison_lbfgsb_nx_{self.nx}_fixed.pdf", dpi=150)
        plt.show()


# Main execution block
if __name__ == "__main__":
    # Grid resolutions to test
    nx_values = [16, 32, 64, 128, 256, 512, 1024]  # Focus on the problematic high-resolution case
    
    # Regularization parameter values to test
    Xi_values = [0.01]  # Use the provided XI value
    
    for nx in nx_values:
        for Xi in Xi_values:
            print(f"\n{'='*50}")
            print(f"Starting optimization for nx = {nx}, Xi = {Xi}")
            print(f"{'='*50}")
        
            # Create optimizer with current grid resolution
            optimizer = PDEOptimizer(
                nx=nx,
                T=3.0,           # Final time
                Xi=Xi,           # Regularization parameter
                max_iterations=150,  # Increased maximum iterations
                tol=1e-7,           # Tighter tolerance
                verbose=True
            )
        
            # Run optimization
            c_optimal, results = optimizer.optimize()
        
            # Print summary of optimization results
            print(f"\nOptimization completed for nx = {nx}")
            print(f"Final objective: {results['objective_history'][-1]:.6e}")
            print(f"Final gradient norm: {results['gradient_norm_history'][-1]:.6e}")
            print(f"Iterations: {results['iterations']}")
            print(f"Converged: {results['converged']}")
            
            # Calculate and print error between optimized and original wave speed
            c_error = np.linalg.norm(c_optimal - optimizer.original_c) / np.linalg.norm(optimizer.original_c)
            print(f"Relative error in wave speed: {c_error:.6e}")

            # Plot optimization results
            optimizer.plot_results(c_optimal, results)