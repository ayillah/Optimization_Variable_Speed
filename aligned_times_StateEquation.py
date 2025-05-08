import numpy as np
import matplotlib.pyplot as plt
from macCormack_stepper_StateEquationInterp import MacCormackStepperInterp
from Grid1D import Grid1D
from scipy.interpolate import interp1d
from state_equation_interp import StateEquationInterp

class MacCormackInterpolationAligned:
    def __init__(self, nx_values, t_init=0.0, T=1.0, num_interp_points=100):
        """Initialize the class with grid resolutions, time parameters, and interpolation settings."""
        self.nx_values = nx_values
        self.t_init = t_init
        self.T = T
        self.num_interp_points = num_interp_points

    def run_interpolation(self):
        """Run interpolation ensuring alignment between original and interpolated solutions."""
        for nx in self.nx_values:
            grid = Grid1D(nx=nx)
            model = StateEquationInterp()
            stepper = MacCormackStepperInterp(grid=grid, model=model)

            # Set initial condition
            lambda_init = model.exact_solution(grid.X, self.t_init)

            # Run MacCormack stepper
            stepper.run(self.t_init, self.T, lambda_init)

            # Extract original time steps and solutions
            original_times = [v[0] for v in stepper.history]  # t values
            transformed_times = [self.T - s for s in original_times]  # Ensure alignment with T - s
            original_solutions = [v[1] for v in stepper.history]  # Corresponding solution snapshots

            # Define perfectly aligned target times
            target_s_values = np.linspace(self.T, 0.0, num=self.num_interp_points)
            target_times = self.T - target_s_values  # Transformed for alignment
           
            # Perform interpolation only once for solutions
            interpolator = interp1d(transformed_times, original_solutions, axis=0, kind='linear', fill_value='extrapolate')
            interpolated_solutions = [interpolator(t) for t in target_times]

            # Save interpolated solutions
            interpolated_array = np.stack(interpolated_solutions, axis=0)   # shape: (num_times, num_vars, nx+1)
            np.save(f"aligned_interpolated_solutions_nx_{nx}.npy", interpolated_array)

            # Generate comparison plots
            self.generate_plots(grid, interpolated_solutions, original_solutions, nx, transformed_times, target_times)

            print(f"Interpolation and plots generated for nx={nx} with aligned time steps.")

    def generate_plots(self, grid, interpolated_solutions, original_solutions, nx, transformed_times, target_times):
        """Plot and compare original vs interpolated solutions at aligned times."""
        for i, t in enumerate(target_times):
            interpolated_u = interpolated_solutions[i]

            # Interpolate original solution at the exact target time for higher accuracy
            original_interpolator = interp1d(transformed_times, original_solutions, axis=0, kind='linear', fill_value='extrapolate')
            original_u = original_interpolator(t)

            plt.figure(figsize=(10, 6))
            plt.plot(grid.X, original_u[0, :], label="Original Solution", color='r', linestyle='-')
            plt.plot(grid.X, interpolated_u[0, :], "o", label=f"Interpolated Solution at t={t:.7f}", color='b')
            plt.xlabel("X")
            plt.ylabel("Solution (u)")
            plt.title(f"Aligned Original vs Interpolated Solutions (nx={nx}, t={t:.7f})")
            plt.legend()
            plt.savefig(f"aligned_comparison_plot_nx_{nx}_t_{t:.7f}.pdf")
            plt.close()

if __name__ == "__main__":
    nx_values = [16, 32, 64, 128, 256, 512, 1024]
    aligned_interpolator = MacCormackInterpolationAligned(nx_values)
    aligned_interpolator.run_interpolation()