import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Grid1D import Grid1D
from state_equation import StateEquation

class GradientCalculator:
    def __init__(self, u, lam, c, times_u, times_lam, T, Xi, grid):
        self.grid = grid
        self.T = T                      # Final time
        self.Xi = Xi                    # Regularization parameter
        self.nx = grid.nx               # Number of spatial nodes (excluding endpoints)
        self.dx = grid.dx               # Spatial step size
        self.numVars = u.shape[1]       # Number of variables 

        # === Interpolate both u and lam to a common target time grid ===
        Nt_common = max(len(times_u), len(times_lam))
        times_common = np.linspace(0, T, Nt_common)

        self.times = times_common
        self.dt = times_common[1] - times_common[0]
        self.Nt = Nt_common

        # Interpolate state and adjoint solutions to the common time grid
        self.u = self.interpolate_solution_in_time(u, times_u, times_common)
        self.lam = self.interpolate_solution_in_time(lam, times_lam, times_common)

        # c(x), trapezoidal quadrature weights, and derivative matrix
        self.c = c
        self.q_t = self.trapezoidal_weights(self.Nt, self.dt)           # quadrature weights in t
        self.q_x = self.trapezoidal_weights(self.nx + 1, self.dx)       # quadrature weights in x
        
        # Use the derivative matrix as specified in the notes
        self.D = self.build_derivative_matrix(self.nx + 1, self.dx)     # d/dx operator

    def interpolate_solution_in_time(self, solution, times_source, times_target):
        """Linear interpolation of a (Nt, numVars, nx+1) solution to a new time grid"""
        Nt_target = len(times_target) 
        numVars = solution.shape[1]
        nx_plus1 = solution.shape[2]

        interpolated = np.zeros((Nt_target, numVars, nx_plus1))

        for i in range(numVars):
            for j in range(nx_plus1):
                interpolator = interp1d(times_source, solution[:, i, j], kind='linear', fill_value="extrapolate")
                interpolated[:, i, j] = interpolator(times_target)
        
        return interpolated

    def trapezoidal_weights(self, N, h):
        """Trapezoidal rule weights for N points with step h"""
        weights = np.ones(N)
        weights[0] = 0.5
        weights[-1] = 0.5
        return weights * h

    def build_derivative_matrix(self, N, h):
        """Build finite difference derivative matrix"""
        D = np.zeros((N, N))
        
        # Interior points (first-order central difference)
        for j in range(1, N - 1):
            D[j, j - 1] = -1 / (2 * h)
            D[j, j + 1] = 1 / (2 * h)

        # Forward difference at left boundary (second-order)
        D[0, 0] = -3 / (2 * h)
        D[0, 1] = 4 / (2 * h)
        D[0, 2] = -1 / (2 * h)

        # Backward difference at right boundary (second-order)
        D[-1, -3] = 1 / (2 * h)
        D[-1, -2] = -4 / (2 * h)
        D[-1, -1] = 3 / (2 * h)

        return D

    def compute_Dc(self):
        """Compute finite difference approximation of dc/dx at grid points"""
        return np.dot(self.D, self.c)

    def compute_Dej(self, j):
        """Compute d/dx of the basis vector e_j"""
        e_j = np.zeros(self.nx + 1)
        e_j[j] = 1.0
        return np.dot(self.D, e_j)

    def build_U(self):
        """Return state variable data"""
        return self.u.copy()

    def build_LAMBDA(self):
        """Return adjoint variable data"""
        return self.lam.copy()

    def compute_DU(self, U):
        """Compute spatial derivatives of U using the derivative matrix"""
        DU = np.zeros_like(U)
        for t in range(self.Nt):
            for i in range(self.numVars):
                DU[t, i] = np.dot(self.D, U[t, i])
        return DU

    def compute_gradient(self, j):
        """
        Compute the gradient dJ/dc_j 
        dJ/dc_j = T*Xi*q_x^T*((Dc)*(De_j)) - w_x^j*e_j^T*(LAMBDA*DU)*q_t
        """
        # Compute spatial derivative of c (dc/dx)
        Dc = self.compute_Dc()

        # Compute spatial derivative of the basis function e_j (De_j)
        Dej = self.compute_Dej(j)

        # Get state and adjoint solutions
        U = self.build_U()
        LAMBDA = self.build_LAMBDA()

        # Compute spatial derivative of state variable (DU)
        DU = self.compute_DU(U)

        # ===== Term 1: Regularization term =====
        # T*Xi*q_x^T*((Dc)*(De_j))
        # Element-wise product of Dc and Dej, weighted by q_x, then summed
        term1 = self.T * self.Xi * np.sum(self.q_x * Dc * Dej)

        # ===== Term 2: Adjoint sensitivity term =====
        # w_x^j*e_j^T*(LAMBDA*DU)*q_t
        term2 = np.zeros(self.numVars)
        
        for i in range(self.numVars):
            # e_j^T selects the jth element of the vector
            # LAMBDA*DU is the element-wise product of LAMBDA and DU
            lambda_du_product = np.zeros(self.Nt)
            for t in range(self.Nt):
                # Get the element-wise product at position j
                lambda_du_product[t] = LAMBDA[t, i, j] * DU[t, i, j]
            
            # Apply the quadrature weight w_x^j and compute dot product with q_t
            term2[i] = self.q_x[j] * np.dot(lambda_du_product, self.q_t)
        
        # Return the final gradient: term1 - term2
        return term1 - term2


# === Main driver ===
if __name__ == "__main__":
    for nx in [16, 32, 64, 128, 256, 512, 1024]:
        print(f"\nComputing and plotting gradient for nx = {nx}")

        grid = Grid1D(nx=nx)
        Nt = 100
        T = 1.0
        Xi = 1.0

        # Load saved state and adjoint solutions
        try:
            u = np.load(f"StateEquation_Solutions_nx_{nx}.npy")
            lam = np.load(f"adjoint_solutions_nx_{nx}.npy")
        except FileNotFoundError:
            print(f"Missing data files for nx = {nx}. Skipping.")
            continue

        times_u = np.linspace(0, T, u.shape[0])
        times_lam = np.linspace(0, T, lam.shape[0])  

        # Import c(x) from model
        model = StateEquation()
        x = grid.X
        c = np.array([model.c(xj) for xj in x])

        # Calculate adjoint gradient
        calc = GradientCalculator(u=u, lam=lam, c=c,
                                 times_u=times_u, times_lam=times_lam,
                                 T=T, Xi=Xi, grid=grid)

        adjoint_gradients = np.zeros((nx + 1, calc.numVars))
        
        # Compute gradient at each spatial location
        for j in range(nx + 1):
            adjoint_gradients[j] = calc.compute_gradient(j)
            print(f"Gradient at j={j}: {adjoint_gradients[j]}")
        
        # Save adjoint gradient
        np.save(f"adjoint_gradient_nx_{nx}.npy", adjoint_gradients)
        print(f"Adjoint gradient data saved as adjoint_gradient_nx_{nx}.npy")
        
        # Plot gradient
        plt.figure(figsize=(10, 6))
        plt.plot(grid.X, adjoint_gradients[:, 0], 'b-', label='Adjoint Gradient')
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('Gradient')
        plt.title(f'Gradient at (nx={nx})')
        plt.legend()
        plt.savefig(f"gradient_plot_nx_{nx}.png")
        plt.show()