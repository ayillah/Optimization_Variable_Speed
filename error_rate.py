import numpy as np
import matplotlib.pyplot as plt

class ErrorRate:
    """
    Compute the error rate of a numerical method.
    """
    def __init__(self, Nx, error):
        self.Nx = Nx
        self.error = error
        
    def error_rate(self):
        """Compute the error rate."""

        # Compute the logs of the nodes and the errors
        logNx = np.log(np.array(self.Nx))
        logError = np.log(np.array(self.error))
        ones = np.ones(len(logNx))

        V = np.array([ones, logNx]).transpose()

        # Solve least squares system
        A = np.matmul(V.transpose(), V)
        b = np.matmul(V.transpose(), logError)

        c = np.linalg.solve(A, b)

        return c[1]


if __name__ == '__main__':

    nodes = [32, 64, 128, 256, 512, 1024]
    error_norms = [2.727550e-02,4.744334e-03,6.359127e-04,8.250237e-05,1.209993e-05,1.507543e-06]

    error_rate = ErrorRate(nodes, error_norms)
    p = error_rate.error_rate()

    print('Error rate = {:6f} '.format(p))

    plt.loglog(nodes, error_norms, 'r-o')
    plt.xlabel('log(Nx)')
    plt.ylabel('log(error_norms)')
    plt.grid()
    plt.show()

