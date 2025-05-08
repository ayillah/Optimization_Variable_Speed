from OutputHandler import OutputHandler
import matplotlib.pyplot as plt
import os

class FrameWritingOutputHandler(OutputHandler):

    def __init__(self, filename, directory='Results'):
        self.n = 0
        self.directory = directory
        self.filename = filename

        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)


    def write(self, X, W, t):
        
        colors = ('k-', 'b-', 'r-', 'm-', 'k--', 'b--', 'r--', 'm--')

        fig, ax = plt.subplots()
        for i in range(len(W)):
          ax.plot(X, W[i,:], colors[i])
        ax.set(ylim=(-4.0,4.0))
        ax.set_title('time={:5f}'.format(t))
        ax.set_box_aspect(1)
        plt.savefig('{}/{}-{}.png'.format(self.directory,
                                          self.filename, self.n))
        plt.close()

        self.n += 1


    def close(self):
        pass
