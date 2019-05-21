import numpy as np
import matplotlib.pyplot as plt

from .experiment import Experiment


class XorExperiment(Experiment):
    """The XOR function with repeated attributes."""
    def __init__(self, **kwargs):
        N_EXAMPLES = 1000
        rng = np.random.RandomState(0)
        X = rng.uniform(-1, 1, size=(N_EXAMPLES, 2))
        bias = np.ones((N_EXAMPLES, 1))
        X = np.hstack([X, bias])
        Z = np.array([self._w_star(x[:2]) for x in X])
        y = np.sign(X[:,0]) != np.sign(X[:,1])
        super().__init__(X, Z, y, **kwargs)

    @staticmethod
    def _w_star(x):
        if x[0] >= 0:
            if x[1] >= 0:
                # pos, pos -> neg + neg -> neg
                w = [-1, -1, 0]
            else:
                # pos, neg -> pos + pos -> pos
                w = [1, -1, 0]
        else:
            if x[1] >= 0:
                # neg, pos -> pos + pos -> pos
                w = [-1, 1, 0]
            else:
                # neg, neg -> neg + neg -> neg
                w = [1, 1, 0]
        return np.array(w)


def plot_xor(path, experiment, model):
    X, y = experiment.X, experiment.y

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    bias = np.ones(xx.size)
    X_grid = np.c_[xx.ravel(), yy.ravel(), bias]
    T = model.predict(X_grid, discretize=False)

    fig, ax = plt.subplots(1, 1)
    ax.contourf(xx, yy, T.reshape(xx.shape), alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    fig.savefig(path, bbox_inches='tight', pad_inches=0)


