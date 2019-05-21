import numpy as np
import tensorflow as tf
from os.path import join
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle

from .experiment import Experiment
from .senn import SENN


_TO_OHE = {
    (255,   0,   0): (0, 0, 0, 1), # r
    (0,   255,   0): (0, 0, 1, 0), # g
    (0,   128, 255): (0, 1, 0, 0), # b
    (128,   0, 255): (1, 0, 0, 0), # v
}
_TO_RAW = {v: k for k, v in _TO_OHE.items()}


class ColorsExperiment(Experiment):
    """Implementation of the toy colors problem.

    _images is  (n_examples, 75 = 5 x 5 x (R, G, B))
    X is        (n_examples, 101 = 5 x 5 x ONEHOT + 1)
    Z is        (n_examples, 100 = 5 x 5 x ONEHOT + 1)
    """
    def __init__(self, **kwargs):
        self.rule = kwargs.pop('rule')

        data = np.load(join('data', 'toy_colors.npz'))
        images = np.vstack([data['arr_0'], data['arr_1']])

        X = np.array([self._raw_to_ohe(x.reshape(5, 5, 3)).ravel() for x in images])
        bias = np.ones((X.shape[0], 1))
        X = np.hstack([X, bias])
        Z = np.vstack([self._ohe_to_expl(x) for x in X])
        y = 1 - np.hstack((data['arr_2'], data['arr_3']))

        super().__init__(X, Z, y, **kwargs)

    @staticmethod
    def _raw_to_ohe(x):
        ohe = [_TO_OHE[tuple(x[r, c])] for r, c in product(range(5), repeat=2)]
        return np.array(ohe).reshape(5, 5, 4)

    @staticmethod
    def _ohe_to_raw(x):
        raw = [_TO_RAW[tuple(x[r, c])] for r, c in product(range(5), repeat=2)]
        return np.array(raw).reshape(5, 5, 3)

    @staticmethod
    def rule0(x):
        y = (x[0, 0] == x[0, 4]).all() and \
            (x[0, 0] == x[4, 0]).all() and \
            (x[0, 0] == x[4, 4]).all()
        return y

    @staticmethod
    def rule1(x):
        y = (x[0, 1] != x[0, 2]).any() or \
            (x[0, 1] != x[0, 3]).any() or \
            (x[0, 2] != x[0, 3]).any()
        return y

    def _ohe_to_expl(self, x):
        x = x[:-1].reshape(5, 5, 4)
        z = np.zeros((5, 5, 4))
        if self.rule == 0:
            if self.rule0(x):
                z[0, 0] = x[0, 0]
                z[0, 4] = x[0, 4]
                z[4, 0] = x[4, 0]
                z[4, 4] = x[4, 4]
            else:
                #pixels = x[np.ix_([0, 4], [0, 4])].reshape(-1, 4)
                #counts = np.sum(pixels, axis=0)
                # TODO pixels should be identical, punish the ones that are not
                z[0, 0] = -x[0, 0]
                z[0, 4] = -x[0, 4]
                z[4, 0] = -x[4, 0]
                z[4, 4] = -x[4, 4]
        else:
            if self.rule1(x):
                z[0, 1] = x[0, 1]
                z[0, 2] = x[0, 2]
                z[0, 3] = x[0, 3]
            else:
                #pixels = np.vstack([
                #    x[0, 1].reshape(1, 4),
                #    x[0, 2].reshape(1, 4),
                #    x[0, 3].reshape(1, 4)
                #])
                #counts = np.sum(pixels, axis=0)
                # TODO pixels should be different, punish the ones that are not
                z[0, 1] = -x[0, 1]
                z[0, 2] = -x[0, 2]
                z[0, 3] = -x[0, 3]
        return np.hstack([z.ravel(), 0])

    def dump_explanation(self, path, x, z_true, z_pred):
        fig, axes = plt.subplots(1, 2, sharey=True)

        x = x[:-1].reshape(5, 5, 4)
        for i, z in enumerate([z_true, z_pred]):
            axes[i].set_aspect('equal')

            z = z[:-1].reshape(5, 5, 4)
            axes[i].imshow(self._ohe_to_raw(x), interpolation='nearest')
            for r, c in product(range(5), repeat=2):
                x_r_c = x[r, c]
                value, index = np.max(x_r_c), np.argmax(x_r_c)
                coeff = z[r, c, index]
                # we don't plot if x[r, c, :] is all zeros
                if value > 0 and coeff != 0:
                    color = cm.RdBu(0.5 * coeff + 0.5)
                    axes[i].add_patch(Circle((c, r), 0.35, color=color))

        fig.savefig(path, bbox_inches=0, pad_inches=0)
        plt.close(fig)
