import numpy as np
import tensorflow as tf
from os.path import join
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle

from collections import defaultdict
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

from . import Experiment, PipeStep, load, dump


_TO_OHE = {
    (255,   0,   0): (0, 0, 0, 1), # r
    (0,   255,   0): (0, 0, 1, 0), # g
    (0,   128, 255): (0, 1, 0, 0), # b
    (128,   0, 255): (1, 0, 0, 0), # v
}
_TO_RAW = {v: k for k, v in _TO_OHE.items()}
_COLORS_RGB = tuple(_TO_OHE.keys())


class ColorsExperiment(Experiment):
    """Implementation of the toy colors problem."""
    def __init__(self, **kwargs):
        self.rule = kwargs.pop('rule')

        pp_path = join('data', 'toy_colors_pp.pickle')
        try:
            X, Z, y, flat_images = load(pp_path)
        except:
            data = np.load(join('data', 'toy_colors.npz'))
            raw_images = np.vstack([data['arr_0'], data['arr_1']])
            n_examples = len(raw_images)

            flat_images = np.array([self._raw_to_flat(ri.reshape(5, 5, 3))
                                    for ri in raw_images])
            X = np.array([self._raw_to_ohe(ri.reshape(5, 5, 3)).ravel()
                          for ri in raw_images])
            X = np.hstack([X, np.ones((n_examples, 1))])
            Z = np.vstack([self._w_star(x) for x in X])
            y = 1 - np.hstack((data['arr_2'], data['arr_3']))

            # DEBUG
            #y2 = np.einsum('bi,bi->b', X, Z)
            #print(np.hstack([y.reshape(-1, 1), y2.reshape(-1, 1)]))
            #quit()

            dump(pp_path, [X, Z, y, flat_images])

        self.flat_images = flat_images

        super().__init__(X, Z, y, **kwargs)

    @staticmethod
    def _raw_to_ohe(ri):
        ohe = [_TO_OHE[tuple(ri[r, c])] for r, c in product(range(5), repeat=2)]
        return np.array(ohe).reshape(5, 5, 4)

    @staticmethod
    def _ohe_to_raw(ri):
        raw = [_TO_RAW[tuple(ri[r, c])] for r, c in product(range(5), repeat=2)]
        return np.array(raw).reshape(5, 5, 3)

    @staticmethod
    def _raw_to_flat(ri):
        fi = [_COLORS_RGB.index(tuple(ri[r, c])) for r, c in product(range(5), repeat=2)]
        return np.array(fi)

    @staticmethod
    def _flat_to_ohe(fi):
        ohe = [_TO_OHE[_COLORS_RGB[fi[r,c]]] for r, c in product(range(5), repeat=2)]
        return np.array(ohe).ravel()

    @staticmethod
    def rule0(x):
        y = (x[0, 0] == x[0, 4]).all() and \
            (x[0, 0] == x[4, 0]).all() and \
            (x[0, 0] == x[4, 4]).all()
        return y

    @staticmethod
    def rule1(x):
        y = (x[0, 1] != x[0, 2]).any() and \
            (x[0, 1] != x[0, 3]).any() and \
            (x[0, 2] != x[0, 3]).any()
        return y

    def _w_star(self, x):
        x = x[:-1].reshape(5, 5, 4)
        z = np.zeros_like(x)
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
                index = np.argmax(x[r, c])
                coeff = z[r, c, index]
                if np.abs(coeff) >= 1e-2:
                    color = cm.RdBu(0.5 * coeff + 0.5)
                    axes[i].add_patch(Circle((c, r), 0.35, color=color))

        fig.savefig(path, bbox_inches=0, pad_inches=0)
        plt.close(fig)

    def explain_lime(self,
                     model,
                     known_examples,
                     target_example,
                     n_repeats=5,
                     n_samples=100,
                     n_features=4,
                     metric='euclidean',
                     kernel_width=1.0):
        CLASS_NAMES = ['negative', 'positive']
        FEATURE_NAMES = ['{r}_{c}'.format(**locals())
                         for r, c in product(range(5), repeat=2)]
        FEATURES = list(range(len(FEATURE_NAMES)))

        lime = LimeTabularExplainer(self.flat_images[known_examples],
                                    class_names=CLASS_NAMES,
                                    feature_names=FEATURE_NAMES,
                                    categorical_features=FEATURES,
                                    discretize_continuous=False,
                                    feature_selection='forward_selection',
                                    kernel_width=kernel_width,
                                    verbose=True)

        def flat_to_x(flat_images):
            n_examples = len(flat_images)
            X = np.array([self._flat_to_ohe(fi.reshape(5, 5)) for fi in flat_images.astype(int)],
                         dtype=np.float32)
            return np.hstack([X, np.ones((n_examples, 1))])

        pipeline = make_pipeline(PipeStep(flat_to_x), model)

        local_model = Ridge(alpha=1, fit_intercept=True, random_state=0)

        counts = defaultdict(int)
        for _ in range(n_repeats):
            explanation = lime.explain_instance(self.flat_images[target_example],
                                                pipeline.predict_proba,
                                                model_regressor=local_model,
                                                num_samples=n_samples,
                                                num_features=n_features,
                                                distance_metric=metric)
            print(explanation.as_list())
            for feat, coeff in explanation.as_list():
                coeff = int(np.sign(coeff))
                counts[(feat, coeff)] += 1

        sorted_counts = sorted(counts.items(), key=lambda _: _[-1])
        sorted_counts = list(sorted_counts)[-n_features:]
        return [fs for fs, _ in sorted_counts]
