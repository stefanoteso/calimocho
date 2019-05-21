import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import check_random_state

from . import load


class Experiment:
    def __init__(self, X, Z, y, n_examples=None, rng=None):

        # Shuffle the examples and truncate if necessary
        rng = np.random.RandomState(0)
        pi = rng.permutation(len(X))
        if n_examples and n_examples < len(pi):
            pi = pi[:n_examples]
        X, Z, y = X[pi], Z[pi], y[pi]

        self.X, self.Z, self.y = X, Z, y
        self.rng = check_random_state(rng)

    def split(self, n_splits=10, prop_known=0.5):
        # Generate balanced folds
        kfold = StratifiedKFold(n_splits=n_splits,
                                shuffle=True,
                                random_state=self.rng)
        for nonts_indices, ts_indices in kfold.split(self.X, self.y):

            # Split the non-test set into known set and training set
            split = StratifiedShuffleSplit(n_splits=1,
                                           test_size=prop_known,
                                           random_state=self.rng)
            for tr_indices, kn_indices in split.split(self.X[nonts_indices],
                                                      self.y[nonts_indices]):

                yield kn_indices, tr_indices, ts_indices
