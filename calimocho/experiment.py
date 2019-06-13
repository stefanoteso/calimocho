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
            n_known = max(int(len(nonts_indices) * prop_known), 5)
            kn_indices = self.rng.choice(nonts_indices, size=n_known)
            tr_indices = np.array(list(set(nonts_indices) - set(kn_indices)))

            assert len(set(kn_indices) & set(tr_indices)) == 0
            assert len(set(kn_indices) & set(ts_indices)) == 0
            assert len(set(tr_indices) & set(ts_indices)) == 0

            yield kn_indices, tr_indices, ts_indices
