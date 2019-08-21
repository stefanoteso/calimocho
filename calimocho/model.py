import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod


class Classifier(ABC):
    """Abstract base class for explainable binary classifiers."""


    @abstractmethod
    def fit(self, X, Z, y,
            mask=None,
            batch_size=None,
            n_epochs=1,
            callback=None,
            warm=True):
        """Trains the model on labels and explanation corrections."""
        pass


    @abstractmethod
    def predict(self, X, discretize=True):
        """Computes a prediction for every instance."""
        pass


    def predict_proba(self, X):
        y_pred = self.predict(X, discretize=False)
        return np.hstack((1 - y_pred, y_pred))


    def margin(self, X, method='min_probability'):
        # TODO implement entropy etc.
        assert method == 'min_probability'
        return np.min(self.predict_proba(X), axis=1)


    @abstractmethod
    def explain(self, X):
        """Returns an explanation for every instance."""
        pass


    @abstractmethod
    def evaluate(self, X, y, which='both'):
        """Compute the model's loss w.r.t. true labels and explanations."""
        pass
