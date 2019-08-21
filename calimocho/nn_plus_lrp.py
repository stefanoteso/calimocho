import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten
from keras.losses import binary_crossentropy
from keras.utils import to_categorical, plot_model
import innvestigate
import innvestigate.utils as iutils
from sklearn.utils import check_random_state

from . import Classifier


_CHECKPOINT = '/tmp/nn_with_lrp.h5'


class NNWithLRP(Classifier):
    """A simple dense neural net with layer-wise relevance propagation (LRP).

    Parameters
    ----------
    w_sizes : list of int, defaults to []
        Sizes of the hidden layers.
    eta : float, defaults to 0.01
        Learning rate.
    lambdas : tuple of float, defaults to (0.1, 0.01)
        Weight of the loss on the corrections
    method : str, defaults to 'lrp.epsilon'
        Name of the explainer, passed straight to innvestigate.
    method_kwargs : dict, defaults to {}
        Arguments to the explainer, passed striaght to innvestigate.
    rng : None or int or numpy.random.RandomStream
        The RNG.
    """

    def __init__(self, **kwargs):
        self.w_sizes = kwargs.pop('w_sizes', [])
        self.eta = kwargs.pop('eta', 0.1)
        self.lambdas = kwargs.pop('lambdas', (0.1, 0.01))
        self.method = kwargs.pop('method', 'lrp.z')
        self.method_kwargs = kwargs.pop('method_kwargs', {})
        self.rng = check_random_state(kwargs.pop('rng', None))
        assert all(l >= 0 for l in self.lambdas) and sum(self.lambdas) <= 1


    def _build(self, X, y):

        # Build the base model
        model = Sequential()
        for l, size in enumerate(self.w_sizes):
            if l == 0:
                model.add(Dense(size, input_dim=X.shape[1], activation='relu'))
            else:
                model.add(Dense(size, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        # Add LRP (or other explainer) layers to the base model
        uncapped_model = iutils.keras.graph.model_wo_softmax(model)
        analyzer = innvestigate.create_analyzer(self.method,
                                                uncapped_model,
                                                **self.method_kwargs)
        analyzer.create_analyzer_model()
        analyzer_model = analyzer._analyzer_model
        assert (len(analyzer._analysis_inputs) == 0
                and analyzer._n_constant_input == 0
                and analyzer._n_debug_output == 0)

        # Create a model that outputs both predictions and explanations
        l0, l1, l2 = 1 - sum(self.lambdas), self.lambdas[0], self.lambdas[1]
        twin_model = Model(inputs=model.inputs,
                           outputs=model.outputs + analyzer_model.outputs)
        twin_model.compile(optimizer='adam',
                           loss=['categorical_crossentropy',
                                 'mean_squared_error'],
                           loss_weights=[l0, l1])

        if False:
            print('inputs =', twin_model.inputs)
            print('outputs =', twin_model.outputs)
            plot_model(model, 'nn_plus_lrp:model.png')
            plot_model(uncapped_model, 'nn_plus_lrp:uncapped_model.png')
            plot_model(analyzer_model, 'nn_plus_lrp:analyzer_model.png')
            plot_model(twin_model, 'nn_plus_lrp:twin_model.png')

        return twin_model


    def fit(self, X, Z, y,
            batch_size=None,
            n_epochs=1,
            callback=None,
            warm=True):
        y = to_categorical(y, num_classes=2)

        if not hasattr(self, 'twin_model'):
            self.twin_model = self._build(X, y)
            self.twin_model.save_weights(_CHECKPOINT)

        if not warm:
            self.twin_model.load_weights(_CHECKPOINT)

        self.twin_model.fit(X,
                            [y, Z],
                            epochs=n_epochs,
                            batch_size=batch_size,
                            verbose=0)


    def predict(self, X, discretize=False):
        y_pred = self.twin_model.predict_on_batch(X)[0][:,1] # second column
        if discretize:
            sign = np.sign(y_pred - 0.5)
            y_pred = (0.5 * sign + 0.5).astype(int)
        return y_pred.reshape(-1, 1)


    def explain(self, X):
        return self.twin_model.predict_on_batch(X)[1]


    def evaluate(self, X, Z, y, which='both'):
        y = to_categorical(y, num_classes=2)
        loss, loss_y, loss_z = self.twin_model.evaluate(X, [y, Z], verbose=0)
        return {'both': (loss_y, loss_z), 'y': loss_y, 'z': loss_z}[which]
