import numpy as np
import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian
from tensorflow.losses import log_loss, mean_squared_error
from tensorflow.train import AdamOptimizer
from sklearn.utils import check_random_state


_CHECKPOINT = '/tmp/senn.ckpt'


class SENN:
    """A tensorflow implementation of Self-Explaining Neural Networks (SENNs).

    x can have any range; it should have a bias term.
    z should be in the range [0, 1].
    y should be in the range [0, 1].

    NOTE: there is strong competition between lambda1 and lambda2: if lambda2
    is too large, then learning from explanations (understandably) does not
    work.  Use only one parameter instead.

    Parameters
    ----------
    eta : float, defaults to 0.01
        Learning rate for Stochastic Gradient Descent.
    lambda1 : float, defaults to 0.1
        Hyperparameter of the loss on z
    lambda2 : float, defaults to 0.01
        Hyperparameter of the regularizer on z.
    rng : None or int or numpy.random.RandomStream
        The RNG.
    """

    def __init__(self, **kwargs):
        self.eta = kwargs.pop('eta', 0.1)
        self.lambda1 = kwargs.pop('lambda1', 0.1)
        self.lambda2 = kwargs.pop('lambda2', 0.01)
        self.rng = check_random_state(kwargs.pop('rng', None))

    def _build_subnets(self, x):
        raise NotImplementedError()

    def _build(self, n_inputs):
        # Build the input/output variables
        x = tf.placeholder(shape=[None, n_inputs], name='x', dtype=tf.float32)
        y = tf.placeholder(shape=[None, 1], name='y', dtype=tf.float32)

        # Build the model
        w, phi = self._build_subnets(x)
        n_hidden = int(w.shape[1])
        assert phi.shape[1] == n_hidden, \
            'w(x) and phi(x) have incompatible shapes: {} {}'.format(w.shape, phi.shape)
        dot = tf.reduce_sum(w * phi, axis=1, keepdims=True, name='dot')
        f = tf.sigmoid(dot, name='f')

        z = tf.placeholder(shape=[None, n_hidden],
                           dtype=tf.float32,
                           name='z')

        # Build the losses
        loss_y = log_loss(y, f)
        loss_z = mean_squared_error(z[:-1], w[:-1])

        # Build the regularizers on w
        # TODO make this work on xor problem
        grad_f = tf.gradients(f, x)[0]
        jacob_phi = batch_jacobian(phi, x)
        w_times_jacob_phi = tf.einsum('boi,bo->bi', jacob_phi, w)
        reg_z = tf.reduce_sum(tf.squared_difference(grad_f, w_times_jacob_phi))
        #reg_z = tf.reduce_sum(tf.abs(w)) / float(n_hidden)

        # Build the optimizers
        self.train_op_y = AdamOptimizer(self.eta) \
                              .minimize(loss_y + self.lambda2 * reg_z)
        self.train_op_y_z = AdamOptimizer(self.eta) \
                                .minimize(loss_y + self.lambda1 * loss_z + self.lambda2 * reg_z)

        # Build the tensorflow session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self._saver = tf.train.Saver()
        self._saver.save(self.session, _CHECKPOINT)

        self.tf_vars = {
            'x': x,
            'z': z,
            'y': y,
            'w': w,
            'phi': phi,
            'f': f,
            'loss_y': loss_y,
            'loss_z': loss_z,
            'reg_z': reg_z,
        }

    def fit(self, X, Z, y, n_epochs=100, batch_size=None, callback=None,
            warm=True):

        if not hasattr(self, 'session'):
            self._build(X.shape[1])

        if not warm:
            self._saver.restore(self.session, _CHECKPOINT)

        if not batch_size:
            batch_size = int(np.sqrt(X.shape[0]))

        trace = []
        for epoch in range(n_epochs):

            batch = self.rng.randint(X.shape[0], size=batch_size)
            if Z is None:
                feed_dict = {
                    self.tf_vars['x']: X[batch],
                    self.tf_vars['y']: y[batch].reshape(-1, 1),
                }
                train_op = self.train_op_y
            else:
                feed_dict = {
                    self.tf_vars['x']: X[batch],
                    self.tf_vars['z']: Z[batch],
                    self.tf_vars['y']: y[batch].reshape(-1, 1),
                }
                train_op = self.train_op_y_z

            self.session.run(train_op, feed_dict=feed_dict)
            if callback is not None:
                info = callback(epoch, self)
                if info is not None:
                    trace.append(info)

        return trace

    def loss_y(self, X, y):
        assert hasattr(self, 'session'), 'fit the model first'
        feed_dict = {
            self.tf_vars['x']: X,
            self.tf_vars['y']: y.reshape(-1, 1),
        }
        return self.session.run(self.tf_vars['loss_y'], feed_dict=feed_dict)

    def loss_z(self, X, Z):
        assert hasattr(self, 'session'), 'fit the model first'
        if Z is None:
            return -1.0
        feed_dict = {
            self.tf_vars['x']: X,
            self.tf_vars['z']: Z,
        }
        return self.session.run(self.tf_vars['loss_z'], feed_dict=feed_dict)

    def predict(self, X, discretize=True):
        assert hasattr(self, 'session'), 'fit the model first'
        feed_dict = {self.tf_vars['x']: X}
        y_pred = self.session.run(self.tf_vars['f'], feed_dict=feed_dict)
        if not discretize:
            return y_pred
        return (0.5 * (np.sign(y_pred - 0.5) + 1)).astype(int)

    def predict_proba(self, X):
        assert hasattr(self, 'session'), 'fit the model first'
        y_pred = self.predict(X, discretize=False)
        return np.hstack((1 - y_pred, y_pred))

    def predict_entropy(self, X, which='labels'):
        raise NotImplementedError()

    def explain(self, X):
        feed_dict = {self.tf_vars['x']: X}
        return self.session.run(self.tf_vars['w'], feed_dict=feed_dict)


class FullFullSENN(SENN):
    """A SENN with two fully-connected subnetworks."""

    def __init__(self, **kwargs):
        self.w_sizes = kwargs.pop('w_sizes', [])
        self.phi_sizes = kwargs.pop('phi_sizes', [])
        super().__init__(**kwargs)

    def _build_subnets(self, x):
        w = x
        for l, units in enumerate(self.w_sizes or []):
            w = tf.layers.dense(w,
                                units=units,
                                activation=tf.sigmoid,
                                name='w{}'.format(l))
            w = 2 * w - 1

        phi = x
        for l, units in enumerate(self.phi_sizes or []):
            phi = tf.layers.dense(phi,
                                  units=units,
                                  activation=tf.sigmoid,
                                  name='phi{}'.format(l))
            phi = 2 * phi - 1

        return w, phi
