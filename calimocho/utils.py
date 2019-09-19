import pickle
import numpy as np
import tensorflow as tf
from tensorflow import gradients
from tensorflow.python.ops import array_ops, math_ops


def load(path, **kwargs):
    with open(path, 'rb') as fp:
        return pickle.load(fp, **kwargs)


def dump(path, what, **kwargs):
    with open(path, 'wb') as fp:
        pickle.dump(what, fp, **kwargs)

def add_column(input):
    ones = tf.ones((tf.shape(input)[0], 1))
    return tf.concat((input, ones), axis=1)


def variable(name, shape, initializer):
    dtype = tf.float32
    var = tf.get_variable(
        name,
        shape,
        initializer=initializer,
        dtype=dtype)
    return var


def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = variable(
        name,
        shape,
        initializer=tf.truncated_normal_initializer(
            stddev=stddev,
            dtype=dtype))

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def hessian_vector_product(ys, xs, v):
    """Multiply the Hessian of `ys` wrt `xs` by `v`.
    This is an efficient construction that uses a backprop-like approach
    to compute the product between the Hessian and another vector. The
    Hessian is usually too large to be explicitly computed or even
    represented, but this method allows us to at least multiply by it
    for the same big-O cost as backprop.
    Implicit Hessian-vector products are the main practical, scalable way
    of using second derivatives with neural networks. They allow us to
    do things like construct Krylov subspaces and approximate conjugate
    gradient descent.
    Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
    x, v)` will return an expression that evaluates to the same values
    as (A + A.T) `v`.
    Args:
      ys: A scalar value, or a tensor or list of tensors to be summed to
          yield a scalar.
      xs: A list of tensors that we should construct the Hessian over.
      v: A list of tensors, with the same shapes as xs, that we want to
         multiply by the Hessian.
    Returns:
      A list of tensors (or if the list would be length 1, a single tensor)
      containing the product between the Hessian and `v`.
    Raises:
      ValueError: `xs` and `v` have different length.
    """

    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")

    # First backprop
    grads = gradients(ys, xs)

    # grads = xs

    assert len(grads) == length

    elemwise_products = [
        math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]

    # Second backprop
    grads_with_none = gradients(elemwise_products, xs)
    return_grads = [
        grad_elem if grad_elem is not None \
            else tf.zeros_like(x) \
        for x, grad_elem in zip(xs, grads_with_none)]

    return return_grads




class DataSet(object):

    def __init__(self, x, labels):

        if len(x.shape) > 2:
            x = np.reshape(x, [x.shape[0], -1])

        assert(x.shape[0] == labels.shape[0])

        x = x.astype(np.float32)

        self._x = x
        self._x_batch = np.copy(x)
        self._labels = labels
        self._labels_batch = np.copy(labels)
        self._num_examples = x.shape[0]
        self._index_in_epoch = 0

    @property
    def x(self):
        return self._x

    @property
    def labels(self):
        return self._labels

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def num_examples(self):
        return self._num_examples

    def batch_indices_list(self, batch_size):
        start = np.maximum(self.index_in_epoch - batch_size, 0)
        end = self.index_in_epoch
        return np.arange(start, end)

    def reset_batch(self):
        self._index_in_epoch = 0
        self._x_batch = np.copy(self._x)
        self._labels_batch = np.copy(self._labels)

    def next_batch(self, batch_size):
        assert batch_size <= self._num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:

            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._x_batch = self._x_batch[perm, :]
            self._labels_batch = self._labels_batch[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._x_batch[start:end], self._labels_batch[start:end]

    def next_batch_no_shuffle(self, batch_size):
        assert batch_size <= self._num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._x_batch[start:end], self._labels_batch[start:end]

