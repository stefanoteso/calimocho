import os.path; directory = os.path.dirname(__file__)
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Input, multiply
from keras.utils import to_categorical, plot_model
import keras.backend as K
import innvestigate
import innvestigate.utils as iutils
from sklearn.utils import check_random_state
import math


import tensorflow as tf
from tensorflow.python.ops import array_ops


from . import Classifier, add_column, variable_with_weight_decay, hessian_vector_product

_CHECKPOINT = '/tmp/nn_with_lrp.h5'


class NNWithIF(Classifier):
    """A simple dense neural net with influence function(IF).

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
        self.batch_size = kwargs.pop('batch_size')
        self.data_sets = kwargs.pop('data_sets')
        self.train_dir = kwargs.pop('train_dir', 'output')
        self.rng = check_random_state(17) #check_random_state(kwargs.pop('rng', None))

        try:
            self.initial_learning_rate2 = kwargs.pop('initial_learning_rate2')
        except:
            self.initial_learning_rate2 = 0
        self.decay_epochs = kwargs.pop('decay_epochs')

        if 'mini_batch' in kwargs:
            self.mini_batch = kwargs.pop('mini_batch')
        else:
            self.mini_batch = True

        if 'damping' in kwargs:
            self.damping = kwargs['damping']
        else:
            self.damping = 0.0

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # Initialize session
        config = tf.ConfigProto(
            intra_op_parallelism_threads=1)  # (inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)

    def _get_all_params(self):

        all_params = []
        for layer in ['relu%s'%i for i in np.arange(1, len(self.layers_all))]:
            # for var_name in ['weights', 'biases']:
            with tf.variable_scope("%s" % layer, reuse=tf.AUTO_REUSE):
                weights = tf.get_variable("weights", dtype=tf.float32)
                all_params.append(weights)

        return all_params

    def _get_train_op(self, total_loss, global_step, learning_rate):
        """
        Return train_op
        """
        with tf.variable_scope("default", reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op

    def _get_accuracy_op(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, float32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar float32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """
        correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), tf.cast(labels, tf.int32), 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32)) / tf.shape(labels)[0]

    def print_model_eval(self):
        params_val = self.sess.run(self.params)

        all_train_feed_dict = self._fill_feed_dict_with_all_ex(self.data_sets.train)
        all_test_feed_dict = self._fill_feed_dict_with_all_ex(self.data_sets.test)

        grad_loss_val, loss_no_reg_val, loss_val, train_acc_val = self.sess.run(
            [self.grad_total_loss_op, self.loss_no_reg, self.total_loss, self.accuracy_op],
            feed_dict=all_train_feed_dict)

        test_loss_val, test_acc_val = self.sess.run(
            [self.loss_no_reg, self.accuracy_op],
            feed_dict=all_test_feed_dict)

        print('Train loss (w reg) on all data: %s' % loss_val)
        print('Train loss (w/o reg) on all data: %s' % loss_no_reg_val)

        print('Test loss (w/o reg) on all data: %s' % test_loss_val)
        print('Train acc on all data:  %s' % train_acc_val)
        print('Test acc on all data:   %s' % test_acc_val)

        return train_acc_val, test_acc_val, loss_val, test_loss_val


    def _placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32,
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,
            shape=(None),
            name='labels_placeholder')
        z_placeholder = tf.placeholder(
            tf.int32,
            shape=(None, self.input_dim),
            name='z_placeholder')
        return input_placeholder, labels_placeholder, z_placeholder

    def _fill_feed_dict_with_all_ex(self, data_set):
        feed_dict = {
            self.input_placeholder: data_set.x,
            self.labels_placeholder: data_set.labels
        }
        return feed_dict

    def _fill_doubled_feed_dict_with_batch_no_shuffle(self, data_set, batch_size=0):
        if batch_size is None:
            return self._fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size

        input_feed, labels_feed = data_set.next_batch_no_shuffle(batch_size)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def _loss(self, logits, labels):
        labels = tf.one_hot(labels, depth=self.num_classes, dtype=tf.float32)
        ll_logits = logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True)
        cross_entropy = tf.reduce_sum(tf.multiply(labels, -ll_logits), reduction_indices=1)  # logist --> tf.nn.log_softmax(logits))

        indiv_loss_no_reg = cross_entropy
        loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.add_to_collection('losses', loss_no_reg)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        return total_loss, loss_no_reg, indiv_loss_no_reg


    def _inference(self, input, nonlinearity=tf.nn.leaky_relu):
        """
        It implements the inference algorithm and ouputs the log probability for each class
        """
        for idx, layer in enumerate(['relu%s'%i for i in np.arange(1, len(self.layers_all))]):
            input = add_column(input)
            with tf.variable_scope(layer, reuse=tf.AUTO_REUSE):
                weights = variable_with_weight_decay(
                    'weights',
                    [self.layers_all[idx]+1, self.layers_all[idx + 1]],
                    stddev=1.0 / math.sqrt(float(self.input_dim)*1),
                    wd=None)

                outputs = nonlinearity(tf.matmul(input, weights))
                input = outputs

        return outputs

    def _init_graph(self, input_dim, num_classes, layers):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.initial_learning_rate = 0.001
        self.l2_params_placeholder = tf.placeholder("float32", None, name="l2_params")
        self.layers_all = list([self.input_dim]) + layers + list([self.num_classes])

        self.sess = tf.Session()  # tf.Session(config=config)
        K.set_session(self.sess)

        # Setup input
        self.input_placeholder, self.labels_placeholder, self.Z_placeholder = self._placeholder_inputs()

        # Setup inference and training
        self.logits = self._inference(self.input_placeholder)

        self.params = self._get_all_params()
        self.damping = tf.Variable(0, trainable=False, dtype=tf.float32)
        self.scale = tf.Variable(10000000000, trainable=False, dtype=tf.float32)  # np.float32(1)

        self.total_loss, self.loss_no_reg, self.indiv_loss_no_reg = self._loss(self.logits, self.labels_placeholder)


        assert self.num_classes is not None


        # Setup gradients and Hessians
        self.grad_total_loss_op = tf.gradients(self.total_loss, self.params)
        self.grad_loss_no_reg_op = tf.gradients(self.loss_no_reg, self.params)

        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape(), name='v_placeholder') for a in
                              self.params]
        self.u_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape(), name='u_placeholder') for a in
                              self.params]

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.Variable(self.initial_learning_rate, name='learning_rate', trainable=False,
                                         dtype=tf.float32)

        # right reasons loss
        self.if_per_sample = tf.map_fn(self.IF_per_sample, (self.input_placeholder, self.labels_placeholder, self.input_placeholder, self.labels_placeholder), dtype=tf.float32, parallel_iterations=50)
        self.right_reasons_loss_if = self.right_reasons_if()
        self.rrr = tf.reduce_sum(self.right_reasons_loss_if) + self.total_loss
        self.train_op = self._get_train_op(self.rrr, self.global_step, self.learning_rate)


        # Setup misc
        self.saver = tf.train.Saver()

        self.hessian_vector = hessian_vector_product(self.total_loss, self.params, self.v_placeholder)

        self.grad_loss_wrt_input_op = tf.gradients(self.total_loss, self.input_placeholder)

        # Because tf.gradients auto accumulates, we probably don't need the add_n (or even reduce_sum)
        self.influence_op = tf.add_n(
            [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) for a, b in
             zip(self.grad_total_loss_op, self.v_placeholder)])

        self.grad_influence_wrt_input_op = tf.gradients(self.influence_op, self.input_placeholder)

        self.adversarial_loss, self.indiv_adversarial_loss = None, None  # self.adversarial_loss(self.logits, self.labels_placeholder)
        if self.adversarial_loss is not None:
            # self.grad_adversarial_loss_op = tf.gradients(self.adversarial_loss, self.params)
            self.grad_adversarial_loss_op = tf.gradients(self.adversarial_loss, np.asarray(self.params)[:, 0].tolist())

        # compute hvp out with tf
        # \sum_i L(Z_i, \hat{\theta})  i are TRAINING indices

        self.accuracy_op = self._get_accuracy_op(self.logits, self.labels_placeholder)

        init = tf.global_variables_initializer()
        self.sess.run(init)


    def fit(self, X, Z, y,
            mask=None,
            batch_size=None,
            n_epochs=1,
            callback=None,
            warm=True, **kwargs):
        """
        Trains a model for a specified number of steps.
        """

        logger.info('Training for %s steps', n_epochs)

        self._init_graph(X.shape[1], len(np.unique(y)), [5])


        sess = self.sess

        for step in range(n_epochs):
            #self.update_learning_rate(step)
            start_time = time.time()

            grad_influence_feed_dict = {self.input_placeholder:X.astype(np.float32), self.labels_placeholder:y.astype(np.int32), self.Z_placeholder:Z}
            #grad_influence_feed_dict[self.input_placeholder_test] = self.data_sets.train.x[[0]]
            #grad_influence_feed_dict[self.labels_placeholder_test] = self.data_sets.train.labels[[0]]
            #self.update_lambda(step, n_epochs, 1, 1, grad_influence_feed_dict)
            grad_influence_feed_dict[self.l2_params_placeholder] = 1.0
            loss_val, if_val, ig_val, if_neg_val, ig_neg_val = 0, 0, 0, 0, 0  # sess.run([self.total_loss, self.right_reasons_loss_if, self.right_reasons_loss_ig, self.right_reasons_loss_if_neg, self.right_reasons_loss_ig_neg], feed_dict=grad_influence_feed_dict)
            _ = sess.run(self.train_op, feed_dict=grad_influence_feed_dict)

            duration = time.time() - start_time

            if True:
                if step % 1 == 0:
                    # Print status to stdout.
                    logger.info('Step %d: loss = %.10f (%.3f sec)' % (step, loss_val, duration))
                    logger.info('Step %d: if = %.10f (%.3f sec)' % (step, np.sum(if_val), duration))
                    logger.info('Step %d: loss+if = %.10f (%.3f sec)' % (step, loss_val + np.sum(if_val), duration))


    def IF_per_sample(self, X_and_y):
        single_input, label, in_test, l_test = X_and_y

        logits = self._inference(array_ops.stop_gradient(tf.reshape(in_test, shape=[1, -1])))
        ll_logits = logits - tf.reduce_logsumexp(logits, axis=1, keepdims=True)
        cross_entropy = ll_logits[:, 1]  #todo tmp test

        grad_cross_entropy_op = [tf.gradients(cross_entropy, w)[0] for w in self.params]

        ignore_hessian = True
        approx = 'lu'
        damping = 0.01
        if not ignore_hessian:
            if approx is 'taylor':
                logger.info('lissa')
                grad_cross_entropy_op_tensor = tf.concat([tf.reshape(grad, [-1, 1]) for grad in grad_cross_entropy_op], axis=0)

                cur_estimate = grad_cross_entropy_op_tensor

                for j in range(30):
                    hvp = tf.matmul(self.hessian_tensor, cur_estimate)
                    cur_estimate = grad_cross_entropy_op_tensor + (1-damping)*cur_estimate - hvp / self.hessian_norm #(hvp_w / self.hessian_norm_w)

                inverse_hvp = cur_estimate / self.hessian_norm

                inverse_hvp = tf.split(inverse_hvp, tf.stack([grad.get_shape()[0]*grad.get_shape()[1] for grad in grad_cross_entropy_op]))
                inverse_hvp = [tf.reshape(hv, grad_cross_entropy_op[i].get_shape()) for i, hv in enumerate(inverse_hvp)]

            elif approx is 'cholesky':
                v_w = tf.concat([tf.reshape(grad, [-1, 1]) for grad in grad_cross_entropy_op], axis=0)
                H_w = tf.linalg.cholesky(self.hessian_tensor)  # shape 10 x 2 x 2
                X_w = tf.linalg.cholesky_solve(H_w, v_w)  # shape 10 x 2 x 1
                inverse_hvp = tf.split(X_w, tf.stack([grad.get_shape()[0]*grad.get_shape()[1] for grad in grad_cross_entropy_op]))
                inverse_hvp = [tf.reshape(hv, grad_cross_entropy_op[i].get_shape()) for i, hv in enumerate(inverse_hvp)]

            elif approx is 'lu':
                v_w = tf.concat([tf.reshape(grad, [-1, 1]) for grad in grad_cross_entropy_op], axis=0)
                X_w = tf.linalg.solve(self.hessian_tensor, v_w)
                inverse_hvp = tf.split(X_w, tf.stack([grad.get_shape()[0]*grad.get_shape()[1] for grad in grad_cross_entropy_op]))
                inverse_hvp = [tf.reshape(hv, grad_cross_entropy_op[i].get_shape()) for i, hv in enumerate(inverse_hvp)]

            elif approx is 'naive':
                v_w = tf.concat([tf.reshape(grad, [-1, 1]) for grad in grad_cross_entropy_op], axis=0)
                hvp_w = tf.matmul(self.hessian_inv_tensor, v_w)
                inverse_hvp = tf.split(hvp_w, tf.stack([grad.get_shape()[0]*grad.get_shape()[1] for grad in grad_cross_entropy_op]))
                inverse_hvp = [tf.reshape(hv, grad_cross_entropy_op[i].get_shape()) for i, hv in
                                 enumerate(inverse_hvp)]

            else:
                raise Exception('%s not supported'%approx)
        else:
            inverse_hvp = grad_cross_entropy_op   # ignore hessian

        logits_2 = self._inference(tf.reshape(single_input, shape=[1, -1]))
        ll_logits_2 = logits_2 - tf.reduce_logsumexp(logits_2, axis=1, keepdims=True)
        indiv_loss_2 = ll_logits_2[:, 1]  #todo tmp test

        grad_total_loss_op = [tf.gradients(indiv_loss_2, w)[0] for w in self.params]
        influence_op = tf.add_n([tf.reduce_sum(tf.multiply(a, b)) for a, b in zip(grad_total_loss_op, inverse_hvp)])  # [ [batch_size,], [batch_size], [batch_size,] ]
        gradXes = tf.gradients(influence_op, single_input)[0] # [ [1,75] ]
        return gradXes

    def right_reasons_if(self, sample_idx=None, **kwargs):
        influence_mask = tf.multiply(tf.cast(self.Z_placeholder, tf.float32), self.if_per_sample)
        rightreasons = tf.reduce_sum(tf.square(influence_mask), axis=1) * self.l2_params_placeholder
        return rightreasons

    def predict(self, X, discretize=True):
        logits = self.sess.run(self.logits, feed_dict={self.input_placeholder: X})
        return np.argmax(logits, axis=1)


    def explain(self, X):
        return self.sess.run(self.if_per_sample, feed_dict={self.input_placeholder: X})

    def evaluate(self, X, Z, y, which='both'):
        loss_y = self.sess.run(self.total_loss, feed_dict={self.input_placeholder: X, self.labels_placeholder: y})
        loss_z = self.sess.run(self.rrr, feed_dict={self.input_placeholder: X, self.labels_placeholder: y})
        return {'both': (loss_y, loss_z), 'y': loss_y, 'z': loss_z}[which]
