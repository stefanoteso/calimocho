import numpy as np
import tensorflow as tf
from innvestigate.analyzer import analyzers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.utils import check_random_state
from os.path import join
from time import time

from calimocho import *


EXPERIMENTS = {
    'xor': XorExperiment,
    'colors0': lambda **kwargs: ColorsExperiment(rule=0, **kwargs),
    'colors1': lambda **kwargs: ColorsExperiment(rule=1, **kwargs),
}


MODELS = {
    'senn': \
        lambda args, rng: \
            FullFullSENN(
                w_sizes=args.w_sizes,
                phi_sizes=args.phi_sizes,
                eta=args.eta,
                lambdas=args.lambdas,
                rng=rng),
    'nn+lrp': \
        lambda args, rng: \
            NNWithLRP(
                w_sizes=args.w_sizes,
                eta=args.eta,
                lambdas=args.lambdas,
                method=args.explainer,
                method_kwargs={},
                rng=rng),
}


def _select_at_random(experiment, model, candidates):
    return model.rng.choice(sorted(candidates))


def _select_by_margin(experiment, model, candidates):
    margin = model.margin(experiment.X)
    nonzero_margin = np.ones_like(margin) * np.inf
    nonzero_margin[candidates] = margin[candidates]
    i = np.argmin(nonzero_margin)
    assert i in candidates
    return i


STRATEGIES = {
    'random': _select_at_random,
    'margin': _select_by_margin,
}


def _move_indices(dst, src, indices):
    dst, src = set(dst), set(src)
    assert all((i in src and i not in dst) for i in indices)
    dst = np.array(list(sorted(dst | set(indices))))
    src = np.array(list(sorted(src - set(indices))))
    return dst, src


def _get_correction(experiment, model, args, i):
    assert args.prop_corrected >= 0, 'prop_corrected must be non-negative'

    z = experiment.Z[i]
    z_hat = model.explain(experiment.X[i].reshape(1, -1)).ravel()

    # ignore the bias
    z_hat[-1] = z[-1]

    # compute the indices of the n_corrected features with largest diff
    # XXX in principle the choice should be noisy
    diff = np.abs(z - z_hat)
    n_inputs = diff.shape[0]

    if args.prop_corrected > 1:
        n_corrected = args.prop_corrected
        assert n_corrected <= n_inputs
    else:
        n_corrected = int(np.round(n_inputs * args.prop_corrected))

    indices = np.argsort(diff)[::-1][:n_corrected]

    # discretize the difference
    correction, mask = np.zeros_like(diff), np.zeros_like(diff)
    correction[indices] = z[indices]
    mask[indices] = 1

    return correction, mask


def _dump_explanations(experiment, model, basename, t, indices, suffix):
    for i in indices:
        path = basename + '__t={}__i={}__{}.png'.format(t, i, suffix)

        x = experiment.X[i].reshape(1, -1)
        experiment.dump_explanation(path,
                                    experiment.X[i],
                                    experiment.Z[i],
                                    model.explain(x).ravel())


def _evaluate(experiment, model, i, ts):

    # compute performance on the query instance
    if i >= 0:
        x_i = experiment.X[i].reshape(1, -1)
        z_i = experiment.Z[i].reshape(1, -1)
        y_i = np.array([experiment.y[i]])

        y_loss_i, z_loss_i = model.evaluate(x_i, z_i, y_i)
        prf_i = prfs(y_i.reshape(1, -1),
                     model.predict(x_i),
                     labels=[0, 1],
                     average='binary')[:3]
    else:
        y_loss_i, z_loss_i = -1, -1
        prf_i = (-1, -1, -1)

    # compute performance on the test set
    y_loss_ts, z_loss_ts = \
        model.evaluate(experiment.X[ts],
                       experiment.Z[ts],
                       experiment.y[ts])
    prf_ts = prfs(experiment.y[ts],
                  model.predict(experiment.X[ts]),
                  labels=[0, 1],
                  average='binary')[:3]

    return ([y_loss_i, z_loss_i] + list(prf_i) +
            [y_loss_ts, z_loss_ts] + list(prf_ts))


def _naive(experiment, model, kn, tr, ts, args, basename, rng):
    max_iters = args.max_iters
    if max_iters <= 0:
        max_iters = len(tr)

    select_query = STRATEGIES[args.strategy]

    # Fit an initial model on the known instances
    model.fit(experiment.X[kn],
              experiment.Z[kn],
              experiment.y[kn],
              n_epochs=args.n_epochs,
              batch_size=args.batch_size,
              warm=False)

    # Do the active learning dance
    corrections = np.zeros_like(experiment.Z)
    corrections_mask = np.zeros_like(experiment.Z)

    trace = []
    for t in range(1, max_iters + 1):
        if not len(tr):
            break

        runtime = time()

        # Select a query instance
        i = select_query(experiment, model, tr)
        kn, tr = _move_indices(kn, tr, [i])

        # Retrieve an explanation correction for this instance
        corrections[i], corrections_mask[i] = \
            _get_correction(experiment, model, args, i)

        x = experiment.X[i].reshape(1, -1)
        path = basename + '__t={}__instance={}'.format(t, i)

        # Dump the explanation of the query instance before training
        _dump_explanations(experiment, model, basename, t, [i], '0')

        # Re-train the model on all the supervision
        model.fit(experiment.X[kn],
                  corrections[kn],
                  experiment.y[kn],
                  mask=corrections_mask[kn],
                  n_epochs=args.n_epochs,
                  batch_size=args.batch_size,
                  warm=True)

        # Dump the explanation of the query instance after training
        _dump_explanations(experiment, model, basename, t, [i], '1')

        runtime = time() - runtime

        trace.append([i] + _evaluate(experiment, model, i, ts) + [runtime])
        message = ' '.join(['{:5.3f}'.format(s) for s in trace[-1][1:]])
        print('{:3d} : {}'.format(t, message))

    return trace


def _sample_centers(vectors, indices, args, rng):
    """Implementation of the kmeans++ seeding algorithm.

    See Appendix A of [1].
    """
    def dist2(x, z):
        return np.sum((x - z)**2)

    centers = [rng.choice(indices)]
    while len(centers) < args.n_queries_per_iter:
        dist_to_centers = [min(dist2(vectors[i], vectors[j]) for j in centers)
                           for i in indices]
        p = dist_to_centers / np.sum(dist_to_centers)
        i = rng.choice(indices, p=p)
        centers, indices = _move_indices(centers, indices, [i])
    return np.array(centers, dtype=int)


def _badge(experiment, model, kn, tr, ts, args, basename, rng):
    """Batch deep active learning.

    References
    ----------

    [1] "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower
         Bounds", Ash et al., 2019. (pre-print).
    """
    max_iters = args.max_iters
    if max_iters <= 0:
        max_iters = len(tr)

    # Fit an initial model on the known instances
    model.fit(experiment.X[kn],
              experiment.Z[kn],
              experiment.y[kn],
              n_epochs=args.n_epochs,
              batch_size=args.batch_size,
              warm=False)

    corrections = np.zeros_like(experiment.Z)
    corrections_mask = np.zeros_like(experiment.Z)

    # Do the badge dance
    trace = []
    for t in range(1, max_iters + 1):
        if not len(tr):
            break

        runtime = time()

        # Select a query instance
        n_examples = experiment.X.shape[0]
        fake_grads = rng.normal(0, 1, size=n_examples) # XXX
        centers = _sample_centers(fake_grads, tr, args, rng)
        kn, tr = _move_indices(kn, tr, centers)

        # Retrieve explanation corrections for the query instances
        for i in centers:
            corrections[i], corrections_mask[i] = \
                _get_correction(experiment, model, args, i)

        # Dump the explanations of the queries before training
        _dump_explanations(experiment, model, basename, t, centers, '0')

        # Re-train the model on all the supervision
        model.fit(experiment.X[kn],
                  experiment.Z[kn],
                  experiment.y[kn],
                  n_epochs=args.n_epochs,
                  batch_size=args.batch_size,
                  warm=True)

        # Dump the explanations of the queries after training
        _dump_explanations(experiment, model, basename, t, centers, '1')

        runtime = time() - runtime

        trace.append([-1] + _evaluate(experiment, model, -1, ts) + [runtime])
        message = ' '.join(['{:5.3f}'.format(s) for s in trace[-1][1:]])
        print('{:3d} : {}'.format(t, message))

    return trace


ALGORITHMS = {
    'naive': _naive,
    'badge': _badge,
}


def run(experiment, args, basename):
    rng = np.random.RandomState(args.seed)

    algo = ALGORITHMS[args.algorithm]
    model = MODELS[args.model](args, rng)

    traces = []
    for k, (kn, tr, ts) in enumerate(experiment.split(n_splits=args.n_splits,
                                                      prop_known=args.prop_known)):

        if args.passive:

            # Fit an initial model on the known instances
            model.fit(experiment.X[tr],
                      experiment.Z[tr],
                      experiment.y[tr],
                      n_epochs=args.n_epochs,
                      batch_size=args.batch_size,
                      warm=False)

            # Explain the test set and dump the explanations on-disk
            for i in ts:
                path = basename + '__instance={}'.format(i)
                x = experiment.X[i].reshape(1, -1)
                experiment.dump_explanation(path + '.png',
                                            experiment.X[i],
                                            experiment.Z[i],
                                            model.explain(x).ravel())

            quit()

        print('fold {} : #kn {}, #tr {}, #ts {}'.format(k + 1, len(kn), len(tr), len(ts)))
        traces.append(algo(experiment, model, kn, tr, ts, args,
                           basename + '__fold={}'.format(k), rng))

    return traces


def _get_basename(args):
    fields = [('m', args.model)]
    if args.model != 'senn':
        fields.append(('x', args.explainer))

    fields.extend([
        ('qss', args.strategy),
        ('n', args.n_examples),
        ('k', args.n_splits),
        ('p', args.prop_known),
        ('c', args.prop_corrected),
        ('T', args.max_iters),
        ('W', ','.join(map(str, args.w_sizes))),
        ('P', ','.join(map(str, args.phi_sizes))),
        ('e', args.eta),
        ('L', ','.join(map(str, args.lambdas))),
        ('E', args.n_epochs),
        ('B', args.batch_size),
        ('s', args.seed),
    ])

    basename = args.experiment + '__' + '__'.join([name + '=' + str(value)
                                                   for name, value in fields])
    return join('results', basename)


def main():
    import argparse

    fmt_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=fmt_class)
    parser.add_argument('algorithm', choices=sorted(ALGORITHMS.keys()),
                        help='name of the active learning algorithm')
    parser.add_argument('experiment', choices=sorted(EXPERIMENTS.keys()),
                        help='name of the experiment')
    parser.add_argument('model', type=str, choices=sorted(MODELS.keys()),
                        help='The model to use')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='RNG seed')

    group = parser.add_argument_group('evaluation')
    group.add_argument('--passive', action='store_true',
                       help='Learn on training set, dump explanations, quit')
    group.add_argument('-n', '--n-examples', type=int, default=None,
                       help='Only use this many examples from the dataset')
    group.add_argument('-k', '--n-splits', type=int, default=10,
                       help='Number of cross-validation folds')
    group.add_argument('-p', '--prop-known', type=float, default=0.05,
                       help='Proportion of examples known before interaction')
    group.add_argument('-T', '--max-iters', type=int, default=100,
                       help='Maximum number of learning iterations')

    group = parser.add_argument_group('interaction')
    group.add_argument('--strategy', choices=sorted(STRATEGIES.keys()),
                       default='random', help='The query selection strategy')
    group.add_argument('-c', '--prop-corrected', type=float, default=1.0,
                       help='Proportion of features corrected at each '
                            'iteration')
    group.add_argument('-q', '--n-queries-per-iter', type=int, default=1,
                       help='Number of queries per iteration (for badge).')

    group = parser.add_argument_group('model')
    group.add_argument('-X', '--explainer', choices=sorted(analyzers.keys()),
                       default='lrp.epsilon',
                       help='Explainer.  Only valid for nn+lrp.')
    group.add_argument('-W', '--w-sizes', type=int, nargs='+', default=[],
                       help='Shapes of the hidden layers for w. '
                            'If empty, w(x) = x.')
    group.add_argument('-P', '--phi-sizes', type=int, nargs='+', default=[],
                       help='Shapes of the hidden layers for phi. '
                            'If empty, phi(x) = x.')
    group.add_argument('-e', '--eta', type=float, default=0.1,
                       help='Learning rate')
    group.add_argument('-L', '--lambdas', type=float, nargs=2, default=(0.1, 0.01),
                       help='Hyperaparameters of the explanation loss and '
                            'explanation regularization, respectively. '
                            'The label loss has weight 1 - sum(lambdas).')
    group.add_argument('-E', '--n-epochs', type=int, default=100,
                       help='Number of epochs per iteration')
    group.add_argument('-B', '--batch-size', type=int, default=None,
                       help='Batch size')

    args = parser.parse_args()
    basename = _get_basename(args)

    np.seterr(all='raise')
    np.set_printoptions(precision=3, linewidth=80, threshold=2**32-1)

    # XXX we seed numpy and tensorflow RNGs too, just to be reproducible...
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    experiment = EXPERIMENTS[args.experiment](n_examples=args.n_examples,
                                              rng=args.seed)

    traces = run(experiment, args, basename)
    dump(basename + '__trace.pickle', {
             'args': args,
             'traces': traces,
         })


if __name__ == '__main__':
    main()
