import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import pairwise_distances
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
                method='lrp.z',
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


def _get_correction(experiment, model, args, i):
    x = experiment.X[i]
    z = experiment.Z[i]
    z_hat = model.explain(x.reshape(1, -1)).ravel()

    # ignore the bias
    #XXX needed for SENNs
    #z_hat[-1] = z[-1]

    # compute the indices of the n_corrected features with largest diff
    diff = np.abs(z - z_hat)
    indices = np.argsort(diff)[::-1][:args.n_corrected]

    # discretize the difference
    correction = np.zeros_like(diff)
    correction[indices] = np.sign(z_hat - z)[indices]
    return correction


def _move_indices(dst, src, indices):
    dst, src = set(dst), set(src)
    assert all((i in src and i not in dst) for i in indices)
    dst = np.array(list(sorted(dst | set(indices))))
    src = np.array(list(sorted(src - set(indices))))
    return dst, src


def _evaluate(experiment, model, i, ts):
    y_loss_i, z_loss_i = \
        model.evaluate(experiment.X[i].reshape(1, -1),
                       experiment.Z[i].reshape(1, -1),
                       np.array([experiment.y[i]]))
    y_loss_ts, z_loss_ts = \
        model.evaluate(experiment.X[ts],
                       experiment.Z[ts],
                       experiment.y[ts])
    return [y_loss_i, z_loss_i, y_loss_ts, z_loss_ts]


def _naive_al(experiment, model, kn, tr, ts, args):
    max_iters = args.max_iters
    if max_iters <= 0:
        max_iters = len(tr)

    model.fit(experiment.X[kn],
              experiment.Z[kn],
              experiment.y[kn],
              n_epochs=args.n_epochs,
              batch_size=args.batch_size,
              warm=False)

    select_query = STRATEGIES[args.strategy]
    corrections, trace = np.zeros_like(experiment.Z), []
    for t in range(max_iters):
        if not len(tr):
            break

        runtime = time()

        i = select_query(experiment, model, tr)
        kn, tr = _move_indices(kn, tr, [i])

        if args.n_corrected >= 1:
            corrections[i] = _get_correction(experiment, model, args, i)
            explanation_feedback = -corrections[kn]
            # c = 2(zhat - z) implies:
            # min_w <w, w + c>
            # = min_w <w, w + 2(zhat - z)>
            # = min_w ||w||^2 + 2<w, zhat - z>
            # = min_w ||w||^2 + 2<w, zhat - z> + const.
            # = min_w ||w||^2 + 2<w, zhat - z> + ||z||^2 - ||zhat||^2
            # = min_w ||w||^2 - 2<w, z> + ||z||^2 + 2<w, zhat> - ||zhat||^2
            # = min_w ||w||^2 + ||w||^2 - 2<w, z> + ||z||^2 - ||w||^2 + 2<w, zhat> - ||zhat||^2
            # = min_w ||w||^2 + ||w - z||^2 - ||w - zhat||^2
        else:
            explanation_feedback = experiment.Z[kn]
            # c = -2z implies:
            # <w, w + c>
            # = <w, w - 2z>
            # = ||w||^2 - 2<w, z>
            # = ||w - z||^2 + const.

        model.fit(experiment.X[kn],
                  explanation_feedback,
                  experiment.y[kn],
                  n_epochs=args.n_epochs,
                  batch_size=args.batch_size,
                  warm=True)

        runtime = time() - runtime

        trace.append([i] + _evaluate(experiment, model, i, ts) + [runtime])
        print('{:3d} : {}'.format(t, trace[-1]))

    return trace


ALGORITHMS = {
    'naive': _naive_al,
}


def eval_active(experiment, args):
    rng = np.random.RandomState(args.seed)

    algo = ALGORITHMS[args.algorithm]
    model = MODELS[args.model](args, rng)

    traces = []
    for k, (kn, tr, ts) in enumerate(experiment.split(n_splits=args.n_splits,
                                                      prop_known=args.prop_known)):
        print('fold {} : #kn {}, #tr {}, #ts {}'.format(k + 1, len(kn), len(tr), len(ts)))
        traces.append(algo(experiment, model, kn, tr, ts, args))

    return traces


def _get_basename(args):
    fields = [
        ('strategy', args.strategy),
        ('n', args.n_examples),
        ('k', args.n_splits),
        ('p', args.prop_known),
        ('c', args.n_corrected),
        ('T', args.max_iters),
        ('W', ','.join(map(str, args.w_sizes))),
        ('P', ','.join(map(str, args.phi_sizes))),
        ('e', args.eta),
        ('L', ','.join(map(str, args.lambdas))),
        ('E', args.n_epochs),
        ('B', args.batch_size),
        ('s', args.seed),
    ]

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
    parser.add_argument('--strategy', choices=sorted(STRATEGIES.keys()),
                        default='random', help='The query selection strategy')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='RNG seed')

    group = parser.add_argument_group('Evaluation')
    group.add_argument('-n', '--n-examples', type=int, default=None,
                       help='Only use this many examples from the dataset')
    group.add_argument('-k', '--n-splits', type=int, default=10,
                       help='Number of cross-validation folds')
    group.add_argument('-p', '--prop-known', type=float, default=0.05,
                       help='Proportion of examples known before interaction')
    group.add_argument('-c', '--n-corrected', type=int, default=1,
                       help='Proportion of features corrected at each '
                            'iteration')
    group.add_argument('-T', '--max-iters', type=int, default=100,
                       help='Maximum number of learning iterations')

    group = parser.add_argument_group('Model')
    group.add_argument('-W', '--w-sizes', type=int, nargs='+', default=[],
                       help='Shapes of the hidden layers for w. '
                            'If empty, w(x) = x.')
    group.add_argument('-P', '--phi-sizes', type=int, nargs='+', default=[],
                       help='Shapes of the hidden layers for phi. '
                            'If empty, phi(x) = x.')
    group.add_argument('-e', '--eta', type=float, default=0.1,
                       help='Learning rate')
    group.add_argument('-L', '--lambdas', type=float, nargs='+', default=(0.1, 0.01),
                       help='Hyperaparameters of the SENN model')
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

    traces = eval_active(experiment, args)
    dump(basename + '__trace.pickle', {
             'args': args,
             'traces': traces,
         })


if __name__ == '__main__':
    main()
