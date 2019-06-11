#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
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
    'full_full': \
        lambda args, rng: \
            FullFullSENN(
                w_sizes=args.w_sizes,
                phi_sizes=args.phi_sizes,
                eta=args.eta,
                lambdas=args.lambdas,
                rng=rng),
}


def _move(dst, src, i):
    dst, src = set(dst), set(src)
    assert i in src and not i in dst
    dst = np.array(list(sorted(dst | {i})))
    src = np.array(list(sorted(src - {i})))
    return dst, src


def _run_fold(experiment, model, args, kn, tr, ts):
    max_iters = args.max_iters
    if max_iters <= 0:
        max_iters = len(tr)

    model.fit(experiment.X[kn],
              experiment.Z[kn],
              experiment.y[kn],
              warm=False)

    trace = []
    for t in range(max_iters):
        runtime = time()

        if not len(tr):
            break

        # TODO implement better strategies
        i = model.rng.choice(sorted(tr))
        model.fit(experiment.X[i].reshape(1, -1),
                  experiment.Z[i].reshape(1, -1),
                  experiment.y[i].reshape(1, -1),
                  warm=True)
        kn, tr = _move(kn, tr, i)

        runtime = runtime - time()

        perf = roc_auc_score(experiment.y[ts], model.predict(experiment.X[ts]))
        trace.append((i, perf))

        print('Iter {t:3d}: {perf:5.3f} in {runtime:5.3f}s'.format(**locals()))

    return trace


def eval_active(experiment, args):
    rng = np.random.RandomState(args.seed)
    model = MODELS[args.model](args, rng)

    traces = []
    for kn, tr, ts in experiment.split(n_splits=args.n_folds,
                                       prop_known=args.prop_known):
        traces.append(_run_fold(experiment, model, args, kn, tr, ts))

    return traces


def _predict(model, X, y):
    return model.predict(X), model.explain(X)


def eval_passive(experiment, args):
    rng = np.random.RandomState(args.seed)
    model = MODELS[args.model](args, rng)
    basename = _get_basename(args)

    split = StratifiedShuffleSplit(n_splits=args.n_splits,
                                   test_size=args.prop_known,
                                   random_state=rng)

    traces = []
    for k, (tr, ts) in enumerate(split.split(experiment.X, experiment.y)):
        print('fold {} : #tr {}, #ts {}'.format(k + 1, len(tr), len(ts)))

        selected = (list(rng.choice(tr, size=5)) +
                    list(rng.choice(ts, size=5)))

        def callback(epoch, model):
            if epoch % 100 != 0:
                return

            perf = []
            for X, Z, y in (
                (experiment.X[ts], experiment.Z[ts], experiment.y[ts]),
                (experiment.X[tr], experiment.Z[tr], experiment.y[tr])):

                y_hat = model.predict(X)
                y_loss = model.loss_y(X, y)
                y_perf = list(prfs(y, y_hat, average='binary')[:3])

                Z_hat = model.explain(X)
                z_loss = model.loss_z(X, Z)

                perf.extend(y_perf + [y_loss, z_loss])

            print(Z[0])
            print(Z_hat[0])

            if args.experiment.startswith('color'):
                for i in selected:
                    path = basename + '__fold={}__instance={}__epoch={}.png'.format(k, i, epoch)
                    x = experiment.X[i].reshape(1, -1)
                    experiment.dump_explanation(path,
                                                experiment.X[i],
                                                experiment.Z[i],
                                                model.explain(x).ravel())

            print('perf =', perf)
            return perf

        trace = model.fit(experiment.X[tr],
                          experiment.Z[tr],
                          experiment.y[tr],
                          n_epochs=args.n_epochs,
                          batch_size=args.batch_size,
                          callback=callback,
                          warm=False)
        traces.append(trace)

        if args.experiment.startswith('xor'):
            print('Plotting...')
            path = basename + '__fold{}.png'.format(k)
            plot_xor(path, experiment, model)

        yhat_tr, zhat_tr = _predict(model, experiment.X[tr], experiment.y[tr])
        perfs_tr = prfs(experiment.y[tr], yhat_tr, average='binary')[:3]

        yhat_ts, zhat_ts = _predict(model, experiment.X[ts], experiment.y[ts])
        perfs_ts = prfs(experiment.y[ts], yhat_ts, average='binary')[:3]

        print('fold {} : tr perf: {}, ts perf: {}'.format(k+1, perfs_tr, perfs_ts))

    return traces


def _get_basename(args):
    fields = [
        ('passive', args.passive),
        ('n', args.n_examples),
        ('k', args.n_splits),
        ('p', args.prop_known),
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
    parser.add_argument('experiment', choices=sorted(EXPERIMENTS.keys()),
                        help='name of the experiment')
    parser.add_argument('model', type=str, choices=sorted(MODELS.keys()),
                        help='The model to use')
    parser.add_argument('--passive', action='store_true',
                        help='Stick to passive learning')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='RNG seed')

    group = parser.add_argument_group('Evaluation')
    group.add_argument('-n', '--n-examples', type=int, default=None,
                       help='Only use this many examples from the dataset')
    group.add_argument('-k', '--n-splits', type=int, default=10,
                       help='Number of cross-validation folds')
    group.add_argument('-p', '--prop-known', type=float, default=0.05,
                       help='Proportion of passively known examples; '
                            'It is used as the proportion of test '
                            'examples when using --passive')
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

    if args.passive:
        traces = eval_passive(experiment, args)
    else:
        traces = eval_active(experiment, args)
    dump(basename + '__trace.pickle', {
             'args': args,
             'traces': traces,
         })


if __name__ == '__main__':
    main()
