#!/usr/bin/env python3

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
    'full_full': \
        lambda args, rng: \
            FullFullSENN(
                w_sizes=args.w_sizes,
                phi_sizes=args.phi_sizes,
                eta=args.eta,
                lambdas=args.lambdas,
                rng=rng),
}


def _select_at_random(experiment, model, candidates):
    return model.rng.choice(sorted(candidates))


def _select_by_margin(experiment, model, candidates):
    margin = model.predict_margin(experiment.X)
    nonzero_margin = np.ones_like(margin) * np.inf
    nonzero_margin[candidates] = margin[candidates]
    return np.argmin(nonzero_margin)


STRATEGIES = {
    'random': _select_at_random,
    'margin': _select_by_margin,
}



def _move(dst, src, i):
    dst, src = set(dst), set(src)
    assert i in src and not i in dst
    dst = np.array(list(sorted(dst | {i})))
    src = np.array(list(sorted(src - {i})))
    return dst, src


def _run_fold_active(experiment, model, args, kn, tr, ts):
    max_iters = args.max_iters
    if max_iters <= 0:
        max_iters = len(tr)

    select_query = STRATEGIES[args.strategy]

    def evaluate(i):
        xi = experiment.X[i].reshape(1, -1)
        zi = experiment.Z[i].reshape(1, -1)
        y_loss_i = model.loss_y(xi, experiment.y[ts])
        z_loss_i = model.loss_z(xi, zi)

        y_loss_ts = model.loss_y(experiment.X[ts], experiment.y[ts])
        z_loss_ts = model.loss_z(experiment.X[ts], experiment.Z[ts])

        return y_loss_i, z_loss_i, y_loss_ts, z_loss_ts

    trace = []
    model.fit(experiment.X[kn],
              experiment.Z[kn],
              experiment.y[kn],
              n_epochs=args.n_epochs,
              batch_size=args.batch_size,
              warm=False)

    for t in range(max_iters):
        if not len(tr):
            break

        runtime = time()
        i = select_query(experiment, model, tr)
        kn, tr = _move(kn, tr, i)

        model.fit(experiment.X[kn],
                  experiment.Z[kn],
                  experiment.y[kn],
                  n_epochs=args.n_epochs,
                  batch_size=args.batch_size,
                  warm=True)

        runtime = time() - runtime

        trace.append([i] + list(evaluate(i)) + [runtime])
        print('{:3d} : {}'.format(t, trace[-1]))

    return trace


def eval_active(experiment, args):
    rng = np.random.RandomState(args.seed)
    model = MODELS[args.model](args, rng)

    traces = []
    for k, (kn, tr, ts) in enumerate(experiment.split(n_splits=args.n_splits,
                                                      prop_known=args.prop_known)):
        print('fold {} : #kn {}, #tr {}, #ts {}'.format(k + 1, len(kn), len(tr), len(ts)))
        traces.append(_run_fold_active(experiment, model, args, kn, tr, ts))

    return traces


def _whatever_at_k(z_senn, z_lime, k):
    # prop of best k elements in z_lime that are in z_senn
    highest_lime = set(np.argsort(z_lime)[-k:]) # indices of k largest elements
    highest_senn = set(np.argsort(z_senn)[-k:]) # indices of k largest elements
    return len(highest_lime & highest_senn)


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

        selection = rng.choice(ts, size=10)

        def callback(epoch, model):
            if epoch % 250 != 0:
                return

            perf = []
            for X, Z, y in (
                (experiment.X[ts], experiment.Z[ts], experiment.y[ts]),
                (experiment.X[tr], experiment.Z[tr], experiment.y[tr])):

                yhat = model.predict(X)

                y_loss = model.loss_y(X, y)
                z_loss = model.loss_z(X, Z)
                y_perf = list(prfs(y, yhat, average='binary')[:3])

                perf.extend(y_perf + [y_loss, z_loss])

            #print(Z[0])
            #print(Z_hat[0])

            if args.experiment.startswith('color') and args.record_lime:

                similarities, dispersions, times_senn, times_lime = [], [], [], []
                for i in selection:
                    x = experiment.X[i].reshape(1, -1)

                    z_senn, t_senn = model.explain(x, return_runtime=True)
                    z_senn = z_senn.ravel()
                    z_lime, Z_lime, t_lime = \
                        experiment.explain_lime(model, tr, i,
                                                n_repeats=args.lime_repeats,
                                                n_samples=args.lime_samples,
                                                n_features=args.lime_features)
                    times_senn.append(t_senn)
                    times_lime.append(t_lime)

                    n_nonzeros = len(np.nonzero(z_lime)[0])
                    similarities.append(_whatever_at_k(z_senn, z_lime, n_nonzeros))

                    n_repeats = len(Z_lime)
                    dispersions.append(1 / (n_repeats * (n_repeats - 1)) * \
                                       np.sum(pairwise_distances(Z_lime, Z_lime)))

                    path = basename + '__fold={}__instance={}__epoch={}'.format(k, i, epoch)
                    experiment.dump_explanation(path + '_senn.png',
                                                experiment.X[i],
                                                experiment.Z[i],
                                                z_senn)
                    experiment.dump_explanation(path + '_lime.png',
                                                experiment.X[i],
                                                experiment.Z[i],
                                                z_lime)

                perf.extend([
                    np.mean(similarities),
                    np.mean(dispersions),
                    np.mean(times_senn),
                    np.mean(times_lime),
                ])

            print('epoch {} : {}'.format(epoch, perf))
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

    return traces


def _get_basename(args):
    fields = [
        ('strategy', args.strategy),
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

    if args.record_lime:
        fields += [
            ('limer', args.lime_repeats),
            ('limes', args.lime_samples),
            ('limef', args.lime_features),
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
    parser.add_argument('--strategy', choices=sorted(STRATEGIES.keys()),
                        default='random', help='The query selection strategy')
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

    group = parser.add_argument_group('LIME (colors{0,1} only)')
    group.add_argument('--record-lime', action='store_true',
                       help='Record LIME performance')
    group.add_argument('--lime-repeats', type=int, default=1,
                       help='Number of times LIME is called')
    group.add_argument('--lime-samples', type=int, default=100,
                       help='Number of samples used by LIME')
    group.add_argument('--lime-features', type=int, default=None,
                       help='Number of features LIME can use at most')

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
