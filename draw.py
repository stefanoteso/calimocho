#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from calimocho import load


def get_style(args):
    arch = ';'.join(map(str, args.w_sizes)) + '|' + ';'.join(map(str, args.phi_sizes))
    l0, l1, l2 = 1 - sum(args.lambdas), args.lambdas[0], args.lambdas[1]
    key = '{} {} $\lambda_0={:3.1f}$ $\lambda_1={:3.1f}$ $\lambda_2={:3.1f}$'.format(
               args.experiment, arch, l0, l1, l2)
    if args.record_lime:
        key += ' lr={} ls={} lf={}'.format(args.lime_repeats, args.lime_samples, args.lime_features)
    return key


def _draw(args, traces, traces_args):
    n_pickles, n_folds, n_iters, n_measures = traces.shape

    if n_measures == 14: # passive
        n_rows = 5
    elif n_measures == 6: # active
        n_rows = 6

    n_cols = int(np.ceil(n_measures / n_rows))
    w, h = 9 * n_cols, 6 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, figsize=(w, h))
    axes = axes.reshape((n_rows, n_cols))

    for m in range(n_measures):
        r, c = m % n_rows, m // n_rows

        axes[r,c].set_xlabel('Epochs')
        axes[r,c].set_ylabel(str(m))

        x = np.arange(n_iters)

        max_y = 1
        for p in range(n_pickles):
            perf = traces[p, :, :, m]
            label = get_style(trace_args[p])

            y = np.mean(perf, axis=0)
            max_y = max(max_y, y.max() + 0.1)
            yerr = np.std(perf, axis=0) / np.sqrt(n_folds)

            axes[r,c].plot(x, y, linewidth=2, label=label)
            axes[r,c].fill_between(x, y - yerr, y + yerr, alpha=0.35, linewidth=0)

        axes[r,c].set_ylim(0, max_y)
        axes[r,c].legend(loc='upper right',
                         fontsize=8,
                         shadow=False)

    fig.savefig(args.basename + '.png',
                bbox_inches='tight',
                pad_inches=0)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('basename', type=str,
                        help='basename of the loss/time PNG plots')
    parser.add_argument('pickles', type=str, nargs='+',
                        help='comma-separated list of pickled results')
    args = parser.parse_args()

    traces, trace_args = [], []
    for path in args.pickles:
        data = load(path)
        traces.append(data['traces'])
        trace_args.append(data['args'])
    traces = np.array(traces)

    if any(args.passive for args in trace_args):
        assert all(args.passive for args in trace_args)
    else:
        assert all(not args.passive for args in trace_args)

    _draw(args, traces, trace_args)
