#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from calimocho import load


def _get_style(args, trace_args):
    # XXX stub

    label = ' '.join([
        trace_args.experiment,
        trace_args.explainer,
        trace_args.strategy
    ])

    color = {
        0: '#FF00FF',
        1: '#0000FF',
        2: '#FF0000',
        3: '#00FF00',
        4: '#000000',
    }[trace_args.n_corrected]
    linestyle = '-'

    return label, color, linestyle


def _draw(args, traces, traces_args):

    CONFIGS = [
        ('query index', 0),
        ('$\ell_y$ on query', 0),
        ('$\ell_z$ on query', 0),
        ('$Pr$ on query', 1),
        ('$Rc$ on query', 1),
        ('$F_1$ on query', 1),
        ('$\ell_y$ on test set', 0),
        ('$\ell_z$ on test set', 0),
        ('$Pr$ on test set', 1),
        ('$Rc$ on test set', 1),
        ('$F_1$ on test set', 1),
        ('runtime', 0),
    ]

    n_pickles, n_folds, n_iters, n_measures = traces.shape
    assert n_measures == len(CONFIGS)

    fig, axes = plt.subplots(n_measures, 1, figsize=(5, n_measures*3))

    for m, ax in enumerate(axes):
        name, max_y = CONFIGS[m]

        ax.set_xlabel('Epochs')
        ax.set_ylabel(name)

        for p in range(n_pickles):
            perf = traces[p, :, :, m]
            label, color, linestyle = _get_style(args, trace_args[p])

            x = np.arange(n_iters)
            y = np.mean(perf, axis=0)
            yerr = np.std(perf, axis=0) / np.sqrt(n_folds)

            ax.plot(x, y,
                    linewidth=2, label=label)
            ax.fill_between(x, y - yerr, y + yerr,
                            color=color, alpha=0.35, linewidth=0)

            max_y = max(max_y, 1.1 * y.max())

        ax.set_ylim(0, max_y)
        ax.legend(loc='upper right', fontsize=8, shadow=False)

    fig.savefig(args.basename + '.png',
                bbox_inches='tight',
                pad_inches=0)
    del fig


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

    _draw(args, traces, trace_args)
