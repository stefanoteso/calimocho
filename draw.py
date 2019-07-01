#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from calimocho import load


def get_style(args, trace_args):
    d = vars(trace_args)

    rule = int(trace_args.experiment[-1])
    arch = ';'.join(map(str, trace_args.w_sizes)) + '|' + ';'.join(map(str, trace_args.phi_sizes))
    l0, l1, l2 = 1 - sum(trace_args.lambdas), trace_args.lambdas[0], trace_args.lambdas[1]

    color, linestyle = '#000000', '-'
    if args.style == 'q1':
        key = '$r = {}$, $s = {}$'.format(trace_args.lime_repeats,
                                          trace_args.lime_samples)
        color = {
            100: '#FF0000',
            1000: '#0000FF',
        }[trace_args.lime_samples]

        linestyle = {
            5: '-',
            10: '-.',
            25: ':',
        }[trace_args.lime_repeats]

    elif args.style == 'q2':
        key = '$\lambda={:3.1f}$, $c = {}$'.format(l0, trace_args.n_corrected)

        color = {
            1: '#0000FF',
            2: '#FF0000',
            3: '#00FF00',
            4: '#000000',
        }[trace_args.n_corrected]

        linestyle = {
            0.0: '-',
            0.9: '-',
            1.0: ':',
        }[l0]
        if l0 == 1.0:
            color = '#7F7F7F'

    elif args.style == 'q3':
        n_layers = len(trace_args.w_sizes)
        key = '$L = {}$, $\lambda={:3.1f}$'.format(n_layers, l0)

        color = {
            1: '#0000FF',
            3: '#FF0000',
            5: '#00FF00',
        }[n_layers]

        linestyle = {
            0.0: '-',
            0.9: '-',
            1.0: ':',
        }[l0]

    else:
        key = '{} {} $\lambda_0={:3.1f}$ $\lambda_1={:3.1f}$ $\lambda_2={:3.1f}$'.format(
                   trace_args.experiment, arch, l0, l1, l2)
        if 'n_corrected' in d and trace_args.n_corrected:
            key += ' c={}'.format(trace_args.n_corrected)
        if trace_args.record_lime:
            key += ' lr={} ls={} lf={}'.format(trace_args.lime_repeats, trace_args.lime_samples, trace_args.lime_features)

    return key, color, linestyle


def _draw(args, traces, traces_args):
    n_pickles, n_folds, n_iters, n_measures = traces.shape

    if n_measures == 14:
        NAMES = [
            'Pr', 'Rc', '$F_1$', '$\ell_Y$', '$\ell_Z$',
            'Pr', 'Rc', '$F_1$', '$\ell_Y$', '$\ell_Z$',
            'Sim', 'Dis', 'Time (SENN)', 'Time (LIME)'
        ]
    elif n_measures == 6:
        NAMES = [
            'Instance',
            '$\ell_Y$ (i)', '$\ell_Z$ (i)',
            '$\ell_Y$ (Test)', '$\ell_Z$ (Test)',
            'Time'
        ]
    else:
        print(n_measures)
        raise RuntimeError()

    for m in range(n_measures):
        if args.measures and not m in args.measures:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))

        ax.set_xlabel('Epochs')
        ax.set_ylabel(NAMES[m])

        max_y = 1
        for p in range(n_pickles):
            perf = traces[p, :, :, m]
            label, color, linestyle = get_style(args, trace_args[p])

            x = np.arange(n_iters)
            if trace_args[p].passive:
                x *= 250
            y = np.mean(perf, axis=0)
            max_y = max(max_y, y.max() + 0.1)
            yerr = np.std(perf, axis=0) / np.sqrt(n_folds)

            ax.plot(x, y, linewidth=2, label=label, color=color, linestyle=linestyle)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=0.35, linewidth=0)

        ax.set_ylim(0, max_y)
        ax.legend(loc='upper right', fontsize=8, shadow=False)

        fig.savefig(args.basename + '__{}.png'.format(m),
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
    parser.add_argument('-m', '--measures', nargs='+', type=int, default=None,
                        help='index of the measure to be plotted')
    parser.add_argument('-s', '--style', type=str, default=None,
                        help='legend and line style')
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
