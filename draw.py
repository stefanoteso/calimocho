#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from calimocho import load


def get_style(args):
    arch = ';'.join(map(str, args.w_sizes)) + '|' + ';'.join(map(str, args.phi_sizes))
    return '{} {} {} $\lambda_1={}$ $\lambda_2={}$'.format(
               args.experiment, arch, args.use_corrections, args.lambda1, args.lambda2)


def draw(args, traces, trace_args):
    print(np.array(traces).shape)
    quit()

    n_pickles, n_folds, n_iters, _ = traces.shape

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Performance')

    for p in range(traces.shape[0]):
        perf = traces[p, :, :, 1]
        label = get_style(trace_args[p])

        y = np.mean(perf, axis=0)
        yerr = np.std(perf, axis=0) / np.sqrt(perf.shape[0])
        x = np.arange(y.shape[0])

        ax.plot(x, y, linewidth=2, label=label)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.35, linewidth=0)

    legend = ax.legend(loc='lower right',
                       fontsize=16,
                       shadow=False)
    fig.savefig(args.basename + '.png',
                bbox_inches='tight',
                pad_inches=0)


def _draw_passive(args, traces, traces_args):
    n_pickles, n_folds, n_iters, n_measures = traces.shape

    MEASURES = [2, 3, 4]
    n_measures = len(MEASURES)

    w, h = 9, 6 * n_measures
    fig, axes = plt.subplots(n_measures, 1, sharex=True, figsize=(w, h))

    for i, m in enumerate(MEASURES):
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(str(m))

        x = np.arange(n_iters)

        max_y = 1
        for p in range(n_pickles):
            perf = traces[p, :, :, m]
            label = get_style(trace_args[p])

            y = np.mean(perf, axis=0)
            max_y = max(max_y, y.max() + 0.1)
            yerr = np.std(perf, axis=0) / np.sqrt(n_folds)

            axes[i].plot(x, y, linewidth=2, label=label)
            axes[i].fill_between(x, y - yerr, y + yerr, alpha=0.35, linewidth=0)

        axes[i].set_ylim(0, max_y)
        axes[i].legend(loc='upper right',
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
        _draw_passive(args, traces, trace_args)
    else:
        raise NotImplementedError()


    # # Traces may have different lenghts, let's cap them to the shortest one
    # # XXX this code is horrible
    # min_len = None
    # for trace in traces:
    #     for fold in trace:
    #         min_len = len(fold) if min_len is None else min(min_len, len(fold))

    # shortened_traces = []
    # for trace in traces:
    #     shortened_folds = []
    #     for fold in trace:
    #         shortened_folds.append(fold[:min_len])
    #     shortened_traces.append(shortened_folds)

    # draw(args, np.array(shortened_traces), trace_args)
