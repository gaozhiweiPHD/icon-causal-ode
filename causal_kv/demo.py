"""Plot learned trajectories against matrix-exponential coupled ODE solutions."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .base import EpisodeGenerator, pair_error, write_json
from .models import from_checkpoint
from .training import setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('pretrained/causal_kv_r3.pt'))
    parser.add_argument('--output', type=Path, default=Path('outputs/demo'))
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--split', choices=['id', 'ood'], default='id')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='cpu')
    args = parser.parse_args()
    device = setup(device=args.device)
    model, saved = from_checkpoint(args.checkpoint, device)
    e = EpisodeGenerator(args.seed, saved['config'], device).sample(4, split=args.split)
    with torch.no_grad():
        x, y = model(e['context1'], e['context2'], e['x0'], e['y0'], rounds=args.rounds)
    errors = pair_error((x, y), (e['x_true'], e['y_true']))
    arrays = dict(time=e['time'], parameters=e['parameters'],
                  x=x.cpu().numpy(), y=y.cpu().numpy(),
                  x_true=e['x_true'].cpu().numpy(), y_true=e['y_true'].cpu().numpy())
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / 'trajectories.npz', **arrays)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        for name, color in (('x', '#1d4ed8'), ('y', '#c2410c')):
            ax.plot(e['time'], arrays[name + '_true'][j, :, 0], color=color,
                    linewidth=2, label=f'{name}: exact')
            ax.plot(e['time'], arrays[name][j, :, 0], '--', color=color,
                    linewidth=1.6, label=f'{name}: causal ICON')
        ax.set(title=f'Episode {j + 1} | relative error {100 * errors[j]:.3f}%',
               xlabel='Time', ylabel='State')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'Coupled ODE · causal KV communication · R={args.rounds}', fontsize=13)
    fig.savefig(args.output / 'trajectories.png', dpi=160)
    plt.close(fig)
    write_json(args.output / 'metrics.json', dict(seed=args.seed, split=args.split,
               rounds=args.rounds, relative_l2=errors.tolist(), mean_relative_l2=float(errors.mean())))
    print(f'Mean relative L2: {100 * errors.mean():.4f}%')
    print(f'Figure: {args.output / "trajectories.png"}')


if __name__ == '__main__':
    main()
