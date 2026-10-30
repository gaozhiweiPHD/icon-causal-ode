"""Evaluate one causal checkpoint on fresh ID/OOD episodes and KV controls."""
import argparse
import csv
from pathlib import Path

import torch

from .base import EpisodeGenerator, pair_error, slice_episode, stats, write_json
from .models import from_checkpoint
from .training import setup


@torch.no_grad()
def evaluate(model, cfg, episodes, rounds, controls, device, batch_size=32):
    if episodes < 1 or batch_size < 1 or not rounds or min(rounds) < 0:
        raise ValueError('positive episode/batch counts and nonnegative rounds required')
    if 'shuffled' in controls and (min(batch_size, episodes) < 2 or episodes % batch_size == 1):
        raise ValueError('shuffled control requires at least two episodes in every batch')
    model.eval()
    rows, summary = [], []
    for split, seed_key in (('id', 'test_seed'), ('ood', 'ood_seed')):
        # Draw the entire split before slicing, preserving the original RNG protocol.
        episode = EpisodeGenerator(cfg[seed_key], cfg, device).sample(episodes, split=split)
        for control in controls:
            for start in range(0, episodes, batch_size):
                e = slice_episode(episode, start, start + batch_size)
                states, _, _, _ = model(e['context1'], e['context2'], e['x0'], e['y0'],
                    rounds=max(rounds), control=control, return_hidden=True)
                for depth in rounds:
                    errors = pair_error(states[depth], (e['x_true'], e['y_true']))
                    rows.extend(dict(split=split, control=control, round=depth,
                                     episode=start + j, relative_l2=float(error))
                                for j, error in enumerate(errors))
            for depth in rounds:
                errors = [r['relative_l2'] for r in rows
                          if r['split'] == split and r['control'] == control and r['round'] == depth]
                summary.append(dict(split=split, control=control, round=depth, **stats(errors)))
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('pretrained/causal_kv_r3.pt'))
    parser.add_argument('--output', type=Path, default=Path('outputs/evaluation'))
    parser.add_argument('--episodes', type=int, default=1024)
    parser.add_argument('--rounds', type=int, nargs='+', default=[0, 1, 2, 3, 5, 10])
    parser.add_argument('--controls', nargs='+', choices=['real', 'masked', 'zero', 'shuffled'],
                        default=['real', 'masked'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    args = parser.parse_args()
    device = setup(device=args.device)
    model, saved = from_checkpoint(args.checkpoint, device)
    rows, summary = evaluate(model, saved['config'], args.episodes, sorted(set(args.rounds)),
                             args.controls, device, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'per_episode.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / 'summary.json', dict(
        checkpoint=str(args.checkpoint), checkpoint_step=saved['step'],
        trained_rounds=saved['rounds'], episodes_per_split=args.episodes,
        batch_size=args.batch_size, test_seed=saved['config']['test_seed'],
        ood_seed=saved['config']['ood_seed'], metrics=summary))
    for row in summary:
        print(f"{row['split']:3} {row['control']:8} R={row['round']:2}: "
              f"mean relative L2 = {100 * row['mean']:.4f}% (n={row['n']})")


if __name__ == '__main__':
    main()
