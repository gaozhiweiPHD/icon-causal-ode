"""Independent online episodes; metadata and labels are kept outside model Context."""
from dataclasses import dataclass
import numpy as np
import torch
from .ode_solvers import local_solution, coupled_solution


@dataclass
class Context:
    inputs: torch.Tensor
    outputs: torch.Tensor

    def subset(self, m):
        return Context(self.inputs[:, :m], self.outputs[:, :m])


def forcing_functions(rng, shape, time):
    t = time.reshape((1,)*len(shape)+(-1,))
    coeff = rng.normal(size=shape+(3, 2))*.45
    fourier = sum((coeff[..., j, 0, None]*np.sin(2*np.pi*(j+1)*t)
                   + coeff[..., j, 1, None]*np.cos(2*np.pi*(j+1)*t))/(j+1)**2 for j in range(3))
    low = rng.uniform(-1, 1, shape)[..., None]*np.sin(rng.uniform(.2, 2, shape)[..., None]*t
                                                   +rng.uniform(0, 6.28, shape)[..., None])
    slow = rng.uniform(-1, 1, shape)[..., None]*np.exp(-rng.uniform(.1, 1.5, shape)[..., None]*t)
    slow += rng.uniform(-.5, .5, shape)[..., None]*t
    kind = rng.integers(0, 3, shape)[..., None]
    u = np.where(kind == 0, fourier, np.where(kind == 1, low, slow))
    # Fixed deterministic scale convention, no target-dependent normalization.
    return u/np.maximum(1., np.max(abs(u), axis=-1, keepdims=True))


class EpisodeGenerator:
    def __init__(self, seed, cfg, device='cpu'):
        self.rng = np.random.default_rng(seed)
        self.time = np.linspace(0, cfg['T'], cfg['time_points'])
        self.m = cfg['num_demonstrations']
        self.device = device

    def tensor(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def sample(self, batch, split='id', strength=None, coupled=True):
        if split not in ('id', 'ood'):
            raise ValueError('split must be id or ood')
        if batch < 1:
            raise ValueError('batch must be positive')
        r = self.rng
        if split == 'ood':
            # Every decay and coupling lies outside the corresponding ID range.
            decay = np.where(r.random((batch, 2)) < .5, r.uniform(.25, .5, (batch, 2)),
                             r.uniform(1.5, 1.9, (batch, 2)))
            coupling = r.uniform(.3, .55, (batch, 2))*r.choice([-1, 1], (batch, 2))
        else:
            decay = r.uniform(.6, 1.4, (batch, 2))
            coupling = r.uniform(-.25, .25, (batch, 2))
        if strength is not None:
            coupling = strength*r.choice([-1, 1], (batch, 2))
        p = np.concatenate((decay, coupling), axis=1)
        episode = {'parameters': p, 'time': self.time}
        for side in (1, 2):
            j = side-1
            z = r.uniform(-1, 1, (batch, self.m+1))
            u = forcing_functions(r, z.shape, self.time)
            target = local_solution(decay[:, j, None], coupling[:, j, None], z, u, self.time)
            inputs = np.stack((np.broadcast_to(z[..., None], u.shape), u), -1)
            episode[f'context{side}'] = Context(self.tensor(inputs[:, :self.m]),
                                               self.tensor(target[:, :self.m, :, None]))
            episode[f'query_{side}_input'] = self.tensor(inputs[:, self.m])
            episode[f'query_{side}_target'] = self.tensor(target[:, self.m, :, None])
            # Metadata for audits only. All demonstrations use these same coefficients.
            episode[f'context{side}_parameters'] = np.broadcast_to(p[:, [j, j+2]][:, None],
                                                                  (batch, self.m, 2)).copy()
        initial = r.uniform(-1, 1, (batch, 2))
        episode['x0'], episode['y0'] = self.tensor(initial[:, 0]), self.tensor(initial[:, 1])
        if coupled:
            truth = coupled_solution(p, initial, self.time)
            episode['x_true'], episode['y_true'] = self.tensor(truth[..., :1]), self.tensor(truth[..., 1:])
        return episode


def batch_interface(e):
    return {**{f'context_{j}_{name}': getattr(e[f'context{j}'], attr)
               for j in (1, 2) for name, attr in [('input', 'inputs'), ('output', 'outputs')]},
            **{k: v for k, v in e.items() if k.startswith('query_')}}
