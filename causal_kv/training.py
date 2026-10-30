import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from importlib.resources import files
import numpy as np
import torch
import yaml
from .base import (EpisodeGenerator, slice_episode, local_prediction,
                   pair_error, stats, write_json)
from .models import random_pair


def setup(seed=1729, device='auto'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    return torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device)


def read_config(path=None):
    resource = Path(path) if path is not None else files('causal_kv').joinpath('default.yaml')
    return yaml.safe_load(resource.read_text())


def local_losses(model, e):
    return [torch.mean((local_prediction(agent.base, e, side)-e[f'query_{side}_target'])**2)
            for side, agent in enumerate(model.agents, 1)]


def coupled_losses(pred, e):
    return [torch.mean((pred[j]-e[name])**2) for j, name in enumerate(('x_true', 'y_true'))]


def optimizer(model, cfg):
    base = [p for a in model.agents for p in a.base.parameters()]
    comm = [p for a in model.agents for p in a.communication.parameters()]
    return torch.optim.AdamW([{'params': base, 'lr': cfg['base_learning_rate'], 'kind': 'base'},
                             {'params': comm, 'lr': cfg['communication_learning_rate'], 'kind': 'communication'}],
                            weight_decay=cfg['weight_decay'])


def schedule(opt, step, cfg):
    fraction = max(0., (step-cfg['warmup_steps'])/max(1, cfg['joint_steps']-cfg['warmup_steps']))
    ratio = (step/cfg['warmup_steps'] if step <= cfg['warmup_steps'] else
             cfg['minimum_lr_fraction']+(1-cfg['minimum_lr_fraction'])*.5*(1+math.cos(math.pi*min(1., fraction))))
    for group in opt.param_groups:
        group['lr'] = cfg[group['kind']+'_learning_rate']*ratio


def stage(model, step, cfg):
    for a in model.agents:
        a.base.requires_grad_(True)
        a.communication.requires_grad_(True)


@torch.no_grad()
def validate(model, coupled, local, cfg, rounds):
    model.eval()
    totals, localtotals, errors, sizes = [], [], [], []
    for start in range(0, len(coupled['x0']), cfg['batch_size']):
        e = slice_episode(coupled, start, start+cfg['batch_size'])
        le = slice_episode(local, start, start+cfg['batch_size'])
        pred = model(e['context1'], e['context2'], e['x0'], e['y0'], rounds=rounds)
        sizes.append(len(e['x0']))
        totals.append(sum(coupled_losses(pred, e)).item())
        localtotals.append(sum(local_losses(model, le)).item())
        errors.extend(pair_error(pred, (e['x_true'], e['y_true'])).tolist())
    return {'val_coupled_mse': float(np.average(totals, weights=sizes)), 'val_local_mse': float(np.average(localtotals, weights=sizes)),
            'val_relative_l2': float(np.mean(errors)), 'val_relative_l2_p90': float(np.quantile(errors, .9)),
            'selection_score': float(np.average(totals, weights=sizes)+cfg['mu']*np.average(localtotals, weights=sizes))}


def gradient_audit(model, episode, rounds):
    model.requires_grad_(True).train()
    states, hidden, _, _ = model(episode['context1'], episode['context2'], episode['x0'], episode['y0'],
                               rounds=rounds, return_hidden=True)
    losses = coupled_losses(states[-1], episode)
    norms = []
    for receiver, sender in ((0, 1), (1, 0)):
        gradients = torch.autograd.grad(losses[receiver], tuple(model.agents[sender].parameters()),
                                        allow_unused=True, retain_graph=True)
        norm = float(torch.sqrt(sum(g.square().sum() for g in gradients if g is not None)))
        norms.append(norm)
        assert norm > 0 and math.isfinite(norm), 'cross-agent gradient absent'
    # Directly isolate one outgoing projected packet. No other cross-agent input.
    h1, h2 = hidden[0]
    c1, c2 = (a.communication for a in model.agents)
    q1, m1 = c1.project(h1)
    q2, m2 = c2.project(h2)
    o1, _ = c1.update(h1, q1, m1, m2)
    o2, _ = c2.update(h2, q2, m2, m1)
    packet_norms = []
    for output, message, receiver, target in ((o1, m2, 0, 'x_true'), (o2, m1, 1, 'y_true')):
        initial = episode['x0' if receiver == 0 else 'y0']
        loss = (model.agents[receiver].decode(output, initial)-episode[target]).square().mean()
        grads = torch.autograd.grad(loss, (message.key, message.value), retain_graph=True)
        packet_norms.append([float(g.norm()) for g in grads])
        assert min(packet_norms[-1]) > 0
    return {'rounds': rounds, 'loss_x_to_agent2_parameter_gradient_norm': norms[0],
            'loss_y_to_agent1_parameter_gradient_norm': norms[1],
            'outgoing_KV_gradient_norms_sender2_then_sender1': packet_norms,
            'remote_KV_detached': False}


def save(path, model, cfg, rounds, step, score, opt=None, generators=None, history=None):
    state = {'model': model.state_dict(), 'config': cfg, 'rounds': rounds, 'mode': model.mode,
             'step': step, 'score': score}
    if opt is not None:
        state.update(optimizer=opt.state_dict(), generator_rng=[g.rng.bit_generator.state for g in generators],
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            history=history, python_rng=random.getstate(), numpy_rng=np.random.get_state())
    temporary = Path(str(path)+'.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def restore(path, model, opt, generators):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    opt.load_state_dict(state['optimizer'])
    for g, rng in zip(generators, state['generator_rng']):
        g.rng.bit_generator.state = rng
    torch.set_rng_state(state['torch_rng'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    if state['cuda_rng'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def hashes():
    package = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.glob('*.py'))}


def train(out, cfg, device, rounds=3, mode='concat', until=None, resume=False):
    label = f'{mode}_r{rounds}'
    out = Path(out)
    work = out/'runs'/label
    if not resume and work.exists() and any(work.iterdir()):
        raise FileExistsError(f'{work} already exists; choose a new output or use --resume')
    if resume:
        previous = torch.load(work/'checkpoints/resume.pt', map_location='cpu', weights_only=False)
        if previous['config'] != cfg or previous['rounds'] != rounds or previous['mode'] != mode:
            raise ValueError('resume requires the same configuration and communication depth')
    stop = cfg['joint_steps'] if until is None else min(until, cfg['joint_steps'])
    if stop < 1 or (resume and stop <= previous['step']):
        raise ValueError('requested stopping step must be positive and exceed the saved step')
    for d in ('checkpoints', 'results', 'configs'):
        (work/d).mkdir(parents=True, exist_ok=True)
    (work/'configs/training.yaml').write_text(yaml.safe_dump(dict(cfg, rounds=rounds, mode=mode)))
    source_hashes = hashes()
    write_json(work/'results/source_hashes.json', source_hashes)
    model = random_pair(cfg, device, mode)
    opt = optimizer(model, cfg)
    generators = [EpisodeGenerator(cfg[k], cfg, device) for k in ('coupled_seed', 'local_seed')]
    val = EpisodeGenerator(cfg['validation_seed'], cfg, device).sample(cfg['validation_episodes'])
    lv = EpisodeGenerator(cfg['local_validation_seed'], cfg, device).sample(cfg['validation_episodes'], coupled=False)
    start, history, best = 0, [], math.inf
    if resume:
        state = restore(work/'checkpoints/resume.pt', model, opt, generators)
        start, best, history = state['step'], state['score'], state['history']
        assert state['config'] == cfg and state['rounds'] == rounds and state['mode'] == mode
    else:
        audit_ep = EpisodeGenerator(6202614, cfg, device).sample(4)
        write_json(work/'results/gradient_diagnostics.json', gradient_audit(model, audit_ep, rounds))
        validation = validate(model, val, lv, cfg, rounds)
        best = validation['selection_score']
        history = [dict(step=0, stage='initial', train_coupled_mse='', train_local_mse='', seconds=0., **validation)]
        save(work/'checkpoints/selected.pt', model, cfg, rounds, 0, best)
    batchloss = []
    began = time.perf_counter()
    previous_seconds = history[-1]['seconds']
    stage(model, start+1, cfg)
    for step in range(start+1, stop+1):
        if step == cfg['warmup_steps']+1:
            stage(model, step, cfg)
        schedule(opt, step, cfg)
        model.train()
        e = generators[0].sample(cfg['batch_size'])
        local = generators[1].sample(cfg['batch_size'], coupled=False)
        opt.zero_grad(set_to_none=True)
        pred = model(e['context1'], e['context2'], e['x0'], e['y0'], rounds=rounds)
        coupled = sum(coupled_losses(pred, e))
        auxiliary = sum(local_losses(model, local))
        loss = coupled+cfg['mu']*auxiliary
        if not torch.isfinite(loss):
            raise RuntimeError(f'{label} nonfinite loss at {step}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        if not torch.isfinite(norm):
            raise RuntimeError('nonfinite gradients')
        opt.step()
        batchloss.append((float(coupled.detach()), float(auxiliary.detach())))
        if step % cfg['validation_interval'] == 0 or step == stop:
            validation = validate(model, val, lv, cfg, rounds)
            row = dict(step=step, stage='warmup' if step <= cfg['warmup_steps'] else 'joint',
                train_coupled_mse=float(np.mean([v[0] for v in batchloss])),
                train_local_mse=float(np.mean([v[1] for v in batchloss])),
                seconds=previous_seconds+time.perf_counter()-began,
                **validation)
            row['learning_rates'] = [g['lr'] for g in opt.param_groups]
            row['remote_bias'] = [a.communication.remote_bias.detach().cpu().tolist() for a in model.agents]
            row['gain'] = [float(a.communication.gain_logit.detach().sigmoid()) for a in model.agents]
            history.append(row)
            batchloss.clear()
            if validation['selection_score'] < best:
                best = validation['selection_score']
                save(work/'checkpoints/selected.pt', model, cfg, rounds, step, best)
            save(work/'checkpoints/resume.pt', model, cfg, rounds, step, best, opt, generators, history)
            write_json(work/'results/history.json', history)
            print(f'{label} step={step}/{cfg["joint_steps"]} val_rel={row["val_relative_l2"]:.6f} '
                  f'coupled={row["val_coupled_mse"]:.4g} local={row["val_local_mse"]:.4g} '
                  f'gain={row["gain"]} seconds={row["seconds"]:.1f}', flush=True)
    assert hashes() == source_hashes, 'SOURCE MODIFIED DURING TRAINING'
    selected = torch.load(work/'checkpoints/selected.pt', map_location='cpu', weights_only=False)
    write_json(work/'results/status.json', dict(completed_steps=stop, planned_steps=cfg['joint_steps'],
        complete=stop == cfg['joint_steps'], selected_step=selected['step'], source_unchanged=True,
        training_seconds=history[-1]['seconds'], initialization='random_from_scratch',
        all_parameters_optimized_from_step_one=True, optimizer_step_counts=sorted(set(int(v['step']) for v in opt.state.values())),
        total_parameters=sum(p.numel() for p in model.parameters()),
        base_parameters_per_agent=sum(p.numel() for p in model.agents[0].base.parameters()),
        communication_parameters_per_agent=sum(p.numel() for p in model.agents[0].communication.parameters())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/train'))
    p.add_argument('--rounds', type=int, default=3, choices=(1, 3, 5))
    p.add_argument('--until', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    args = p.parse_args()
    cfg = read_config(args.config)
    device = setup(cfg['seed'], args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'results/environment.json', dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        hostname=platform.node(), slurm_job=os.environ.get('SLURM_JOB_ID'), threads=torch.get_num_threads()))
    train(args.output, cfg, device, args.rounds, 'concat', args.until, args.resume)


if __name__ == '__main__':
    main()
