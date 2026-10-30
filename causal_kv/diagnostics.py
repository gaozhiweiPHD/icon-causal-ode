"""Numerical causal audits and matrix-free local singular-gain estimates."""
import torch
from .models import Message, causal_mask


def magnitude(values):
    return torch.sqrt(sum(v.square().sum() for v in values))


def normalized(values):
    return tuple(v/magnitude(values).clamp_min(1e-20) for v in values)


def singular_gain(model, hidden, iterations=16, seed=1729):
    fixed = tuple(h.detach().requires_grad_(True) for h in hidden)
    generator = torch.Generator(device=fixed[0].device).manual_seed(seed)
    vector = normalized(tuple(torch.randn(h.shape, generator=generator, device=h.device) for h in fixed))
    def phi(*h):
        return model.advance(*h)[:2]
    history = []
    for _ in range(iterations):
        _, jv = torch.autograd.functional.jvp(phi, fixed, vector)
        history.append(float(magnitude(jv)))
        jtu = torch.autograd.grad(phi(*fixed), fixed, grad_outputs=normalized(jv))
        vector = normalized(tuple(v.detach() for v in jtu))
    _, jv = torch.autograd.functional.jvp(phi, fixed, vector)
    return float(magnitude(jv)), history


@torch.no_grad()
def causality_audit(model, episode, max_rounds=10, tolerance=2e-5):
    model.eval()
    n = len(model.agents[0].base.time)
    cutoff = n//2-1
    past = slice(0, cutoff+1)
    results = {'cutoff_index': cutoff, 'future_start_index': cutoff+1,
               'tolerance': tolerance, 'decoder_layers': [], 'communication_rounds': [],
               'remote_packet_perturbations': [], 'observed_demo_effect': []}
    h_initial = []
    for side, agent in enumerate(model.agents, 1):
        base = agent.base
        context = episode[f'context{side}']
        memory = base.encode_context(context)
        query = episode[f'query_{side}_input']
        initial, forcing = query[:, 0, 0], query[..., 1:]
        changed = forcing.clone()
        changed[:, cutoff+1:] += 7.13
        a = base.query_hidden(memory, initial, forcing, all_layers=True)
        b = base.query_hidden(memory, initial, changed, all_layers=True)
        for layer, (x, y) in enumerate(zip(a, b)):
            results['decoder_layers'].append(dict(agent=side, layer=layer,
                past_max_abs=float((x[:, past]-y[:, past]).abs().max())))
        out_a = base.predict_from_memory(memory, initial, forcing)
        out_b = base.predict_from_memory(memory, initial, changed)
        results['decoder_layers'].append(dict(agent=side, layer='physical_output',
            past_max_abs=float((out_a[:, past]-out_b[:, past]).abs().max())))
        # Fully observed demonstrations are allowed to contain their full response.
        from .base import Context
        altered = Context(context.inputs.clone(), context.outputs.clone())
        altered.outputs[:, :, cutoff+1:] += 2.3
        new_memory = base.encode_context(altered)
        out_c = base.predict_from_memory(new_memory, initial, forcing)
        results['observed_demo_effect'].append(dict(agent=side,
            early_query_output_change=float((out_a[:, 1:cutoff+1]-out_c[:, 1:cutoff+1]).abs().max()),
            early_demo_memory_change=float((memory[:, :cutoff+1]-new_memory[:, :cutoff+1]).abs().max())))
        h_initial.append(agent.initialize(context, episode['x0' if side == 1 else 'y0']))
    # Perturb future states of each sender separately; audit BOTH agents at EVERY depth.
    for sender in (0, 1):
        original = tuple(h.clone() for h in h_initial)
        modified = [h.clone() for h in h_initial]
        modified[sender][:, cutoff+1:] += torch.linspace(-9., 8., modified[sender].shape[-1], device=modified[sender].device)
        for r in range(max_rounds+1):
            results['communication_rounds'].append(dict(perturbed_agent=sender+1, round=r,
                past_max_abs=max(float((x[:, past]-y[:, past]).abs().max()) for x,y in zip(original, modified))))
            if r == max_rounds:
                break
            # Explicit packet-only perturbation, including future K and future V.
            for receiver in (0, 1):
                block = model.agents[receiver].communication
                q, local = block.project(original[receiver])
                _, remote = model.agents[1-receiver].communication.project(original[1-receiver])
                k, v = remote.key.clone(), remote.value.clone()
                k[:, :, cutoff+1:] += 11.7
                v[:, :, cutoff+1:] -= 6.4
                ref = block.update(original[receiver], q, local, remote)[0]
                pert = block.update(original[receiver], q, local, Message(k, v))[0]
                results['remote_packet_perturbations'].append(dict(receiver=receiver+1, round=r+1,
                    past_max_abs=float((ref[:, past]-pert[:, past]).abs().max())))
            original = model.advance(*original)[:2]
            modified = model.advance(*modified)[:2]
    errors = [row['past_max_abs'] for key in ('decoder_layers', 'communication_rounds', 'remote_packet_perturbations')
              for row in results[key]]
    results['maximum_past_change'] = max(errors)
    results['passed'] = max(errors) <= tolerance
    assert results['passed'], results
    return results
