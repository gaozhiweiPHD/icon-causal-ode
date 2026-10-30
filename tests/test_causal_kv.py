import copy
import math
import pytest
import torch
from causal_kv.base import Context, EpisodeGenerator
from causal_kv.models import CausalLocalICON, causal_mask, CoupledKV, Message, concat_kv, random_pair
from causal_kv.training import (gradient_audit, hashes, optimizer, read_config, restore,
                               save, setup, local_losses)


@pytest.fixture
def example():
    setup(42, 'cpu')
    cfg = dict(T=1., time_points=8, num_demonstrations=4, hidden_dim=16,
        num_attention_heads=4, num_transformer_layers=2, dropout=0.,
        initial_gain=.1, initial_remote_bias=-3., base_learning_rate=1e-4,
        communication_learning_rate=1e-3, weight_decay=1e-4, gradient_clip=1.)
    bases = [CausalLocalICON(cfg) for _ in range(2)]
    model = CoupledKV(bases)
    e = EpisodeGenerator(345, cfg).sample(3)
    return model, e, cfg


def run(model, e, **kwargs):
    return model(e['context1'], e['context2'], e['x0'], e['y0'], **kwargs)


def test_shapes_and_concat(example):
    model, e, _ = example
    states, hidden, _, bytes_ = run(model, e, rounds=3, return_hidden=True)
    assert len(states) == len(hidden) == 4
    assert hidden[0][0].shape == (3, 8, 16)
    assert states[-1][0].shape == (3, 8, 1)
    assert bytes_ == [2048]*3
    for a in model.agents:
        _, packet = a.communication.project(hidden[0][0])
        assert packet.key.shape == (3, 4, 8, 4)
        k, v = concat_kv(packet, packet)
        assert k.shape == v.shape == (3, 4, 16, 4)
        with pytest.raises(ValueError):
            concat_kv(packet, Message(packet.key[:, :2], packet.value[:, :2]))


def test_r0_and_masked_identity_initial_conditions(example):
    model, e, _ = example
    model.eval()
    expected = []
    zero = torch.zeros(3, 8, 1)
    for i, a in enumerate(model.agents, 1):
        expected.append(a.base(e[f'context{i}'], e['x0' if i == 1 else 'y0'], zero))
    for mode in ('concat',):
        model.mode = mode
        for a in model.agents:
            a.communication.mode = mode
        for rounds, control in ((0, 'real'), (5, 'masked')):
            got = run(model, e, rounds=rounds, control=control)
            for j, pred in enumerate(got):
                torch.testing.assert_close(pred, expected[j], atol=2e-6, rtol=2e-5)
                assert torch.equal(pred[:, 0, 0], e['x0' if j == 0 else 'y0'])


def test_synchronous_update_and_shared_modules(example):
    model, e, _ = example
    h1 = model.agents[0].initialize(e['context1'], e['x0'])
    h2 = model.agents[1].initialize(e['context2'], e['y0'])
    c1, c2 = (a.communication for a in model.agents)
    q1, m1 = c1.project(h1)
    q2, m2 = c2.project(h2)
    expected1 = c1.update(h1, q1, m1, m2)[0]
    expected2 = c2.update(h2, q2, m2, m1)[0]
    n1, n2, _, _ = model.advance(h1, h2)
    assert torch.equal(expected1, n1) and torch.equal(expected2, n2)
    pointers = [p.data_ptr() for p in model.parameters()]
    run(model, e, rounds=5)
    assert pointers == [p.data_ptr() for p in model.parameters()]
    assert set(map(id, c1.parameters())).isdisjoint(map(id, c2.parameters()))
    assert set(map(id, c1.parameters())).isdisjoint(map(id, model.agents[0].base.parameters()))


def test_bidirectional_cross_gradients_and_no_detach(example):
    model, e, _ = example
    diagnostics = gradient_audit(model, e, 3)
    assert diagnostics['loss_x_to_agent2_parameter_gradient_norm'] > 0
    assert diagnostics['loss_y_to_agent1_parameter_gradient_norm'] > 0
    out = run(model, e, rounds=1, control='masked')
    remote_grad = torch.autograd.grad(out[0].sum(), tuple(model.agents[1].parameters()), allow_unused=True)
    assert all(g is None or torch.count_nonzero(g) == 0 for g in remote_grad)


def test_no_leakage_and_local_task_separation(example):
    model, e, _ = example
    first = run(model, e, rounds=3)
    e['x_true'].fill_(1e6)
    e['y_true'].fill_(-1e6)
    e['query_1_target'].fill_(1e7)
    e['query_2_input'].fill_(7.)
    e['parameters'][:] = 99
    second = run(model, e, rounds=3)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    # Coupled API does not accept a physical forcing argument.
    with pytest.raises(TypeError):
        run(model, e, forcing=e['x_true'])


def test_mask_attention_and_shuffle_episode_axis(example):
    model, e, _ = example
    states, _, diag, _ = run(model, e, rounds=1, control='masked', diagnostics=True, return_hidden=True)
    assert all(torch.count_nonzero(d['remote_mass']) == 0 for d in diag[0])
    states, _, diag, _ = run(model, e, rounds=1, diagnostics=True, return_hidden=True)
    assert all(((d['remote_mass'] >= 0) & (d['remote_mass'] <= 1)).all() for d in diag[0])
    key = torch.arange(3.).view(3, 1, 1, 1)
    packet = Message(key, key+1)
    shuffled = packet.shuffled()
    assert shuffled.key[:, 0, 0, 0].tolist() == [2., 0., 1.]
    assert torch.equal(packet.key, key)


def test_deterministic_resume(example, tmp_path):
    model, e, cfg = example
    opt = optimizer(model, cfg)
    generators = [EpisodeGenerator(s, cfg) for s in (12, 34)]
    def step():
        sample = generators[0].sample(3)
        local = generators[1].sample(3, coupled=False)
        opt.zero_grad(set_to_none=True)
        pred = run(model, sample, rounds=3)
        loss = sum((v-sample[k]).square().mean() for v, k in zip(pred, ('x_true', 'y_true')))+.2*sum(local_losses(model, local))
        loss.backward()
        opt.step()
    step()
    path = tmp_path/'resume.pt'
    save(path, model, cfg, 3, 1, 1., opt, generators, [])
    step()
    expected = copy.deepcopy(model.state_dict())
    restore(path, model, opt, generators)
    step()
    assert all(torch.equal(v, expected[k]) for k, v in model.state_dict().items())


def test_real_random_initialization_and_integrity():
    cfg = read_config()
    before = hashes()
    model = random_pair(cfg, 'cpu').eval()
    e = EpisodeGenerator(765, cfg).sample(4)
    with torch.no_grad():
        zero = run(model, e, rounds=0)
        five = run(model, e, rounds=5)
    # Conservative initialization: tiny physical change, not catastrophic injection.
    difference = max(float((a-b).abs().max()) for a, b in zip(zero, five))
    assert difference < .03
    assert before == hashes()


def test_layernorm_shift_equivariance(example):
    model, e, _ = example
    h1 = model.agents[0].initialize(e['context1'], e['x0'])
    h2 = model.agents[1].initialize(e['context2'], e['y0'])
    shifts = [torch.randn(3, 8, 1)*.2 for _ in range(2)]
    original = model.advance(h1, h2)[:2]
    shifted = model.advance(h1+shifts[0], h2+shifts[1])[:2]
    for a, b, c in zip(original, shifted, shifts):
        torch.testing.assert_close(b, a+c, atol=2e-6, rtol=2e-5)


def test_matrix_free_singular_gain(example):
    from causal_kv.diagnostics import singular_gain
    model, e, _ = example
    hidden = tuple(a.initialize(e[f'context{j}'], e['x0' if j == 1 else 'y0'])[:1]
                   for j, a in enumerate(model.agents, 1))
    value, history = singular_gain(model, hidden, iterations=3)
    assert math.isfinite(value) and value > 0 and len(history) == 3


def test_episode_isolation_and_message_immutability(example):
    model, e, _ = example
    original = run(model, e, rounds=1)
    changed = dict(e)
    changed['context2'] = Context(e['context2'].inputs.clone(), e['context2'].outputs.clone())
    changed['context2'].inputs[1] += 2
    changed['context2'].outputs[1] -= 3
    altered = run(model, changed, rounds=1)
    assert torch.equal(original[0][[0, 2]], altered[0][[0, 2]])
    assert not torch.equal(original[0][1], altered[0][1])
    h1 = model.agents[0].initialize(e['context1'], e['x0'])
    h2 = model.agents[1].initialize(e['context2'], e['y0'])
    q1, m1 = model.agents[0].communication.project(h1)
    _, m2 = model.agents[1].communication.project(h2)
    snapshot = (m2.key.clone(), m2.value.clone())
    model.agents[0].communication.update(h1, q1, m1, m2)
    assert torch.equal(m2.key, snapshot[0]) and torch.equal(m2.value, snapshot[1])


def test_causality_all_decoder_layers_and_every_round(example):
    from causal_kv.diagnostics import causality_audit
    model, e, _ = example
    result = causality_audit(model, e, max_rounds=10)
    assert result['passed']
    assert len(result['communication_rounds']) == 22
    assert all(row['early_demo_memory_change'] > 0 for row in result['observed_demo_effect'])


def test_physical_time_mask_two_blocks(example):
    model, e, _ = example
    h = model.agents[0].initialize(e['context1'], e['x0'])
    mask = causal_mask(8, h)
    two = torch.cat((mask, mask), -1)
    assert two.shape == (8, 16)
    for i in range(8):
        for j in range(8):
            assert torch.isfinite(two[i, j]) == (j <= i)
            assert torch.isfinite(two[i, j+8]) == (j <= i)


def test_from_scratch_independent_random_weights():
    cfg = read_config()
    setup(1729, 'cpu')
    a = random_pair(cfg)
    setup(1729, 'cpu')
    b = random_pair(cfg)
    assert all(torch.equal(v, b.state_dict()[k]) for k,v in a.state_dict().items())
    assert not torch.equal(a.agents[0].base.query_projection.weight, a.agents[1].base.query_projection.weight)
    assert all(p.requires_grad for p in a.parameters())
