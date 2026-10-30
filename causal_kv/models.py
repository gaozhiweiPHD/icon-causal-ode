"""Causal query ICONs plus causal round-tied dynamic KV blocks. No pretrained weights."""
import copy
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .local_model import LocalBackbone


@dataclass(frozen=True)
class Message:
    key: torch.Tensor       # [B, heads=4, time=64, dim=16]; NEVER detached
    value: torch.Tensor

    def shuffled(self):
        if len(self.key) < 2:
            raise ValueError('shuffle control requires at least two episodes')
        return Message(self.key.roll(1, 0), self.value.roll(1, 0))

    def bytes_per_episode(self):
        return sum(v.numel()*v.element_size() for v in (self.key, self.value))//len(self.key)


def concat_kv(local, remote):
    for a, b in ((local.key, remote.key), (local.value, remote.value)):
        if (a.ndim != 4 or a.shape[:2] != b.shape[:2] or a.shape[-1] != b.shape[-1]
                or a.dtype != b.dtype or a.device != b.device):
            raise ValueError('incompatible batch/head/dimension/dtype/device')
    if local.key.shape != local.value.shape or remote.key.shape != remote.value.shape:
        raise ValueError('K/V shape mismatch')
    return torch.cat((local.key, remote.key), 2), torch.cat((local.value, remote.value), 2)


def causal_mask(n, reference):
    return torch.full((n, n), -torch.inf, dtype=reference.dtype, device=reference.device).triu(1)


class CausalLocalICON(LocalBackbone):
    def query_hidden(self, memory, initial, forcing, all_layers=False):
        if forcing.ndim == 2:
            forcing = forcing[..., None]
        b, n, c = forcing.shape
        if n != len(self.time) or c != 1:
            raise ValueError('forcing must match query time grid')
        initial = initial.reshape(b, 1, 1)
        t = self.time.view(1, n, 1).expand(b, n, 1)
        hidden = self.query_projection(torch.cat((initial.expand_as(t), forcing, t), -1))
        mask = causal_mask(n, hidden)
        if all_layers:
            states = [hidden]
            for layer in self.query_decoder.layers:
                hidden = layer(hidden, memory, tgt_mask=mask)
                states.append(hidden)
            return states
        return self.query_decoder(hidden, memory, tgt_mask=mask)

    def predict_from_memory(self, memory, initial, forcing):
        h = self.output(self.query_hidden(memory, initial, forcing))
        return initial.reshape(-1, 1, 1)+self.time[None, :, None]*h if self.enforce_initial else h


class CommunicationBlock(nn.Module):
    def __init__(self, decoder_layer, mode='concat', initial_bias=-3., initial_gain=.1):
        super().__init__()
        if mode != 'concat':
            raise ValueError(mode)
        self.mode = mode
        # Independent copies; initialized from THIS agent's final decoder layer.
        self.attention = copy.deepcopy(decoder_layer.self_attn)
        # These additive biases cancel in the residual differences below.
        self.attention.out_proj.register_parameter('bias', None)
        self.norm = copy.deepcopy(decoder_layer.norm1)
        self.norm_ff = copy.deepcopy(decoder_layer.norm3)
        self.linear1 = copy.deepcopy(decoder_layer.linear1)
        self.linear2 = copy.deepcopy(decoder_layer.linear2)
        self.linear2.register_parameter('bias', None)
        self.remote_bias = nn.Parameter(torch.full((self.attention.num_heads,), float(initial_bias)))
        self.gain_logit = nn.Parameter(torch.tensor(math.log(initial_gain/(1-initial_gain))))
        if self.attention.dropout != 0 or not decoder_layer.norm_first:
            raise ValueError('only pre-norm dropout-free block supported')

    def project(self, hidden):
        b, n, d = hidden.shape
        h = self.attention.num_heads
        projected = F.linear(self.norm(hidden), self.attention.in_proj_weight, self.attention.in_proj_bias)
        def heads(t):
            return t.reshape(b, n, h, d//h).transpose(1, 2)
        q, k, v = map(heads, projected.chunk(3, -1))
        return q, Message(k, v)

    def ffn(self, hidden):
        return self.linear2(F.relu(self.linear1(self.norm_ff(hidden))))

    def update(self, hidden, q, local, remote, control='real', diagnostics=False):
        if control not in ('real', 'masked', 'shuffled', 'zero'):
            raise ValueError(control)
        if control == 'shuffled':
            remote = remote.shuffled()
        elif control == 'zero':
            remote = Message(torch.zeros_like(remote.key), torch.zeros_like(remote.value))
        d, n = q.shape[-1], local.key.shape[2]
        if q.shape[2] != n or remote.key.shape[2] != n:
            raise ValueError('both KV blocks must use the same physical time grid')
        temporal = causal_mask(n, q)
        local_logits = (q @ local.key.transpose(-1, -2))/math.sqrt(d)
        local_attn = (local_logits+temporal).softmax(-1) @ local.value
        key, value = concat_kv(local, remote)
        logits = (q @ key.transpose(-1, -2))/math.sqrt(d)
        # Float mask is a learned per-head remote logit bias. Local slots
        # receive zero; masked control uses -inf for EVERY remote slot.
        remote_mask = torch.full_like(self.remote_bias, -torch.inf) if control == 'masked' else self.remote_bias
        bias = torch.cat((logits.new_zeros((1, len(remote_mask), 1, n)),
            remote_mask[None, :, None, None].expand(1, -1, 1, remote.key.shape[2])), -1)
        weights = (logits+bias+torch.cat((temporal, temporal), -1)).softmax(-1)
        attention_delta = weights @ value-local_attn
        mass = weights[..., n:].sum(-1)
        delta = attention_delta.transpose(1, 2).contiguous().reshape_as(hidden)
        # Output bias cancels exactly in the attention difference.
        delta = F.linear(delta, self.attention.out_proj.weight, None)
        gain = self.gain_logit.sigmoid()
        intermediate = hidden+gain*delta
        updated = intermediate+gain*(self.ffn(intermediate)-self.ffn(hidden))
        diag = None
        if diagnostics:
            diag = {'remote_mass': mass.mean((1, 2)), 'gain': gain.expand(len(hidden)),
                'remote_domination': (mass > .9).float().mean((1, 2)),
                'local_domination': (mass < .1).float().mean((1, 2))}
        return updated, diag


class Agent(nn.Module):
    def __init__(self, base, mode='concat', initial_bias=-3., initial_gain=.1):
        super().__init__()
        self.base = base
        self.communication = CommunicationBlock(base.query_decoder.layers[-1], mode, initial_bias, initial_gain)

    def initialize(self, context, initial):
        # The ONLY coupled query input is own initial, fixed zero forcing, time.
        b, n = len(initial), len(self.base.time)
        memory = self.base.encode_context(context)
        t = self.base.time.view(1, n, 1).expand(b, n, 1)
        return self.base.query_hidden(memory, initial, torch.zeros_like(t))

    def decode(self, hidden, initial):
        output = self.base.output(hidden)
        return initial[:, None, None]+self.base.time[None, :, None]*output if self.base.enforce_initial else output


class CoupledKV(nn.Module):
    def __init__(self, bases, mode='concat', initial_bias=-3., initial_gain=.1):
        super().__init__()
        if not torch.equal(bases[0].time, bases[1].time):
            raise ValueError('physical time grids must match; no RoPE/relative positions supported')
        self.agents = nn.ModuleList([Agent(b, mode, initial_bias, initial_gain) for b in bases])
        self.mode = mode
        assert set(map(id, self.agents[0].parameters())).isdisjoint(map(id, self.agents[1].parameters()))

    def advance(self, h1, h2, control='real', diagnostics=False):
        c1, c2 = (a.communication for a in self.agents)
        # Both messages come from OLD states. No outputs or targets are transmitted.
        q1, m1 = c1.project(h1)
        q2, m2 = c2.project(h2)
        n1, d1 = c1.update(h1, q1, m1, m2, control, diagnostics)
        n2, d2 = c2.update(h2, q2, m2, m1, control, diagnostics)
        return n1, n2, (m1, m2), (d1, d2)

    def forward(self, context1, context2, x0, y0, rounds=5, control='real', diagnostics=False, return_hidden=False):
        if rounds < 0:
            raise ValueError('negative rounds')
        h1 = self.agents[0].initialize(context1, x0)
        h2 = self.agents[1].initialize(context2, y0)
        hidden, states, info, payload = [(h1, h2)], [], [], []
        states.append((self.agents[0].decode(h1, x0), self.agents[1].decode(h2, y0)))
        for _ in range(rounds):
            h1, h2, messages, diag = self.advance(h1, h2, control, diagnostics)
            states.append((self.agents[0].decode(h1, x0), self.agents[1].decode(h2, y0)))
            hidden.append((h1, h2))
            payload.append(sum(m.bytes_per_episode() for m in messages))
            info.append(diag)
        if return_hidden:
            return states, hidden, info, payload
        return states[-1]


def random_pair(cfg, device='cpu', mode='concat'):
    # Two independent default-PyTorch initializations. Never read old checkpoints.
    bases = [CausalLocalICON(cfg) for _ in range(2)]
    return CoupledKV(bases, mode, cfg['initial_remote_bias'], cfg['initial_gain']).to(device)


def from_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = random_pair(saved['config'], device, saved['mode'])
    model.load_state_dict(saved['model'])
    return model.eval(), saved
