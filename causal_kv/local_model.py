"""Lightweight ICON-inspired encoder/decoder; not the preserved ICON_LM architecture."""
import torch
from torch import nn
from .data import Context


class LocalBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h, layers = cfg['hidden_dim'], cfg['num_attention_heads'], cfg['num_transformer_layers']
        self.enforce_initial = cfg.get('enforce_initial', True)
        self.register_buffer('time', torch.linspace(0, cfg['T'], cfg['time_points']))
        self.context_projection = nn.Linear(4, d)
        self.query_projection = nn.Linear(3, d)
        self.context_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, h, 2*d, cfg['dropout'], batch_first=True, norm_first=True), layers,
            enable_nested_tensor=False)
        self.query_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d, h, 2*d, cfg['dropout'], batch_first=True, norm_first=True), layers)
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def encode_context(self, context: Context):
        """Private context memory; reusable across rounds without changing communication."""
        ci, co = context.inputs, context.outputs
        b, m, n, _ = ci.shape
        if ci.shape[-1] != 2 or co.shape != (b, m, n, 1) or n != len(self.time) or m < 1:
            raise ValueError('invalid context shape')
        t = self.time.view(1, 1, n, 1).expand(b, m, n, 1)
        tokens = self.context_projection(torch.cat((ci, co, t), -1)).reshape(b*m, n, -1)
        # Within-demo temporal attention preserves input/output pairing. Cross attention
        # sees all demo tokens, without a fixed demo index embedding (permutation invariant).
        return self.context_encoder(tokens).reshape(b, m*n, -1)

    def forward(self, context, initial, forcing):
        return self.predict_from_memory(self.encode_context(context), initial, forcing)


def local_prediction(model, episode, side):
    query = episode[f'query_{side}_input']
    return model(episode[f'context{side}'], query[:, 0, 0], query[..., 1:])
