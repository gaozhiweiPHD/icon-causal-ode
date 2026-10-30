"""Packaging-sensitive checks: checkpoint load, resume and isolated evaluation."""
from pathlib import Path

import pytest
import torch

from causal_kv.base import EpisodeGenerator
from causal_kv.evaluation import evaluate
from causal_kv.models import from_checkpoint
from causal_kv.training import read_config, setup, train, validate

ROOT = Path(__file__).resolve().parents[1]


def test_pretrained_checkpoint():
    setup(42, 'cpu')
    model, saved = from_checkpoint(ROOT / 'pretrained/causal_kv_r3.pt')
    assert saved['step'] == 50000 and saved['rounds'] == 3
    assert sum(p.numel() for p in model.parameters()) == 403084
    e = EpisodeGenerator(20261030, saved['config']).sample(2)
    with torch.no_grad():
        outputs = model(e['context1'], e['context2'], e['x0'], e['y0'], rounds=3)
    for output, initial in zip(outputs, (e['x0'], e['y0'])):
        assert torch.isfinite(output).all()
        assert torch.equal(output[:, 0, 0], initial)


def test_train_resume_and_evaluate(tmp_path):
    cfg = read_config(ROOT / 'configs/smoke.yaml')
    setup(cfg['seed'], 'cpu')
    train(tmp_path, cfg, torch.device('cpu'), rounds=3, until=2)
    with pytest.raises(FileExistsError):
        train(tmp_path, cfg, torch.device('cpu'), rounds=3)
    train(tmp_path, cfg, torch.device('cpu'), rounds=3, resume=True)
    checkpoint = tmp_path / 'runs/concat_r3/checkpoints/resume.pt'
    model, saved = from_checkpoint(checkpoint)
    assert saved['step'] == 4
    rows, summary = evaluate(model, cfg, 5, [0, 3], ['real', 'masked'], 'cpu', batch_size=3)
    assert len(rows) == 2 * 5 * 2 * 2
    assert len(summary) == 8
    assert all(r['n'] == 5 for r in summary)
    # Masked communication leaves the decoded local prediction unchanged.
    for split in ('id', 'ood'):
        a = [r['relative_l2'] for r in rows if r['split'] == split and r['control'] == 'masked' and r['round'] == 0]
        b = [r['relative_l2'] for r in rows if r['split'] == split and r['control'] == 'masked' and r['round'] == 3]
        assert a == pytest.approx(b, rel=2e-5, abs=2e-6)
    # Validation must weight a short final batch by its actual episode count.
    val = EpisodeGenerator(70, cfg).sample(5)
    local = EpisodeGenerator(71, cfg).sample(5, coupled=False)
    by_three = validate(model, val, local, cfg, 3)
    together = validate(model, val, local, dict(cfg, batch_size=5), 3)
    assert by_three == pytest.approx(together, rel=2e-5, abs=2e-7)
    with pytest.raises(ValueError, match='shuffled'):
        evaluate(model, cfg, 4, [3], ['shuffled'], 'cpu', batch_size=3)
