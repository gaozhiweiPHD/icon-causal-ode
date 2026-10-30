from pathlib import Path
import pytest
from causal_kv.data import EpisodeGenerator
from causal_kv.training import read_config, setup


@pytest.fixture
def cfg():
    setup(42, 'cpu')
    return read_config(Path(__file__).resolve().parents[1] / 'configs/default.yaml')


@pytest.fixture
def episode(cfg):
    return EpisodeGenerator(12, cfg).sample(3)
