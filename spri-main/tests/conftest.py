from copy import deepcopy
import numpy as np
import pytest
from spri.config import load_config, merge, validate

SMALL = {'runtime': {'device': 'cpu', 'threads': 1, 'run_id': 'test'}, 'data': {'core': 2, 'history_length': 4}, 'din': {'dim': 8, 'epochs': 1, 'batch_size': 8, 'negatives': 3}, 'index': {'dim': 8, 'sizes': [4, 4, 8, 8], 'stage1_epochs': 1, 'stage2_epochs': 2, 'unfreeze_epoch': 1, 'batch_size': 4, 'kmeans_iters': 1, 'encoder_dropout': 0.0, 'pgq': {'topk': 4, 'warmup_steps': 1, 'refresh_steps': 1}, 'alignment': {'start': 0, 'warmup': 0}, 'sem2beh': {'start': 0, 'warmup': 0}}, 'model': {'dim': 16, 'ff_dim': 32, 'heads': 2, 'head_dim': 8, 'encoder_layers': 1, 'decoder_layers': 1, 'dropout': 0.0, 'shc': {'dropout': 0.0}, 'guidance': {'start': 0, 'warmup': 0}}, 'train': {'epochs': 2, 'batch_size': 4, 'eval_batch_size': 3, 'accumulation': 2, 'warmup_steps': 0, 'patience': 2}, 'evaluation': {'beam': 5, 'topk': [1, 5], 'selection_metric': 'ndcg@5'}, 'features': {'model': 'test-only-encoder'}}

@pytest.fixture
def config():
    return validate(merge(load_config(), deepcopy(SMALL)))

@pytest.fixture
def fake_text_encoder(monkeypatch):
    import sentence_transformers
    class Encoder:
        def __init__(self, model, revision=None, device=None):
            assert model == "test-only-encoder"
        def encode(self, texts, **kwargs):
            return np.random.default_rng(17).normal(size=(len(texts), 12)).astype(np.float32)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Encoder)
