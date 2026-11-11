import copy
import math
import numpy as np
import pytest
import torch
from spri.config import load_config
from spri.data.splits import prepare, samples, train_sequences
from spri.indexing.quantizer import VectorQuantizer
from spri.indexing.pcr import PrefixConditionedRefinement
from spri.indexing.pgq import PrefixGuidedQuantization
from spri.indexing.tokenizer import SPRIIndexTokenizer
from spri.indexing.materialize import encode_catalog
from spri.indexing.materialize import make_index
from spri.indexing.index import MaterializedIndex, LEFT, RIGHT
from spri.indexing.losses import alignment, SemanticToBehaviorAlignment
from spri.recommendation.shc import StructuredHistoryConstruction
from spri.recommendation.guidance import soft_prefix_guidance
from spri.evaluation.metrics import first_rank, rank_metrics
from spri.artifacts import write_json
from spri.evaluation.aggregate import aggregate

torch.set_num_threads(1)




def test_shared_split_excludes_heldout_events_even_with_repeated_items():
    metadata = {str(i): {"category": "", "title": "", "fine": ""} for i in range(4)}
    events = [{"user": "u", "item": str(i % 4), "timestamp": i, "ordinal": i} for i in range(8)]
    split = prepare(events, metadata, core=1)
    training = list(samples(split, "train", 20))
    assert [s["target_position"] for s in training] == [1, 2, 3, 4, 5]
    assert train_sequences(split) == [[0, 1, 2, 3, 0, 1]]
    assert list(samples(split, "valid", 20))[0]["target_position"] == 6
    test = list(samples(split, "test", 20))[0]
    assert test["target_position"] == 7 and test["history"][-1] == 2


def test_vq_gradient_coefficients_match_manuscript():
    q = VectorQuantizer(2, 1, 0.25)
    with torch.no_grad():
        q.embedding.weight.copy_(torch.tensor([[0.0], [10.0]]))
    x = torch.tensor([[2.0]], requires_grad=True)
    _, _, loss = q(x)
    loss.backward()
    assert x.grad.item() == pytest.approx(4.0)
    assert q.embedding.weight.grad[0].item() == pytest.approx(-1.0)
    assert q.embedding.weight.grad[1].item() == 0


@pytest.mark.parametrize("quantization,reconstruction", [(True, True), (False, True), (True, False)])
def test_pcr_detaches_both_prefix_paths(config, quantization, reconstruction):
    c = copy.deepcopy(config["index"])
    c.update(conditional_quantization=quantization, conditional_reconstruction=reconstruction, conditional_dropout=0.0)
    pcr = PrefixConditionedRefinement(7, c)
    prefix = [torch.randn(3, c["dim"], requires_grad=True) for _ in range(3)]
    x = torch.randn(3, 7, requires_grad=True)
    reconstruction, _, _, loss, _ = pcr(x, prefix)
    (reconstruction.square().mean() + loss).backward()
    assert all(z.grad is None for z in prefix)
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_alignment_gradient_direction(config):
    x = torch.randn(5, 8, requires_grad=True)
    y = torch.randn(5, 8, requires_grad=True)
    alignment(x, y, config["index"]["alignment"]).backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert y.grad is None


def test_sem2beh_target_and_source_gradient_contract():
    objective = SemanticToBehaviorAlignment(8, 11)
    source = torch.randn(4, 8, requires_grad=True)
    target = torch.randn(4, 11, requires_grad=True)
    objective(source, target, detach_source=False).backward()
    assert source.grad is not None and target.grad is None
    assert objective.predictor[0].weight.grad.abs().sum() > 0


def test_pgq_checkpoint_and_eval_are_immutable(config):
    pgq = PrefixGuidedQuantization(4, 8, config["index"]["pgq"])
    parent = torch.tensor([0, 0, 1, 2])
    assert pgq.candidates(parent).shape == (4, 2)
    pgq.observe(parent, torch.tensor([0, 1, 2, 4]))
    table = pgq.candidates(parent)
    assert table.shape == (4, 4)
    assert all(len(set(row)) == 4 for row in pgq.shortlist.tolist())
    clone = PrefixGuidedQuantization(4, 8, config["index"]["pgq"])
    clone.load_state_dict(pgq.state_dict())
    clone.eval()
    before = {k: v.clone() for k, v in clone.state_dict().items()}
    assert torch.equal(table, clone.candidates(parent))
    clone.observe(parent, torch.tensor([7, 7, 7, 7]))
    assert all(torch.equal(before[k], v) for k, v in clone.state_dict().items())


def test_materialization_is_stable_and_unique(tmp_path):
    ids = ["b", "a", "c"]
    raw = [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 4]]
    index = make_index(ids, raw, [4, 4, 8, 8], {"split": "x"})
    assert index.codes.tolist() == [[0, 1, 2, 11], [0, 1, 2, 3], [0, 1, 2, 4]]
    shuffled = make_index(ids[::-1], raw[::-1], [4, 4, 8, 8], {"split": "x"})
    assert dict(zip(ids, map(tuple, index.codes))) == dict(zip(ids[::-1], map(tuple, shuffled.codes)))
    index.save(tmp_path / "index.json")
    loaded = MaterializedIndex.load(tmp_path / "index.json")
    assert loaded.digest == index.digest
    assert len(index.inverse) == 3


def test_shc_layout_mask_locality_and_only_behavior_update(config):
    module = StructuredHistoryConstruction(8, config["model"]["shc"]).eval()
    embedding = torch.nn.Embedding(50, 8)
    tokens = torch.tensor([[[5, 9, 13, 21], [6, 10, 14, 22], [0, 0, 0, 0]]])
    mask = torch.tensor([[True, True, False]])
    blocks = module.contextualize(tokens, embedding)
    assert torch.allclose(blocks[..., :3, :], (embedding(tokens) + module.roles.weight)[..., :3, :])
    assert not torch.allclose(blocks[..., 3, :], (embedding(tokens) + module.roles.weight)[..., 3, :])
    altered = tokens.clone()
    altered[0, 1] += 1
    assert torch.allclose(module.contextualize(altered, embedding)[0, 0], blocks[0, 0])
    output, attention = module(tokens, mask, embedding)
    assert output.shape == (1, 17, 8)
    assert attention.sum() == 12
    assert torch.allclose(output[0, 0], embedding.weight[LEFT])
    assert torch.allclose(output[0, 11], embedding.weight[RIGHT])
    loss = output[attention].sum()
    loss.backward()
    assert embedding.weight.grad[0].abs().sum() == 0


def test_guidance_is_finite_and_only_at_behavior_position():
    logits = torch.zeros(1, 5, 30)
    labels = torch.tensor([[5, 9, 13, 21, 1]])
    config = {"enabled": True, "bias": 0.8, "start": 50, "warmup": 100}
    support = {(5, 9, 13): {21, 22}}
    assert torch.equal(soft_prefix_guidance(logits, labels, support, 50, config), logits)
    result = soft_prefix_guidance(logits, labels, support, 100, config)
    assert result[0, 3, 21] == 0
    assert result[0, 3, 23].item() == pytest.approx(-0.4)
    assert torch.equal(result[:, :3], logits[:, :3]) and torch.equal(result[:, 4], logits[:, 4])
    assert torch.isfinite(result).all()


def test_invalid_tuples_keep_rank_and_duplicate_target_counts_once():
    target = [1, 2, 3, 4]
    predictions = [[9, 9, 9, 9], target, target]
    rank = first_rank(predictions, target)
    assert rank == 2
    metrics = rank_metrics(rank, [1, 2, 5])
    assert metrics["recall@1"] == 0 and metrics["recall@5"] == 1
    assert metrics["ndcg@5"] == pytest.approx(1 / math.log2(3))


def test_strict_configuration(config):
    with pytest.raises(ValueError, match="Unknown"):
        load_config(overrides=["index.conditonal_quantization=false"])
    with pytest.raises(ValueError, match="Beam"):
        load_config(overrides=["evaluation.beam=2"])
    for value in (-0.1, 1.0):
        with pytest.raises(ValueError, match="Dropout"):
            load_config(overrides=[f"index.semantic_conditional_dropout={value}"])


def test_semantic_conditional_dropout_masks_each_item_before_conditioner(config, monkeypatch):
    c = copy.deepcopy(config["index"])
    c.update(semantic_conditional_dropout=0.5, encoder_dropout=0.0)
    model = SPRIIndexTokenizer([7, 7, 7, 7], c).train()
    fields = [torch.randn(4, 7) for _ in range(4)]
    contexts, prefixes = [], []
    handles = [layer.register_forward_pre_hook(lambda module, args: contexts.append(args[0].detach().clone()))
               for layer in model.conditioners[1:]]
    handles += [layer.register_forward_hook(lambda module, args, out: prefixes.append(out[0].detach().clone()))
                for layer in model.quantizers[:2]]
    draws = [torch.tensor([[0.1], [0.9], [0.1], [0.9]]),
             torch.tensor([[0.9], [0.1], [0.9], [0.1]])]
    calls = []
    def draw(shape, device=None):
        assert shape == (4, 1)
        calls.append(shape)
        return draws[len(calls) - 1].to(device)
    monkeypatch.setattr(torch, "rand", draw)
    try:
        model(fields, stage=1)
        assert len(calls) == len(contexts) == 2
        for context, prefix, random_values in zip(contexts, prefixes, draws):
            assert torch.equal(context, prefix * (random_values >= 0.5))
        contexts.clear()
        prefixes.clear()
        calls.clear()
        model.config["semantic_conditional_dropout"] = 0.0
        model(fields, stage=1)
        assert not calls
        assert all(torch.equal(x, y) for x, y in zip(contexts, prefixes))
    finally:
        for handle in handles:
            handle.remove()


def test_catalog_export_disables_conditional_dropout_and_preserves_pgq(config, monkeypatch):
    c = copy.deepcopy(config["index"])
    c.update(semantic_conditional_dropout=0.9, conditional_dropout=0.9)
    model = SPRIIndexTokenizer([7, 7, 7, 7], c).train()
    fields = [torch.randn(6, 7) for _ in range(4)]
    model(fields)  # Establish training assignments and a refreshable PGQ clock.
    before = {key: value.clone() for key, value in model.pgq.state_dict().items()}
    def unexpected_draw(*args, **kwargs):
        raise AssertionError("Conditional masks must not be sampled during export")
    monkeypatch.setattr(torch, "rand", unexpected_draw)
    first = encode_catalog(model, fields, 2)
    second = encode_catalog(model, fields, 6)
    assert np.array_equal(first, second)
    assert all(torch.equal(value, model.pgq.state_dict()[key]) for key, value in before.items())


def test_aggregate_rejects_incompatible_runs(tmp_path):
    a = {"comparison_key": "a", "dataset": "beauty", "partition": "test", "experiment_kind": "smoke",
         "seed": 1, "metrics": {"ndcg@5": 0.2}}
    b = {**a, "seed": 2, "metrics": {"ndcg@5": 0.4}}
    write_json(tmp_path / "a.json", a)
    write_json(tmp_path / "b.json", b)
    result = aggregate([tmp_path / "a.json", tmp_path / "b.json"])
    assert result["metrics"]["ndcg@5"]["mean"] == pytest.approx(0.3)
    b["comparison_key"] = "different-beam"
    write_json(tmp_path / "b.json", b)
    with pytest.raises(ValueError, match="comparison_key"):
        aggregate([tmp_path / "a.json", tmp_path / "b.json"])



def test_yaml_defaults_match_builtin():
    from spri.config import DEFAULTS
    import yaml
    from pathlib import Path
    assert yaml.safe_load(Path("configs/default.yaml").read_text()) == DEFAULTS


def test_main_rejects_removed_development_settings():
    for override in ["features.backend=hash_smoke", "index.experimental.sequence_weight=0.1", "runtime.index_from=old"]:
        with pytest.raises(ValueError, match="Unknown"):
            load_config(overrides=[override])
