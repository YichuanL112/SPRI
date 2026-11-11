import copy
import json
from pathlib import Path
import pytest
import torch
from spri.config import load_config
from spri.pipeline import Pipeline
from spri.artifacts import read_json, file_digest, load_checkpoint, save_checkpoint, seed_all
from spri.indexing.trainer import train_index
from spri.indexing.tokenizer import SPRIIndexTokenizer
from spri.indexing.materialize import make_index
from spri.recommendation.model import SPRIRecommender
from spri.recommendation.trainer import train_generator
from spri.data.sequence_dataset import SequenceDataset
from spri.data import yelp
from spri.evaluation.metrics import first_rank


@pytest.fixture
def fixture_config(tmp_path, config, fake_text_encoder):
    metadata = [{"asin": f"i{i}", "title": f"item {i}", "categories": [["category"]],
                 "description": f"description {i % 2}"} for i in range(6)]
    reviews = [{"reviewerID": f"u{u}", "asin": f"i{(u + t) % 6}", "unixReviewTime": t}
               for u in range(4) for t in range(7)]
    for name, rows in (("meta", metadata), ("reviews", reviews)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    config["data"].update(reviews=str(tmp_path / "reviews"), metadata=str(tmp_path / "meta"))
    config["runtime"]["output"] = str(tmp_path / "runs")
    return config



def test_full_pipeline_cache_and_re_evaluation(fixture_config):
    pipeline = Pipeline(fixture_config)
    pipeline.run("all")
    root = pipeline.root
    result = read_json(root / "test_result.json")
    assert result["experiment_kind"] == "experiment"
    assert len(read_json(root / "test_predictions.json")) == 4
    assert "sem2beh" in (root / "tokenizer/history.jsonl").read_text()
    original_index = file_digest(root / "index.json")
    original_checkpoint = file_digest(root / "generator/best.pt")
    pipeline.run("all")  # Verified cache; no training or export repeated.
    assert file_digest(root / "index.json") == original_index
    assert file_digest(root / "generator/best.pt") == original_checkpoint

    changed_eval = copy.deepcopy(fixture_config)
    changed_eval["evaluation"]["beam"] = 6
    Pipeline(changed_eval).run("evaluate")
    assert len(list((root / "evaluations").iterdir())) == 2
    assert read_json(root / "test_result.json")["beam"] == 6

    # Reject input-protocol changes before either cache reuse or fresh evaluation.
    for key, value in (("history_length", 2), ("train_sampling", "last_prefix")):
        for beam in (6, 7):
            changed_data = copy.deepcopy(changed_eval)
            changed_data["data"][key] = value
            changed_data["evaluation"]["beam"] = beam
            with pytest.raises(ValueError, match=f"data.{key}"):
                Pipeline(changed_data).run("evaluate")
    assert read_json(root / "test_result.json")["beam"] == 6
    assert len(list((root / "evaluations").iterdir())) == 2

    changed_history = copy.deepcopy(fixture_config)
    changed_history["data"]["history_length"] = 2
    with pytest.raises(ValueError, match="data.history_length"):
        Pipeline(changed_history).input_fields()



def test_yelp_adapter_reads_zip_and_keeps_timestamps(tmp_path):
    import zipfile
    business = {"business_id": "b", "name": "shop", "categories": "food",
                "attributes": {"OutdoorSeating": "True"}, "stars": 4.9, "review_count": 999}
    review = {"business_id": "b", "user_id": "u", "date": "2020-01-01 14:00:00"}
    archive = tmp_path / "yelp.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("business.json", json.dumps(business))
        zf.writestr("review.json", json.dumps(review))
    meta = yelp.metadata(str(archive) + "::business.json")
    assert "OutdoorSeating" in meta["b"]["fine"]
    assert "999" not in meta["b"]["fine"]
    row = next(yelp.interactions(str(archive) + "::review.json"))
    assert row["timestamp"] == 1577887200


def test_tokenizer_resume_matches_uninterrupted_training(fixture_config, tmp_path, monkeypatch):
    p = Pipeline(fixture_config)
    for stage in ("prepare", "features", "train-din"):
        p.run(stage)
    fields, split = p.input_fields(), p.split()
    c = fixture_config["index"]
    seed_all(77)
    expected = train_index(fields, c, torch.device("cpu"), tmp_path / "whole", {"test": 1})
    original_save = save_checkpoint
    def interrupt(path, value):
        original_save(path, value)
        if Path(path).name == "last.pt" and value["stage"] == 2 and value["epoch"] == 0:
            raise RuntimeError("simulated process interruption")
    monkeypatch.setattr("spri.indexing.trainer.save_checkpoint", interrupt)
    seed_all(77)
    with pytest.raises(RuntimeError, match="simulated"):
        train_index(fields, c, torch.device("cpu"), tmp_path / "resumed", {"test": 1})
    monkeypatch.setattr("spri.indexing.trainer.save_checkpoint", original_save)
    actual = train_index(fields, c, torch.device("cpu"), tmp_path / "resumed", {"test": 1}, resume=True)
    for name, tensor in expected.state_dict().items():
        assert torch.equal(tensor.cpu(), actual.state_dict()[name].cpu()), name


def test_beam_scores_match_manual_level_masked_probabilities(fixture_config):
    index = make_index(["a", "b"], [[0, 0, 0, 0], [1, 1, 1, 1]], [2, 2, 2, 2], {"split": "test"})
    model = SPRIRecommender(index, fixture_config["model"]).eval()
    batch = {"history": torch.tensor(index.tokens[:1]).unsqueeze(1), "item_mask": torch.ones(1, 1, dtype=torch.bool)}
    # Enumerating all 16 tuples tests branching, ranking and finite invalid catalog combinations.
    predictions, scores = model.generate(batch, 16)
    assert predictions.shape == (1, 16, 4)
    assert len({tuple(x) for x in predictions[0].tolist()}) == 16
    from transformers.modeling_outputs import BaseModelOutput
    enc = model.encode(batch["history"], batch["item_mask"])
    for code, score in zip(predictions[0], scores[0]):
        prefix = torch.tensor([[0]])
        expected = 0.0
        for level in range(4):
            result = model.t5(encoder_outputs=BaseModelOutput(last_hidden_state=enc.last_hidden_state),
                              attention_mask=enc.attention_mask, decoder_input_ids=prefix, use_cache=False)
            offset = index.offsets[level]
            probability = result.logits[0, -1, offset:offset + 2].float().log_softmax(-1)
            expected += float(probability[code[level] - offset].detach())
            prefix = torch.cat([prefix, code[level].reshape(1, 1)], 1)
        assert float(score) == pytest.approx(expected, abs=1e-5)


def test_generator_resume_matches_uninterrupted(fixture_config, tmp_path, monkeypatch):
    p = Pipeline(fixture_config)
    for stage in ("prepare", "features", "train-din", "train-index", "materialize"):
        p.run(stage)
    index = p.item_index()
    training, validation = p.dataset("train", index), p.dataset("valid", index)
    c, ev = fixture_config["train"], fixture_config["evaluation"]
    seed_all(91)
    model = SPRIRecommender(index, fixture_config["model"])
    train_generator(model, training, validation, c, ev, torch.device("cpu"), tmp_path / "full_gen", {"test": 1})
    original_save = save_checkpoint
    def interrupt(path, value):
        original_save(path, value)
        if Path(path).name == "last.pt" and value["epoch"] == 0:
            raise RuntimeError("simulated interruption")
    monkeypatch.setattr("spri.recommendation.trainer.save_checkpoint", interrupt)
    seed_all(91)
    model = SPRIRecommender(index, fixture_config["model"])
    with pytest.raises(RuntimeError, match="simulated"):
        train_generator(model, training, validation, c, ev, torch.device("cpu"), tmp_path / "resume_gen", {"test": 1})
    monkeypatch.setattr("spri.recommendation.trainer.save_checkpoint", original_save)
    model = SPRIRecommender(index, fixture_config["model"])
    train_generator(model, training, validation, c, ev, torch.device("cpu"), tmp_path / "resume_gen", {"test": 1}, resume=True)
    full = load_checkpoint(tmp_path / "full_gen/last.pt")
    resumed = load_checkpoint(tmp_path / "resume_gen/last.pt")
    assert full["step"] == resumed["step"]
    for name in full["model"]:
        assert torch.equal(full["model"][name], resumed["model"][name]), name

