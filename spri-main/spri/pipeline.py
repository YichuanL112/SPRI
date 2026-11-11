"""Explicit stages with validated inputs, output hashes and resumable training."""
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import platform
import torch
from torch.utils.data import DataLoader
from .artifacts import (digest, file_digest, read_json, write_json, save_checkpoint,
                        load_checkpoint, seed_all, code_digest)
from .data import amazon, yelp
from .data.splits import prepare, validate_split
from .data.features import encode_fields
from .data.sequence_dataset import SequenceDataset
from .behavior.trainer import train_din
from .indexing.trainer import train_index
from .indexing.tokenizer import SPRIIndexTokenizer
from .indexing.materialize import make_index, encode_catalog
from .indexing.index import MaterializedIndex
from .recommendation.model import SPRIRecommender
from .recommendation.trainer import train_generator
from .evaluation.metrics import evaluate
from .evaluation.diagnostics import index_statistics, popularity_metrics


class Pipeline:
    STAGES = ("prepare", "features", "train-din", "train-index", "materialize", "train", "evaluate", "report")

    def __init__(self, config, resume=False):
        self.config, self.resume = config, resume
        self.root = (Path(config["runtime"]["output"]) / config["data"]["name"]
                     / config["runtime"]["run_id"] / f"seed-{config['runtime']['seed']}").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        device = config["runtime"]["device"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        torch.set_num_threads(config["runtime"]["threads"])
        self.source_digest = code_digest()
        write_json(self.root / "requested_config.json", config)

    def signature(self, *parts):
        return digest([self.source_digest, *parts])

    def cached(self, stage, signature):
        path = self.root / f"{stage}.manifest.json"
        if not path.exists():
            return False
        manifest = read_json(path)
        if manifest["signature"] != signature:
            raise ValueError(f"{stage}: input/config/source changed; use a new runtime.run_id")
        for name, checksum in manifest["outputs"].items():
            if not (self.root / name).is_file() or file_digest(self.root / name) != checksum:
                raise ValueError(f"{stage}: artifact missing or modified: {name}")
        return True

    def commit(self, stage, signature, outputs):
        write_json(self.root / f"{stage}.manifest.json",
                   {"stage": stage, "signature": signature, "source_digest": self.source_digest,
                    "outputs": {str(Path(p)): file_digest(self.root / p) for p in outputs}})
        write_json(self.root / f"{stage}.config.json", self.config)

    def require(self, stage):
        path = self.root / f"{stage}.manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Run stage '{stage}' first in {self.root}")
        manifest = read_json(path)
        if manifest["source_digest"] != self.source_digest:
            raise ValueError(f"{stage}: source changed; use a new runtime.run_id")
        self.cached(stage, manifest["signature"])
        recorded = read_json(self.root / f"{stage}.config.json")
        sections = {"features": ["features"], "train-din": ["din"],
                    "train-index": ["index"], "train": ["model", "train"]}.get(stage, [])
        for section in sections:
            if recorded[section] != self.config[section]:
                raise ValueError(f"{stage}: {section} settings differ from the saved run")
        data_keys = {"train-din": ("history_length",),
                     "train": ("history_length", "train_sampling")}.get(stage, ())
        for key in data_keys:
            if recorded["data"][key] != self.config["data"][key]:
                raise ValueError(f"{stage}: data.{key} differs from the saved run; use the original configuration")
        if stage == "prepare":
            keys = ("name", "format", "core", "reviews", "metadata")
            if any(recorded["data"][key] != self.config["data"][key] for key in keys):
                raise ValueError("Prepared data settings differ from the saved run")
        return manifest

    def split(self):
        self.require("prepare")
        return validate_split(read_json(self.root / "split.json"))

    def input_fields(self):
        self.require("features")
        self.require("train-din")
        semantic = load_checkpoint(self.root / "semantic.pt")
        behavior = load_checkpoint(self.root / "behavior.pt")
        split = self.split()
        if semantic["split"] != split["digest"] or behavior["split"] != split["digest"]:
            raise ValueError("Features and split do not share provenance")
        if semantic["item_ids"] != split["item_ids"] or behavior["item_ids"] != split["item_ids"]:
            raise ValueError("Feature item ordering mismatch")
        return semantic["fields"] + [behavior["features"]]

    def item_index(self):
        self.require("materialize")
        index = MaterializedIndex.load(self.root / "index.json")
        split = self.split()
        if index.payload["provenance"]["split"] != split["digest"]:
            raise ValueError("Index and split mismatch")
        return index

    def run(self, stage):
        stages = self.STAGES if stage == "all" else [stage]
        for current in stages:
            seed_all(self.config["runtime"]["seed"])
            print(f"STAGE {current} -> {self.root}", flush=True)
            getattr(self, current.replace("-", "_"))()


    def prepare(self):
        c = self.config["data"]
        if not c["reviews"] or not c["metadata"]:
            raise ValueError("Set data.reviews and data.metadata explicitly")
        raw_hashes = {key: file_digest(str(c[key]).split("::", 1)[0]) for key in ("reviews", "metadata")}
        sig = self.signature(c, raw_hashes)
        if self.cached("prepare", sig):
            return
        adapter = amazon if c["format"] == "amazon" else yelp
        split = prepare(adapter.interactions(c["reviews"]), adapter.metadata(c["metadata"]), c["core"])
        write_json(self.root / "split.json", split)
        self.commit("prepare", sig, ["split.json"])
        print(split["statistics"], flush=True)

    def features(self):
        split = self.split()
        c = self.config["features"]
        sig = self.signature(c, split["digest"])
        if self.cached("features", sig):
            return
        fields = encode_fields(split, c, self.device)
        save_checkpoint(self.root / "semantic.pt", {"fields": [torch.from_numpy(x) for x in fields],
                                                   "item_ids": split["item_ids"], "split": split["digest"],
                                                   "config": c})
        self.commit("features", sig, ["semantic.pt"])

    def train_din(self):
        split = self.split()
        c = self.config
        provenance = {"signature": self.signature(c["din"], split["digest"], c["data"]["history_length"], c["runtime"]["seed"]),
                      "split": split["digest"]}
        sig = provenance["signature"]
        if self.cached("train-din", sig):
            return
        directory = self.root / "din"
        self.guard_restart(directory)
        features = train_din(split, c["din"], c["data"]["history_length"], self.device, directory,
                             provenance, self.resume and (directory / "last.pt").exists())
        save_checkpoint(self.root / "behavior.pt", {"features": features, "item_ids": split["item_ids"],
                                                   "split": split["digest"], "provenance": provenance})
        self.commit("train-din", sig, ["behavior.pt", "din/last.pt"])

    def guard_restart(self, directory):
        if (directory / "last.pt").exists() and not self.resume:
            raise FileExistsError(f"Partial run at {directory}; pass --resume or use a new runtime.run_id")

    def train_index(self):
        split, fields = self.split(), self.input_fields()
        sig = self.signature(self.config["index"], file_digest(self.root / "semantic.pt"),
                             file_digest(self.root / "behavior.pt"), self.config["runtime"]["seed"])
        if self.cached("train-index", sig):
            return
        directory = self.root / "tokenizer"
        self.guard_restart(directory)
        train_index(fields, self.config["index"], self.device, directory, {"signature": sig},
                    self.resume and (directory / "last.pt").exists())
        self.commit("train-index", sig, ["tokenizer/last.pt", "tokenizer/stage2.pt"])

    def materialize(self):
        self.require("train-index")
        split, fields = self.split(), self.input_fields()
        provenance = {"split": split["digest"], "tokenizer": file_digest(self.root / "tokenizer/stage2.pt")}
        sig = self.signature(provenance)
        if self.cached("materialize", sig):
            return
        state = load_checkpoint(self.root / "tokenizer/stage2.pt")
        model = SPRIIndexTokenizer(state["input_dims"], state["config"]).to(self.device)
        model.load_state_dict(state["model"])
        codes = encode_catalog(model, [x.to(self.device) for x in fields], self.config["index"]["batch_size"])
        index = make_index(split["item_ids"], codes, state["config"]["sizes"], provenance)
        index.save(self.root / "index.json")
        write_json(self.root / "index_statistics.json", index_statistics(index))
        self.commit("materialize", sig, ["index.json", "index_statistics.json"])

    def dataset(self, partition, index):
        c = self.config["data"]
        return SequenceDataset(self.split(), index, partition, c["history_length"], c["train_sampling"])

    def train(self):
        index, c = self.item_index(), self.config
        sig = self.signature(c["model"], c["train"], c["evaluation"], index.digest,
                             c["data"], c["runtime"]["seed"])
        if self.cached("train", sig):
            return
        directory = self.root / "generator"
        self.guard_restart(directory)
        model = SPRIRecommender(index, c["model"])
        train_generator(model, self.dataset("train", index), self.dataset("valid", index),
                        c["train"], c["evaluation"], self.device, directory,
                        {"signature": sig, "index": index.digest},
                        self.resume and (directory / "last.pt").exists())
        self.commit("train", sig, ["generator/best.pt", "generator/last.pt"])

    def evaluate(self):
        self.require("train")
        index = self.item_index()
        state = load_checkpoint(self.root / "generator/best.pt")
        if state["provenance"]["index"] != index.digest:
            raise ValueError("Checkpoint/index mismatch")
        sig = self.signature(file_digest(self.root / "generator/best.pt"), self.config["evaluation"], index.digest)
        previous = self.root / "evaluate.manifest.json"
        if previous.exists() and read_json(previous)["signature"] == sig and self.cached("evaluate", sig):
            return
        model = SPRIRecommender(index, state["model_config"]).to(self.device)
        model.load_state_dict(state["model"])
        comparison = deepcopy(read_json(self.root / "train.config.json"))
        comparison.pop("runtime")
        comparison["evaluation"] = self.config["evaluation"]
        # External path spellings are replaced by the prepared dataset content digest.
        comparison["data"].pop("reviews")
        comparison["data"].pop("metadata")
        comparison["split_digest"] = self.split()["digest"]
        comparison["source_digest"] = self.source_digest
        archive = Path("evaluations") / sig[:16]
        outputs = []
        for partition in ("valid", "test"):
            loader = DataLoader(self.dataset(partition, index), batch_size=self.config["train"]["eval_batch_size"])
            metrics, details = evaluate(model, loader, self.device, self.config["evaluation"], index)
            result = {"dataset": self.config["data"]["name"], "partition": partition,
                      "seed": self.config["runtime"]["seed"],
                      "experiment_kind": "experiment",
                      "comparison_key": digest(comparison), "index": index.digest,
                      "beam": self.config["evaluation"]["beam"], "checkpoint_epoch": state["epoch"] + 1,
                      "checkpoint_sha256": file_digest(self.root / "generator/best.pt"), "metrics": metrics}
            write_json(self.root / f"{partition}_result.json", result)
            write_json(self.root / f"{partition}_predictions.json", details)
            write_json(self.root / archive / f"{partition}_result.json", result)
            write_json(self.root / archive / f"{partition}_predictions.json", details)
            outputs += [f"{partition}_result.json", f"{partition}_predictions.json",
                        archive / f"{partition}_result.json", archive / f"{partition}_predictions.json"]
            print(partition + " " + str(metrics), flush=True)
        write_json(self.root / archive / "config.json", self.config)
        self.commit("evaluate", sig, outputs + [archive / "config.json"])

    def report(self):
        self.require("evaluate")
        details = read_json(self.root / "test_predictions.json")
        write_json(self.root / "popularity.json", popularity_metrics(self.split(), details))
        versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers", "PyYAML")}
        write_json(self.root / "environment.json", {"python": platform.python_version(), "packages": versions,
                                                   "device": str(self.device), "source_digest": self.source_digest})

