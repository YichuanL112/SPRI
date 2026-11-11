"""Explicit configuration, strict overrides, and portable path resolution."""
from copy import deepcopy
import yaml

DEFAULTS = {
    "runtime": {"seed": 2025, "device": "auto", "threads": 4, "output": "runs", "run_id": "default"},
    "data": {"name": "beauty", "format": "amazon", "reviews": "", "metadata": "",
             "core": 5, "history_length": 20, "train_sampling": "all_prefixes"},
    "features": {"model": "sentence-transformers/sentence-t5-base",
                 "revision": None, "batch_size": 128, "normalize": False},
    "din": {"dim": 96, "epochs": 10, "batch_size": 128, "lr": 0.001,
            "weight_decay": 0.00001, "negatives": 32, "dropout": 0.1},
    "index": {
        "dim": 96, "sizes": [256, 256, 512, 512], "eta": [0.25, 0.2, 0.2, 0.15],
        "encoder_dropout": 0.15, "conditional_dropout": 0.1,
        "semantic_conditional_dropout": 0.1,
        "semantic_conditioning": True, "conditional_quantization": True, "conditional_reconstruction": True,
        "stage1_epochs": 2000, "stage2_epochs": 2000, "unfreeze_epoch": 600,
        "batch_size": 1024, "lr": 0.0005, "codebook_lr": 0.0001, "weight_decay": 0.00001,
        "kmeans_iters": 10, "kmeans_max_items": 50000,
        "pgq": {"enabled": True, "topk": 24, "warmup_steps": 200, "refresh_steps": 250,
                "decay": 0.99, "gamma": 0.35, "global_fraction": 0.35,
                "parent_adapt": 0.4, "reserve": 2},
        "alignment": {"weight": 0.0012, "temperature": 0.07, "start": 0, "warmup": 100,
                      "view_dropout": 0.1, "view_noise": 0.01},
        "sem2beh": {"enabled": True, "weight": 0.003, "start": 50, "warmup": 150,
                    "detach_source": True}
    },
    "model": {"dim": 128, "ff_dim": 1024, "heads": 6, "head_dim": 64,
              "encoder_layers": 4, "decoder_layers": 4, "dropout": 0.1,
              "shc": {"enabled": True, "dropout": 0.1, "temperature": 1.0,
                      "learnable_scale": False, "scale_init": 1.0},
              "guidance": {"enabled": True, "bias": 0.8, "start": 50, "warmup": 100}},
    "train": {"epochs": 200, "batch_size": 128, "eval_batch_size": 32, "accumulation": 2,
              "lr": 0.001, "weight_decay": 0.1, "warmup_steps": 10000,
              "patience": 20, "max_grad_norm": 1.0},
    "evaluation": {"beam": 50, "topk": [5, 10], "selection_metric": "ndcg@10"}
}


def merge(base, updates, prefix=""):
    for key, value in updates.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected mapping: {prefix}{key}")
            merge(base[key], value, f"{prefix}{key}.")
        else:
            old = base[key]
            if old is not None and not isinstance(value, type(old)):
                # YAML permits an integer value for a floating-point parameter.
                if isinstance(old, float) and type(value) is int:
                    value = float(value)
                else:
                    raise ValueError(f"Invalid type for {prefix}{key}: {value!r}")
            base[key] = value
    return base


def validate(c):
    if c["data"]["format"] not in {"amazon", "yelp"}:
        raise ValueError("data.format must be amazon or yelp")
    if c["data"]["train_sampling"] not in {"all_prefixes", "last_prefix"}:
        raise ValueError("Unknown training sample policy")
    sizes = c["index"]["sizes"]
    if len(sizes) != 4 or any(type(k) is not int or k < 2 for k in sizes):
        raise ValueError("SPRI requires four codebooks with sizes >= 2")
    if sizes[2] % sizes[1]:
        raise ValueError("PGQ requires K2 divisible by K1 for contiguous warmup blocks")
    pgq = c["index"]["pgq"]
    if not 1 <= pgq["topk"] <= sizes[2] or pgq["refresh_steps"] < 1:
        raise ValueError("Invalid PGQ shortlist/refresh settings")
    if not 0 <= pgq["decay"] < 1 or not 0 <= pgq["global_fraction"] <= 1:
        raise ValueError("Invalid PGQ decay/global fraction")
    if len(c["index"]["eta"]) != 4 or min(c["index"]["eta"]) < 0:
        raise ValueError("index.eta must contain four nonnegative values")
    for section, names in {
        "runtime": ["threads"], "data": ["core", "history_length"],
        "features": ["batch_size"],
        "din": ["dim", "epochs", "batch_size", "negatives"],
        "index": ["dim", "stage2_epochs", "batch_size", "kmeans_max_items"],
        "model": ["dim", "ff_dim", "heads", "head_dim", "encoder_layers", "decoder_layers"],
        "train": ["epochs", "batch_size", "eval_batch_size", "accumulation", "patience"]
    }.items():
        if any(type(c[section][k]) is not int or c[section][k] <= 0 for k in names):
            raise ValueError(f"Positive integer required in {section}: {names}")
    if c["index"]["stage1_epochs"] < 0 or c["index"]["unfreeze_epoch"] < 0:
        raise ValueError("Stage epoch settings must be nonnegative")
    if not c["evaluation"]["topk"] or min(c["evaluation"]["topk"]) < 1:
        raise ValueError("evaluation.topk must be nonempty and positive")
    if c["evaluation"]["beam"] < max(c["evaluation"]["topk"]):
        raise ValueError("Beam width must cover all evaluation cutoffs")
    if c["evaluation"]["selection_metric"] not in {
        f"{m}@{k}" for m in ("recall", "ndcg") for k in c["evaluation"]["topk"]
    }:
        raise ValueError("selection_metric must be among reported metrics")
    for value in [c["din"]["dropout"], c["index"]["encoder_dropout"],
                  c["index"]["conditional_dropout"], c["index"]["semantic_conditional_dropout"],
                  c["index"]["alignment"]["view_dropout"],
                  c["model"]["dropout"], c["model"]["shc"]["dropout"]]:
        if not 0 <= value < 1:
            raise ValueError("Dropout probabilities must be in [0, 1)")
    if min(c["model"]["shc"]["temperature"], c["index"]["alignment"]["temperature"]) <= 0:
        raise ValueError("Temperatures must be positive")
    return c


def load_config(paths=(), overrides=()):
    c = deepcopy(DEFAULTS)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            merge(c, yaml.safe_load(f) or {})
    for assignment in overrides:
        key, sep, raw = assignment.partition("=")
        if not sep:
            raise ValueError(f"Expected key=value: {assignment}")
        value = yaml.safe_load(raw)
        update = value
        for part in reversed(key.split(".")):
            update = {part: update}
        merge(c, update)
    return validate(c)

