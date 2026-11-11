"""Aggregate only compatible runs with distinct seeds; never import paper reference values."""
import numpy as np
from ..artifacts import read_json


def aggregate(paths):
    runs = [read_json(path) for path in paths]
    if not runs:
        raise ValueError("No result files")
    for field in ("comparison_key", "dataset", "partition", "experiment_kind"):
        if len({r[field] for r in runs}) != 1:
            raise ValueError(f"Cannot aggregate differing {field}")
    seeds = [r["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds in aggregation")
    metrics = {}
    for key in runs[0]["metrics"]:
        values = [r["metrics"][key] for r in runs]
        metrics[key] = {"mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    return {"dataset": runs[0]["dataset"], "partition": runs[0]["partition"],
            "experiment_kind": runs[0]["experiment_kind"], "seeds": seeds,
            "n_runs": len(runs), "std_ddof": 1, "metrics": metrics,
            "comparison_key": runs[0]["comparison_key"], "sources": [str(p) for p in paths]}

