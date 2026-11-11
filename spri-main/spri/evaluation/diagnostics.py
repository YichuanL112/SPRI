from collections import Counter
import numpy as np
from ..data.splits import train_sequences


def index_statistics(index):
    prefixes = Counter(map(tuple, index.base_codes[:, :3].tolist()))
    return {"catalog_size": len(index.item_ids), "semantic_prefixes": len(prefixes),
            "max_prefix_occupancy": max(prefixes.values()),
            "mean_prefix_occupancy": float(np.mean(list(prefixes.values()))),
            "base_code_usage": [len(set(index.base_codes[:, l].tolist())) for l in range(4)],
            "base_sizes": index.payload["base_sizes"], "materialized_sizes": index.sizes,
            "vocab_size": index.vocab_size, "max_collision_group": index.payload["collision_max_group"]}


def popularity_metrics(split, details):
    counts = Counter(item for seq in train_sequences(split) for item in seq)
    ordered = sorted(range(len(split["item_ids"])), key=lambda item: (-counts[item], split["item_ids"][item]))
    total, cumulative = sum(counts.values()), 0
    groups = {}
    for item in ordered:
        # Boundary items stay in the group containing their preceding cumulative mass.
        groups[item] = "head" if cumulative < total / 3 else "middle" if cumulative < 2 * total / 3 else "tail"
        cumulative += counts[item]
    result = {}
    for group in ("head", "middle", "tail"):
        rows = [row for row in details if groups[row["target_row"]] == group]
        result[group] = {"examples": len(rows), "items": sum(g == group for g in groups.values()),
                         "metrics": ({k: float(np.mean([r["metrics"][k] for r in rows]))
                                      for k in rows[0]["metrics"]} if rows else {})}
    return result

