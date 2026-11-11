from collections import defaultdict
import numpy as np
import torch
from ..artifacts import digest
from .index import MaterializedIndex


def make_index(item_ids, base_codes, codebook_sizes, provenance):
    raw = np.asarray(base_codes, dtype=np.int64)
    if raw.shape != (len(item_ids), 4) or np.any(raw < 0) or np.any(raw >= codebook_sizes):
        raise ValueError("Base codes are outside the four codebooks")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Item IDs must be unique")
    final = raw.copy()
    counts = defaultdict(int)
    # Stable within-tuple ranks depend on item identity, not the order of encoding batches.
    for row in sorted(range(len(item_ids)), key=lambda i: item_ids[i]):
        key = tuple(raw[row])
        final[row, 3] += counts[key] * codebook_sizes[3]
        counts[key] += 1
    sizes = list(codebook_sizes)
    sizes[3] = max(sizes[3], int(final[:, 3].max()) + 1)
    payload = {"schema": 1, "item_ids": list(item_ids), "base_codes": raw.tolist(),
               "codes": final.tolist(), "base_sizes": list(codebook_sizes), "sizes": sizes,
               "provenance": provenance, "collision_max_group": max(counts.values())}
    payload["digest"] = digest(payload)
    return MaterializedIndex(payload)


@torch.no_grad()
def encode_catalog(model, fields, batch_size):
    model.eval()  # No dropout, co-occurrence updates or candidate refresh during export.
    codes = [model([x[start:start + batch_size] for x in fields])["codes"].cpu().numpy()
             for start in range(0, len(fields[0]), batch_size)]
    return np.concatenate(codes)

