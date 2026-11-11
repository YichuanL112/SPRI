"""The split is created once and consumed by DIN, indexing and recommendation."""
from collections import Counter, defaultdict
from ..artifacts import digest


def prepare(interactions, metadata, core=5):
    events = [r for r in interactions if r["item"] in metadata]
    while True:
        users = Counter(r["user"] for r in events)
        items = Counter(r["item"] for r in events)
        retained = [r for r in events if users[r["user"]] >= max(3, core) and items[r["item"]] >= core]
        if len(retained) == len(events):
            break
        events = retained
    if not events:
        raise ValueError("No interactions remain after metadata intersection and core filtering")
    item_ids = sorted({r["item"] for r in events})
    item2row = {item: i for i, item in enumerate(item_ids)}
    groups = defaultdict(list)
    for event in events:
        groups[event["user"]].append(event)
    split_users = []
    for user, rows in sorted(groups.items()):
        rows.sort(key=lambda r: (r["timestamp"], r["ordinal"]))
        sequence = [item2row[r["item"]] for r in rows]
        split_users.append({"user_id": user, "sequence": sequence,
                            "event_ids": [r["ordinal"] for r in rows],
                            "train_end": len(sequence) - 2})
    result = {"schema": 1, "item_ids": item_ids,
              "metadata": [metadata[item] for item in item_ids], "users": split_users,
              "statistics": {"users": len(groups), "items": len(item_ids), "interactions": len(events)},
              "protocol": {"core": core, "split": "chronological_leave_one_out", "tie_break": "source_ordinal"}}
    result["digest"] = digest(result)
    return result


def validate_split(split):
    payload = {k: v for k, v in split.items() if k != "digest"}
    if split.get("digest") != digest(payload):
        raise ValueError("Prepared split digest mismatch")
    n = len(split["item_ids"])
    if len(set(split["item_ids"])) != n:
        raise ValueError("Duplicate catalog item IDs")
    for user in split["users"]:
        if user["train_end"] != len(user["sequence"]) - 2 or user["train_end"] < 1:
            raise ValueError("Invalid leave-one-out boundary")
        if not all(0 <= item < n for item in user["sequence"]):
            raise ValueError("Item outside catalog")
    return split


def samples(split, partition, history_length, policy="all_prefixes"):
    for user in split["users"]:
        seq, end = user["sequence"], user["train_end"]
        if partition == "train":
            targets = range(1, end) if policy == "all_prefixes" else [end - 1]
        elif partition == "valid":
            targets = [end]
        elif partition == "test":
            targets = [end + 1]
        else:
            raise ValueError(f"Unknown partition: {partition}")
        for pos in targets:
            if pos < 1:
                continue
            yield {"user_id": user["user_id"], "target_position": pos,
                   "history": seq[max(0, pos - history_length):pos], "target": seq[pos]}


def train_sequences(split):
    return [u["sequence"][:u["train_end"]] for u in split["users"]]

