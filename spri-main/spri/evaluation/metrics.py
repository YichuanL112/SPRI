import math
import numpy as np
import torch


def first_rank(predictions, target):
    target = tuple(target)
    return next((rank for rank, code in enumerate(predictions, 1) if tuple(code) == target), None)


def rank_metrics(rank, topk):
    return {f"{metric}@{k}": (float(rank is not None and rank <= k) if metric == "recall"
                              else (1 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0))
            for k in topk for metric in ("recall", "ndcg")}


@torch.no_grad()
def evaluate(model, loader, device, evaluation, index):
    model.eval()
    details = []
    for batch in loader:
        tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        predictions, scores = model.generate(tensors, evaluation["beam"])
        for row, (codes, log_probs) in enumerate(zip(predictions.cpu().tolist(), scores.cpu().tolist())):
            target_row = int(batch["target"][row])
            rank = first_rank(codes, index.tokens[target_row].tolist())
            details.append({"user_id": batch["user_id"][row],
                            "target_position": int(batch["target_position"][row]),
                            "target_row": target_row, "rank": rank,
                            "predictions": codes, "scores": log_probs,
                            "valid_identifiers": [tuple(c) in index.inverse for c in codes],
                            "metrics": rank_metrics(rank, evaluation["topk"])})
    if not details:
        raise ValueError("Evaluation partition is empty")
    keys = details[0]["metrics"]
    means = {key: float(np.mean([r["metrics"][key] for r in details])) for key in keys}
    means["valid_identifier_rate"] = float(np.mean([v for r in details for v in r["valid_identifiers"]]))
    return means, details

