import torch
from ..indexing.losses import ramp


def soft_prefix_guidance(logits, labels, support, epoch, config):
    if not config["enabled"]:
        return logits
    bias = config["bias"] * ramp(epoch, config["start"], config["warmup"])
    if bias == 0:
        return logits
    # Only the fourth prediction position is biased. Other logits remain unchanged.
    unsupported = torch.ones((len(labels), logits.shape[-1]), dtype=torch.bool, device=logits.device)
    for row, prefix in enumerate(labels[:, :3].detach().cpu().tolist()):
        suffixes = support.get(tuple(prefix), ())
        if not suffixes:
            raise ValueError("Training target prefix is absent from the fixed catalog")
        unsupported[row, list(suffixes)] = False
    adjusted = logits.clone()
    adjusted[:, 3] = adjusted[:, 3] - bias * unsupported
    return adjusted

