"""Inter-level, intra-level and semantic-to-behavior alignment objectives."""
import torch
from torch import nn
from torch.nn import functional as F


def ramp(epoch, start, warmup):
    if epoch < start:
        return 0.0
    return min(1.0, (epoch - start) / warmup) if warmup > 0 else 1.0


def matching(a, b, temperature):
    if len(a) < 2:
        return a.sum() * 0
    logits = F.normalize(a.float(), dim=-1) @ F.normalize(b.float(), dim=-1).T / temperature
    return F.cross_entropy(logits, torch.arange(len(a), device=a.device))


def alignment(semantic, behavioral, config):
    inter = matching(semantic, behavioral.detach(), config["temperature"])
    views = [F.dropout(semantic, config["view_dropout"], training=True)
             + torch.randn_like(semantic) * config["view_noise"] for _ in range(2)]
    intra = (matching(views[0], views[1], config["temperature"])
             + matching(views[1], views[0], config["temperature"])) / 2
    return (inter + 2 * intra) / 3


class SemanticToBehaviorAlignment(nn.Module):
    """Predict frozen DIN features from semantic quantized vectors."""
    def __init__(self, dim, target_dim):
        super().__init__()
        self.predictor = nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, target_dim))

    def forward(self, semantic, target, detach_source=True):
        prediction = self.predictor(semantic.detach() if detach_source else semantic)
        return (1 - F.cosine_similarity(prediction, target.detach(), dim=-1)).mean()





