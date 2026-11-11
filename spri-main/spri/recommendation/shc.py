"""SHC owns role embeddings, item fusion, separators and boundary sentinels."""
import math
import torch
from torch import nn
from ..indexing.index import PAD, SEP, LEFT, RIGHT


class StructuredHistoryConstruction(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        self.config = config
        self.roles = nn.Embedding(4, dim)
        self.query, self.key, self.value = (nn.Linear(dim, dim) for _ in range(3))
        self.gate = nn.Linear(2 * dim, dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(config["dropout"])
        scale = torch.tensor(float(config["scale_init"]))
        if config["learnable_scale"]:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)

    def contextualize(self, tokens, embedding):
        x = embedding(tokens) + self.roles.weight
        semantic, behavior = x[..., :3, :], x[..., 3, :]
        scores = (self.query(behavior).unsqueeze(-2) * self.key(semantic)).sum(-1)
        scores = scores / (math.sqrt(behavior.shape[-1]) * self.config["temperature"])
        context = (scores.softmax(-1).unsqueeze(-1) * self.value(semantic)).sum(-2)
        gate = torch.sigmoid(self.gate(torch.cat([behavior, context], -1)))
        behavior = behavior + self.scale * self.dropout(gate * self.output(context))
        return torch.cat([x[..., :3, :], behavior.unsqueeze(-2)], -2)

    def forward(self, history, item_mask, embedding):
        if history.ndim != 3 or history.shape[-1] != 4 or item_mask.shape != history.shape[:2]:
            raise ValueError("SHC expects history [batch,items,4] and item_mask [batch,items]")
        if (item_mask[:, 1:] & ~item_mask[:, :-1]).any():
            raise ValueError("Valid items must precede padding")
        batch, length, _ = history.shape
        if self.config["enabled"]:
            x = self.contextualize(history, embedding)
            sep = embedding(torch.full((batch, length, 1), SEP, device=history.device))
            blocks = torch.cat([x, sep], -2).flatten(1, 2)
            block_mask = item_mask.repeat_interleave(5, 1)
            # Keep the right sentinel immediately after the last real item.
            output = embedding(torch.full((batch, 5 * length + 2), PAD, device=history.device))
            output[:, 0] = embedding(torch.tensor(LEFT, device=history.device))
            output[:, 1:1 + 5 * length] = blocks.masked_fill(~block_mask[..., None], 0)
            right = 1 + 5 * item_mask.sum(-1)
            output[torch.arange(batch, device=history.device), right] = embedding(torch.tensor(RIGHT, device=history.device))
            positions = torch.arange(output.shape[1], device=history.device)[None, :]
            return output, positions <= right[:, None]
        # Whole-component ablation: flat codes, without SHC roles/fusion/sentinels.
        output = embedding(history).flatten(1, 2)
        return output, item_mask.repeat_interleave(4, 1)

