"""DIN with target-dependent local activation and sampled-softmax training."""
import torch
from torch import nn


class DeepInterestNetwork(nn.Module):
    def __init__(self, item_count, dim, dropout):
        super().__init__()
        self.item_count = item_count
        self.embedding = nn.Embedding(item_count + 1, dim, padding_idx=item_count)
        self.activation = nn.Sequential(nn.Linear(4 * dim, 64), nn.PReLU(),
                                        nn.Linear(64, 16), nn.PReLU(), nn.Linear(16, 1))
        self.scorer = nn.Sequential(nn.Linear(2 * dim, 200), nn.PReLU(), nn.Dropout(dropout),
                                   nn.Linear(200, 80), nn.PReLU(), nn.Linear(80, 1))
        nn.init.normal_(self.embedding.weight, std=0.01)
        with torch.no_grad():
            self.embedding.weight[item_count].zero_()

    def forward(self, history, candidates):
        # history: [B,H]; candidates: [B,C].
        h = self.embedding(history)[:, None, :, :]
        q = self.embedding(candidates)[:, :, None, :]
        h, q = torch.broadcast_tensors(h, q)
        weights = self.activation(torch.cat([q, h, q - h, q * h], -1)).squeeze(-1)
        weights = weights.masked_fill(history[:, None, :] == self.item_count, 0)
        pooled = (weights[..., None] * h).sum(2)
        return self.scorer(torch.cat([pooled, self.embedding(candidates)], -1)).squeeze(-1)

