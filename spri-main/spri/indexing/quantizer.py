"""Nearest-neighbor VQ with the loss coefficients in the current manuscript."""
import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, size, dim, codebook_loss_weight):
        super().__init__()
        self.embedding = nn.Embedding(size, dim)
        nn.init.normal_(self.embedding.weight, std=dim ** -0.5)
        self.codebook_loss_weight = codebook_loss_weight

    def forward(self, query, candidates=None):
        # Float32 distance arithmetic avoids cancellation under reduced precision.
        x = query.float()
        if candidates is None:
            w = self.embedding.weight.float()
            distances = x.square().sum(-1, keepdim=True) + w.square().sum(-1) - 2 * x @ w.T
            codes = distances.argmin(-1)
        else:
            w = self.embedding(candidates).float()
            distances = (x[:, None, :] - w).square().sum(-1)
            codes = candidates.gather(1, distances.argmin(-1, keepdim=True)).squeeze(1)
        selected = self.embedding(codes)
        encoder_loss = F.mse_loss(query, selected.detach())
        codebook_loss = F.mse_loss(query.detach(), selected)
        loss = encoder_loss + self.codebook_loss_weight * codebook_loss
        straight_through = query + (selected - query).detach()
        return straight_through, codes, loss

    @torch.no_grad()
    def initialize(self, points, iterations):
        if iterations <= 0:
            return
        n, k = len(points), self.embedding.num_embeddings
        centers = points[torch.randperm(n, device=points.device)[:min(n, k)]].clone()
        if n < k:
            extra = points[torch.randint(n, (k - n,), device=points.device)]
            centers = torch.cat([centers, extra + 1e-3 * torch.randn_like(extra)])
        for _ in range(iterations):
            distances = points.square().sum(-1, keepdim=True) + centers.square().sum(-1) - 2 * points @ centers.T
            codes = distances.argmin(-1)
            totals = torch.zeros_like(centers).index_add_(0, codes, points)
            counts = torch.bincount(codes, minlength=k)
            used = counts > 0
            centers[used] = totals[used] / counts[used, None]
        self.embedding.weight.copy_(centers)

