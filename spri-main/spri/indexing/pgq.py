"""Stateful parent-guided candidate allocation, including checkpointed clocks."""
import torch
from torch import nn


class PrefixGuidedQuantization(nn.Module):
    def __init__(self, parents, children, config):
        super().__init__()
        self.config = dict(config)
        self.parent_count, self.child_count = parents, children
        if children % parents:
            raise ValueError("The fine codebook must divide into equal parent blocks")
        width = children // parents
        base = torch.arange(children).reshape(parents, width)
        self.register_buffer("base", base)
        self.register_buffer("counts", torch.zeros(parents, children))
        self.register_buffer("shortlist", torch.stack([
            torch.tensor((row.tolist() + [j for j in range(children) if j not in row.tolist()])[:config["topk"]])
            for row in base
        ]))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_refresh", torch.tensor(-1, dtype=torch.long))

    @torch.no_grad()
    def refresh_if_due(self):
        cfg, step = self.config, int(self.step)
        if not cfg["enabled"] or step < cfg["warmup_steps"]:
            return
        if int(self.last_refresh) >= 0 and step - int(self.last_refresh) < cfg["refresh_steps"]:
            return
        counts = self.counts.cpu()
        popularity = counts.sum(0)
        scores = counts / (popularity + 1e-6).pow(cfg["gamma"])
        mass, mean = counts.sum(1), counts.sum(1).mean()
        # Stable sorting makes ties reproducible across repeated materializations.
        global_order = torch.argsort(popularity.masked_fill(popularity == 0, float("inf")), stable=True).tolist()
        rows = []
        for parent in range(self.parent_count):
            base = self.base[parent].tolist()
            reserve = min(cfg["reserve"], len(base), cfg["topk"])
            rarity = max(0.0, min(1.0, 1 - float(mass[parent] / (mean + 1e-6)))) if mean > 0 else 0.0
            mix = max(0.0, min(1.0, cfg["global_fraction"] + cfg["parent_adapt"] * rarity))
            remaining = cfg["topk"] - reserve
            n_global = round(remaining * mix)
            n_parent = remaining - n_global
            parent_order = torch.argsort(scores[parent], descending=True, stable=True).tolist()
            candidates = base[:reserve] + parent_order[:n_parent] + global_order[:n_global] + base + list(range(self.child_count))
            distinct = list(dict.fromkeys(candidates))[:cfg["topk"]]
            rows.append(distinct)
        self.shortlist.copy_(torch.tensor(rows, device=self.shortlist.device))
        self.last_refresh.fill_(step)

    def candidates(self, parent_codes):
        if not self.config["enabled"]:
            return None
        if self.training:
            self.refresh_if_due()
        table = self.base if int(self.step) < self.config["warmup_steps"] else self.shortlist
        return table[parent_codes]

    @torch.no_grad()
    def observe(self, parent_codes, child_codes):
        if not self.training or not self.config["enabled"]:
            return
        batch = torch.bincount(parent_codes * self.child_count + child_codes,
                               minlength=self.parent_count * self.child_count).reshape_as(self.counts)
        rho = self.config["decay"]
        self.counts.mul_(rho).add_(batch, alpha=1 - rho)
        self.step.add_(1)

