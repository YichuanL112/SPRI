"""SPRIIndexTokenizer: three semantic levels followed by prefix-conditioned behavior."""
import torch
from torch import nn
from torch.nn import functional as F
from .quantizer import VectorQuantizer
from .pgq import PrefixGuidedQuantization
from .pcr import PrefixConditionedRefinement, mlp
from .losses import SemanticToBehaviorAlignment


class SPRIIndexTokenizer(nn.Module):
    def __init__(self, input_dims, config):
        super().__init__()
        self.config = config
        self.input_dims = list(input_dims)
        d = config["dim"]
        self.encoders = nn.ModuleList([mlp(input_dims[l], d, d, config["encoder_dropout"]) for l in range(3)])
        self.conditioners = nn.ModuleList([nn.Identity(), mlp(d, d, d), mlp(d, d, d)])
        self.quantizers = nn.ModuleList([VectorQuantizer(config["sizes"][l], d, config["eta"][l]) for l in range(3)])
        self.decoders = nn.ModuleList([mlp((l + 1) * d, input_dims[l], 2 * d) for l in range(3)])
        self.pgq = PrefixGuidedQuantization(config["sizes"][1], config["sizes"][2], config["pgq"])
        self.pcr = PrefixConditionedRefinement(input_dims[3], config)
        self.sem2beh = SemanticToBehaviorAlignment(d, input_dims[3])

    def semantic(self, fields):
        quantized, codes, losses, latents = [], [], [], []
        for level in range(3):
            latent = self.encoders[level](fields[level])
            query = latent
            if level > 0 and self.config["semantic_conditioning"]:
                context = quantized[-1]
                if self.training and self.config["semantic_conditional_dropout"] > 0:
                    # One Bernoulli mask per item and level; no inverted-dropout scaling.
                    keep = torch.rand((len(context), 1), device=context.device) >= self.config["semantic_conditional_dropout"]
                    context = context * keep
                query = query + self.conditioners[level](context)
            candidates = self.pgq.candidates(codes[-1]) if level == 2 else None
            z, code, loss = self.quantizers[level](query, candidates)
            quantized.append(z)
            codes.append(code)
            losses.append(loss + F.mse_loss(self.decoders[level](torch.cat(quantized, -1)), fields[level]))
            latents.append(latent)
        self.pgq.observe(codes[1], codes[2])
        return quantized, codes, losses, latents

    def forward(self, fields, stage=2):
        quantized, codes, losses, latents = self.semantic(fields)
        if stage == 2:
            reconstruction, z, code, loss, latent = self.pcr(fields[3], quantized)
            losses.append(loss + F.mse_loss(reconstruction, fields[3]))
            codes.append(code)
            latents.append(latent)
            quantized.append(z)
            total = 0.5 * sum(losses[:3]) + losses[3]
        else:
            total = sum(losses)
        return {"loss": total, "codes": torch.stack(codes, -1),
                "quantized": quantized, "latents": latents, "level_losses": losses}

    def freeze_shallow(self, frozen):
        for components in (self.encoders, self.conditioners, self.quantizers, self.decoders):
            for level in (0, 1):
                components[level].requires_grad_(not frozen)

    @torch.no_grad()
    def initialize_codebooks(self, fields):
        was_training = self.training
        self.eval()
        limit = min(len(fields[0]), self.config["kmeans_max_items"])
        ids = torch.randperm(len(fields[0]), device=fields[0].device)[:limit]
        x = [f[ids] for f in fields]
        prefix, codes = [], []
        for level in range(3):
            query = self.encoders[level](x[level])
            if level and self.config["semantic_conditioning"]:
                query = query + self.conditioners[level](prefix[-1])
            self.quantizers[level].initialize(query, self.config["kmeans_iters"])
            candidates = self.pgq.candidates(codes[-1]) if level == 2 else None
            z, code, _ = self.quantizers[level](query, candidates)
            prefix.append(z)
            codes.append(code)
        self.train(was_training)

    @torch.no_grad()
    def initialize_behavior(self, fields):
        was_training = self.training
        self.eval()
        ids = torch.randperm(len(fields[0]), device=fields[0].device)[:self.config["kmeans_max_items"]]
        x = [f[ids] for f in fields]
        prefix, _, _, _ = self.semantic(x)
        self.pcr.quantizer.initialize(self.pcr.query(self.pcr.encoder(x[3]), prefix), self.config["kmeans_iters"])
        self.train(was_training)

