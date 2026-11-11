"""Behavioral refinement with two independently controlled prefix paths."""
import torch
from torch import nn
from .quantizer import VectorQuantizer


def mlp(input_dim, output_dim, hidden_dim, dropout=0.0):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden_dim, output_dim))


class PrefixConditionedRefinement(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        d = config["dim"]
        self.config = dict(config)
        self.encoder = mlp(input_dim, d, d, config["encoder_dropout"])
        self.conditioner = mlp(d, d, d)
        self.quantizer = VectorQuantizer(config["sizes"][3], d, config["eta"][3])
        decoder_dim = 4 * d if config["conditional_reconstruction"] else d
        self.decoder = mlp(decoder_dim, input_dim, 2 * d)

    def query(self, latent, prefix):
        if not self.config["conditional_quantization"]:
            return latent
        context = torch.stack([z.detach() for z in prefix]).mean(0)
        if self.training and self.config["conditional_dropout"] > 0:
            mask = torch.rand((len(context), 1), device=context.device) >= self.config["conditional_dropout"]
            context = context * mask
        return latent + self.conditioner(context)

    def forward(self, features, prefix):
        latent = self.encoder(features)
        quantized, codes, vq_loss = self.quantizer(self.query(latent, prefix))
        sources = [z.detach() for z in prefix] + [quantized] if self.config["conditional_reconstruction"] else [quantized]
        reconstruction = self.decoder(torch.cat(sources, -1))
        return reconstruction, quantized, codes, vq_loss, latent

