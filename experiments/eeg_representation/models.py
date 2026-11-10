"""Shared, modest EEG encoders and explicitly scoped training objectives.

Inputs are normalized EEG [batch, channel, time]. No labels or simulator
metadata enter an encoder. Forecast heads receive history only. Torch is an
optional dependency: this module is imported only for fitting/evaluation.
"""
from __future__ import annotations

import copy
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class PatchEncoder(nn.Module):
    def __init__(self, channels=1, embedding_dim=128, layers=4, heads=4, patch_samples=32, **_):
        super().__init__()
        self.channels, self.embedding_dim, self.patch_samples = channels, embedding_dim, patch_samples
        self.patch = nn.Conv1d(channels, embedding_dim, patch_samples, stride=patch_samples)
        layer = nn.TransformerEncoderLayer(embedding_dim, heads, embedding_dim * 4,
                                            dropout=0., batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embedding_dim)

    def tokens(self, x):
        x = F.pad(x, (0, (-x.shape[-1]) % self.patch_samples))
        z = self.patch(x).transpose(1, 2)
        position = torch.arange(z.shape[1], device=z.device, dtype=z.dtype)[:, None]
        scale = torch.exp(torch.arange(0, self.embedding_dim, 2, device=z.device,
                                      dtype=z.dtype) * (-math.log(10000.) / self.embedding_dim))
        pe = torch.zeros(z.shape[1], self.embedding_dim, device=z.device, dtype=z.dtype)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        return self.norm(self.transformer(z + pe[None]))

    def forward(self, x):
        return self.tokens(x).mean(dim=1)


class CausalConv(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x):
        # GroupNorm is over the available history only. No future is supplied.
        return x + F.gelu(self.norm(self.conv(F.pad(x, (self.padding, 0)))))


class TCNEncoder(nn.Module):
    def __init__(self, channels=1, embedding_dim=128, layers=4, patch_samples=32, **_):
        super().__init__()
        self.channels, self.embedding_dim, self.patch_samples = channels, embedding_dim, patch_samples
        self.patch = nn.Conv1d(channels, embedding_dim, patch_samples, stride=patch_samples)
        self.blocks = nn.Sequential(*(CausalConv(embedding_dim, 2 ** i) for i in range(layers)))

    def tokens(self, x):
        x = F.pad(x, (0, (-x.shape[-1]) % self.patch_samples))
        return self.blocks(self.patch(x)).transpose(1, 2)

    def forward(self, x):
        return self.tokens(x).mean(dim=1)


def encoder_config(config, channels=1):
    cfg = dict(config.get('model', {}))
    sample_hz = config.get('simulation', {}).get('model_sample_rate_hz', config.get('observation', {}).get('sample_hz', config.get('sample_hz', 500)))
    return dict(architecture=cfg.get('architecture', 'transformer'), channels=channels,
                embedding_dim=int(cfg.get('embedding_dim', 128)), layers=int(cfg.get('layers', 4)),
                heads=int(cfg.get('heads', 4)), covariance_rank=int(cfg.get('covariance_rank',4)),
                patch_samples=int(cfg.get('patch_samples', round(cfg.get('patch_ms', 64) * sample_hz / 1000))))


def make_encoder(config):
    cfg = dict(config)
    architecture = cfg.pop('architecture', 'transformer')
    if architecture not in {'transformer', 'tcn'}:
        raise ValueError(f'Unknown encoder architecture: {architecture}')
    result = (PatchEncoder if architecture == 'transformer' else TCNEncoder)(**cfg)
    if sum(p.numel() for p in result.parameters()) > 2_000_000:
        raise ValueError('Shared encoder exceeds the declared two-million-parameter ceiling')
    return result


class ReconstructionModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embedding_dim, encoder.channels * encoder.patch_samples)

    def forward(self, x):
        z = self.head(self.encoder.tokens(x))
        z = z.reshape(len(x), -1, self.encoder.channels, self.encoder.patch_samples)
        return z.permute(0, 2, 1, 3).flatten(2)[..., :x.shape[-1]]


class MultiTaskModel(nn.Module):
    def __init__(self, encoder, task_specs: Mapping):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({name: nn.Linear(encoder.embedding_dim,
            int(spec['n_classes']) if spec['kind'] == 'classification' else 1)
            for name, spec in task_specs.items()})

    def forward(self, x, task):
        return self.heads[task](self.encoder(x))


class GaussianForecast(nn.Module):
    """A positive-diagonal plus low-rank Gaussian; AR is a triangular reference.

    The AR reference models residual y[j]-mean[j] = rho[j] * residual[j-1]
    + sigma[j] epsilon[j]. Teacher forcing is used ONLY in joint likelihood
    evaluation, never as encoder input or for free-running forecasts.
    """
    def __init__(self, encoder, target_shape, kind='lowrank', rank=4):
        super().__init__()
        self.encoder, self.target_shape, self.kind = encoder, tuple(target_shape), kind
        self.dimension, self.rank = math.prod(target_shape), rank
        extra = rank if kind == 'lowrank' else (1 if kind == 'ar' else 0)
        self.head = nn.Linear(encoder.embedding_dim, self.dimension * (2 + extra))

    def forward(self, x):
        p = self.head(self.encoder(x)).reshape(len(x), self.dimension, -1)
        mean, diagonal = p[..., 0], F.softplus(p[..., 1]) + 1e-5
        if self.kind == 'lowrank':
            return torch.distributions.LowRankMultivariateNormal(mean, p[..., 2:] / math.sqrt(self.rank), diagonal)
        if self.kind == 'diagonal':
            return torch.distributions.Independent(torch.distributions.Normal(mean, diagonal.sqrt()), 1)
        return ARGaussian(mean, diagonal.sqrt(), torch.tanh(p[..., 2]) * .98)


class ARGaussian:
    def __init__(self, mean, scale, rho):
        self.mean, self.scale, self.rho = mean, scale, rho

    def log_prob(self, value):
        residual = value - self.mean
        previous = F.pad(residual[..., :-1], (1, 0))
        return torch.distributions.Normal(self.rho * previous, self.scale).log_prob(residual).sum(-1)

    def rsample(self, sample_shape=torch.Size()):
        eps = torch.randn(*sample_shape, *self.mean.shape, device=self.mean.device) * self.scale
        out = [eps[..., 0]]
        for j in range(1, self.mean.shape[-1]):
            out.append(eps[..., j] + self.rho[..., j] * out[-1])
        return torch.stack(out, -1) + self.mean

    @property
    def variance(self):
        values = [self.scale[..., 0].square()]
        for j in range(1, self.mean.shape[-1]):
            values.append(self.scale[..., j].square() + self.rho[..., j].square() * values[-1])
        return torch.stack(values, -1)


class MomentumContrast(nn.Module):
    """EMA targets + InfoNCE negatives + explicit variance regularization.

    No amplitude rescaling invariance is imposed: only weak additive noise is
    used. Batch size >=2 is required; singleton batches are paired by loader.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(nn.Linear(encoder.embedding_dim, encoder.embedding_dim), nn.GELU(),
                                       nn.Linear(encoder.embedding_dim, encoder.embedding_dim))
        self.teacher = copy.deepcopy(encoder).requires_grad_(False)
        self.teacher_projector = copy.deepcopy(self.projector).requires_grad_(False)

    def loss(self, x, temperature=.2):
        z = self.projector(self.encoder(x + .02 * torch.randn_like(x)))
        with torch.no_grad():
            target = self.teacher_projector(self.teacher(x + .02 * torch.randn_like(x)))
        similarity = F.normalize(z, dim=-1) @ F.normalize(target, dim=-1).T / temperature
        contrast = F.cross_entropy(similarity, torch.arange(len(x), device=x.device))
        variance = F.relu(1 - torch.sqrt(z.var(0, unbiased=False) + 1e-4)).mean()
        centered = z - z.mean(0)
        covariance = centered.T @ centered / max(len(x) - 1, 1)
        covariance = covariance - torch.diag_embed(covariance.diagonal())
        return contrast + variance + .01 * covariance.square().mean()

    @torch.no_grad()
    def update_teacher(self, momentum=.99):
        for source, target in [(self.encoder, self.teacher), (self.projector, self.teacher_projector)]:
            for p, q in zip(source.parameters(), target.parameters()):
                q.mul_(momentum).add_(p, alpha=1-momentum)


def gap_mask(x, fraction=.4, gap_samples=64, generator=None):
    if not 0 < fraction < 1 or gap_samples < 1:
        raise ValueError('Mask fraction must lie in (0,1); gaps must be positive')
    mask = torch.zeros_like(x, dtype=torch.bool)
    target_count = max(1, min(x.shape[-1]-1, round(x.shape[-1] * fraction)))
    # Random contiguous gaps; exact fraction, with one visible sample minimum.
    for i in range(len(x)):
        order = torch.randperm(math.ceil(x.shape[-1] / gap_samples), generator=generator).tolist()
        indices = [j for block in order for j in range(block*gap_samples, min((block+1)*gap_samples, x.shape[-1]))]
        mask[i, :, indices[:target_count]] = True
    return mask


def masked_loss(prediction, target, mask, kind='mse'):
    if not mask.any():
        raise ValueError('No masked targets')
    error = prediction - target
    return (error.abs() if kind == 'l1' else error.square())[mask].mean()


def spectral_envelope_loss(prediction, target, sample_hz=500, weights=None):
    """Masked completion is reconstructed before diagnostics are computed.

    Training-derived inverse feature scales are capped to avoid overweighting
    near-zero/noise-dominated bins. Caller saves the fitted weights.
    """
    losses = []
    for nfft in (32, 64, 128):
        nfft = min(nfft, target.shape[-1])
        window = torch.hann_window(nfft, device=prediction.device)
        # Unfold covers the full window instead of truncating it to its first FFT block.
        a = torch.fft.rfft(prediction.unfold(-1, nfft, max(1, nfft//2)) * window)
        b = torch.fft.rfft(target.unfold(-1, nfft, max(1, nfft//2)) * window)
        # Dimensionless normalized signals; log1p has no physical-voltage floor.
        losses.append((torch.log1p(a.abs().square()) - torch.log1p(b.abs().square())).square().mean())
    for width in (max(1, int(sample_hz*.032)), max(1, int(sample_hz*.128))):
        width = min(width, target.shape[-1])
        a = F.avg_pool1d(prediction.square(), width, stride=max(1,width//2))
        b = F.avg_pool1d(target.square(), width, stride=max(1,width//2))
        losses.append((a-b).square().mean())
    weights = weights if weights is not None else [1.] * len(losses)
    return sum(float(w)*loss for w, loss in zip(weights, losses)) / len(losses)
