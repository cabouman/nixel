"""
LINR ("nixel") framework — implements docs/theory-v1.md, docs/theory-v2.md,
and docs/API_def.md.

Three objects: LinrDecoder (shared theta, B), Reconstruction (per-field z, a),
ForwardOperator (pluggable measurement model A). Three operations: pretrain,
reconstruct, render.

Conventions (theory-v1.md):
  * coordinates r = (x, y) in [-1, 1]^2, cell-centered.
  * nixel-unit coordinate u = (r+1)/2 * G - 0.5  (integer u = nixel centers).
  * Fourier features gamma(u): real 2-D DFT basis of the PxP patch (F = P^2),
    integer frequencies -> periodic per nixel.
  * f(r) = exp(a) * act(MLP(gamma(u), interp(z, u))).
  * coarse-to-fine (theory-v2): features with band max(|m|,|n|) > bandwidth are
    masked to zero; bandwidth defaults to M = P/2 (full bandwidth == v1).
"""

import abc
import dataclasses
import hashlib
import math
import os
import random
import time
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi
DEFAULT_DECODER = "decoder.linrd"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_cosine_lr(opt, base_lr, t, T):
    """Cosine decay base_lr -> 0 over T steps; lr = base at t=0, ~0 at t=T."""
    lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * min(t, T) / max(1, T)))
    for g in opt.param_groups:
        g["lr"] = lr


# --------------------------------------------------------------------------- #
# Coordinates and interpolation (cell-centered, consistent everywhere)
# --------------------------------------------------------------------------- #
def pixel_grid(N, device=None):
    """(N, N, 2) cell-centered coordinates in [-1,1]: pixel [i,j] -> (x,y) with
    x = (2j+1)/N - 1 (column), y = (2i+1)/N - 1 (row). No y-flip."""
    idx = (torch.arange(N, device=device, dtype=torch.float32) * 2 + 1) / N - 1
    gy, gx = torch.meshgrid(idx, idx, indexing="ij")
    return torch.stack([gx, gy], dim=-1)


def bilinear_interp(coords, grid):
    """Bilinear sample of grid (C, S, S) at coords (M, 2) in [-1,1]. Cell-centered
    (align_corners=False) with replicate (clamp) padding. Returns (M, C)."""
    C, H, W = grid.shape
    x, y = coords[:, 0], coords[:, 1]
    fx = (x + 1) * 0.5 * W - 0.5
    fy = (y + 1) * 0.5 * H - 0.5
    x0 = fx.floor().long().clamp(0, W - 1)
    y0 = fy.floor().long().clamp(0, H - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y1 = (y0 + 1).clamp(0, H - 1)
    wx = (fx - x0.to(fx.dtype)).clamp(0, 1).unsqueeze(-1)
    wy = (fy - y0.to(fy.dtype)).clamp(0, 1).unsqueeze(-1)
    c00, c01 = grid[:, y0, x0].t(), grid[:, y0, x1].t()
    c10, c11 = grid[:, y1, x0].t(), grid[:, y1, x1].t()
    top = c00 * (1 - wx) + c01 * wx
    bot = c10 * (1 - wx) + c11 * wx
    return top * (1 - wy) + bot * wy


# --------------------------------------------------------------------------- #
# Fourier basis: real 2-D DFT basis of the PxP nixel patch (theory-v1.md §4)
# --------------------------------------------------------------------------- #
def fourier_frequencies(P):
    """Integer frequency vectors for the complete real basis (F = P^2 functions).
    Returns (cos_freqs, sin_freqs), each (2, n) int tensors of (m, n) columns."""
    M = P // 2
    cos_f, sin_f = [], []
    # interior columns n = 1..M-1, all m = -M+1..M
    for n in range(1, M):
        for m in range(-M + 1, M + 1):
            cos_f.append((m, n))
            sin_f.append((m, n))
    # self-symmetric columns n in {0, M}, m = 0..M; drop sine at self-conjugate corners
    for n in (0, M):
        for m in range(0, M + 1):
            cos_f.append((m, n))
            if m not in (0, M):
                sin_f.append((m, n))
    cos_t = torch.tensor(cos_f, dtype=torch.float32).t()  # (2, n_cos)
    sin_t = torch.tensor(sin_f, dtype=torch.float32).t()  # (2, n_sin)
    return cos_t, sin_t


# --------------------------------------------------------------------------- #
# Forward operators (pluggable measurement model A)
# --------------------------------------------------------------------------- #
class ForwardOperator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, field: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        ...


class ImageGrid(ForwardOperator):
    """Identity imaging: sample the field on the N x N pixel grid -> (N, N)."""

    def __init__(self, N, device=None):
        self.N = N
        self.device = device or get_device()

    def __call__(self, field):
        coords = pixel_grid(self.N, self.device).view(-1, 2)
        out = torch.cat([field(c) for c in torch.split(coords, 1 << 18)])
        return out.view(self.N, self.N)


# --------------------------------------------------------------------------- #
# Fields (anything samplable at coordinates) — the pretraining data interface
# --------------------------------------------------------------------------- #
class ArrayField:
    """A stored N x N raster, bilinearly sampled at continuous coordinates."""

    def __init__(self, image):                      # image: (N, N) tensor in any range
        self.image = image
        self.N = image.shape[-1]

    def sample(self, coords):
        return bilinear_interp(coords, self.image.unsqueeze(0)).squeeze(-1)


def eval_ellipses(coords, ellipses):
    """Analytic value of a sum of ellipses at coords (...,2) in [-1,1].
    ellipses: list of (intensity, a, b, x0, y0, phi_deg)."""
    x, y = coords[..., 0], coords[..., 1]
    val = torch.zeros_like(x)
    for intensity, a, b, x0, y0, phi in ellipses:
        t = math.radians(phi)
        ct, st = math.cos(t), math.sin(t)
        xr = (x - x0) * ct + (y - y0) * st
        yr = -(x - x0) * st + (y - y0) * ct
        inside = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        val = val + intensity * inside.to(val.dtype)
    return val


def random_phantom(rng):
    """Random Shepp-Logan-like phantom: a body ellipse + signed structures."""
    ell = [(1.0, rng.uniform(0.55, 0.78), rng.uniform(0.72, 0.92),
            rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0, 180))]
    for _ in range(rng.randint(3, 9)):
        ell.append((rng.uniform(-0.6, 0.6),
                    rng.uniform(0.05, 0.35), rng.uniform(0.05, 0.35),
                    rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6),
                    rng.uniform(0, 180)))
    return ell


SHEPP_LOGAN = [   # the classic (modified / Toft) Shepp-Logan ellipses
    (1.0,  0.69,   0.92,   0.0,    0.0,     0.0),
    (-0.8, 0.6624, 0.874,  0.0,   -0.0184,  0.0),
    (-0.2, 0.11,   0.31,   0.22,   0.0,   -18.0),
    (-0.2, 0.16,   0.41,  -0.22,   0.0,    18.0),
    (0.1,  0.21,   0.25,   0.0,    0.35,    0.0),
    (0.1,  0.046,  0.046,  0.0,    0.1,     0.0),
    (0.1,  0.046,  0.046,  0.0,   -0.1,     0.0),
    (0.1,  0.046,  0.023, -0.08,  -0.605,   0.0),
    (0.1,  0.023,  0.023,  0.0,   -0.606,   0.0),
    (0.1,  0.023,  0.046,  0.06,  -0.605,   0.0),
]


def shepp_logan():
    """The classic (modified) Shepp-Logan phantom as an ellipse list, oriented for
    pixel_grid's y-down convention (y0 and phi flipped vs. the textbook y-up values)
    so it renders head-up."""
    return [(I, a, b, x0, -y0, -phi) for (I, a, b, x0, y0, phi) in SHEPP_LOGAN]


class EllipseField:
    """A continuous (analytic) ellipse phantom, sampled at any coordinates."""

    def __init__(self, ellipses):
        self.ellipses = ellipses

    def sample(self, coords):
        return eval_ellipses(coords, self.ellipses)


# --------------------------------------------------------------------------- #
# Config / report objects
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TrainConfig:
    epochs: int = 1
    iters_per_image: int = 2000
    lr: float = 1e-3
    coords_per_step: int = 4096
    grid: int = 32                       # G = N // P, the latent grid per image
    on_step: Optional[Callable] = None   # callback(step, loss)


@dataclasses.dataclass
class ReconConfig:
    steps: int = 1500
    lr: float = 5e-3
    coords_per_step: int = 65536
    grid: Optional[int] = None           # default N // P
    init_scale: float = 1.0
    regularizer: Optional[Callable] = None
    on_step: Optional[Callable] = None
    adapt_theta: bool = False            # also fine-tune theta (warm-started prior)
    theta_lr: Optional[float] = None     # theta LR while adapting; None -> lr (equal-rate, validated best)
    theta_warmup: int = 0                # z-only steps before unfreezing theta


@dataclasses.dataclass
class ProgressiveConfig:
    """Coarse-to-fine training (theory-v2 §2.6). A stage = iters_per_stage steps at
    a fixed bandwidth, with a fresh cosine LR; bands sweep K = k0 .. M one ring at
    a time. pretrain_joint() reuses the same fields for the equal-budget baseline."""
    iters_per_stage: int = 1500
    lr: float = 1e-3                     # latent (z, a) LR
    theta_lr_frac: float = 1.0           # theta LR = theta_lr_frac * lr (<1 -> theta gentler than z)
    coords_per_step: int = 65536
    grid: int = 128                      # G = N // P
    k0: int = 0                          # starting band
    on_step: Optional[Callable] = None   # callback(step, K, loss)


@dataclasses.dataclass
class TrainReport:
    loss_history: List[float]
    final_loss: float
    seconds: float


@dataclasses.dataclass
class ReconResult:
    recon: "Reconstruction"
    loss_history: List[float]
    final_loss: float


# --------------------------------------------------------------------------- #
# Decoder (shared) and Reconstruction (per field)
# --------------------------------------------------------------------------- #
class LinrDecoder(nn.Module):
    def __init__(self, pixels_per_nixel, channels=8, hidden=128, layers=4,
                 out_act="linear", device=None):
        super().__init__()
        assert pixels_per_nixel % 2 == 0, "P must be even"
        assert out_act in ("linear", "nonneg")
        self.P = pixels_per_nixel
        self.channels = channels
        self.out_act = out_act
        self.config = dict(pixels_per_nixel=pixels_per_nixel, channels=channels,
                           hidden=hidden, layers=layers, out_act=out_act)

        cos_t, sin_t = fourier_frequencies(self.P)
        self.register_buffer("Bcos", cos_t)
        self.register_buffer("Bsin", sin_t)
        self.num_features = cos_t.shape[1] + sin_t.shape[1]   # = P^2
        # L-infinity band index of each Fourier feature (theory-v2 §2.2):
        # K(b) = max(|m|,|n|), aligned with gamma = [cos..., sin...]. Used to mask
        # high-frequency bands for coarse-to-fine (progressive) training.
        self.register_buffer("feature_band",
                             torch.cat([cos_t.abs().amax(dim=0), sin_t.abs().amax(dim=0)]))
        self.M = self.P // 2          # maximum band index = full bandwidth
        self.bandwidth = self.M       # current active band (default: full = v1)

        net = [nn.Linear(self.num_features + channels, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            net += [nn.Linear(hidden, hidden), nn.ReLU()]
        net += [nn.Linear(hidden, 1)]
        self.mlp = nn.Sequential(*net)
        if device:
            self.to(device)

    # ----- core differentiable primitive -----
    def fourier(self, coords, G):
        u = (coords + 1) * 0.5 * G - 0.5            # nixel-unit coordinate
        return torch.cat([torch.cos(TWO_PI * u @ self.Bcos),
                          torch.sin(TWO_PI * u @ self.Bsin)], dim=-1)

    def decode(self, coords, z, a=0.0):
        G = z.shape[-1]
        feats = self.fourier(coords, G)
        if self.bandwidth < self.M:                  # progressive: zero high bands
            feats = feats * (self.feature_band <= self.bandwidth).to(feats.dtype)
        h = torch.cat([feats, bilinear_interp(coords, z)], dim=-1)
        out = self.mlp(h).squeeze(-1)
        if self.out_act == "nonneg":
            out = F.softplus(out)
        if not torch.is_tensor(a):
            a = out.new_tensor(float(a))
        return out * torch.exp(a)                   # exp(a) > 0 preserves nonneg

    # ----- coarse-to-fine bandwidth control (theory-v2 §2.3-2.4) -----
    def set_bandwidth(self, K):
        """Activate Fourier bands with max(|m|,|n|) <= K. K in [0, M]; K=M is full."""
        assert 0 <= K <= self.M, f"bandwidth must be in [0, {self.M}]"
        self.bandwidth = int(K)

    def init_progressive(self, K0=0):
        """Prepare coarse-to-fine training: zero the first-layer FEATURE columns so
        newly unmasked bands begin at zero (seamless warm start), and set band K0.
        Latent columns, biases, and deeper layers keep their random init."""
        with torch.no_grad():
            self.mlp[0].weight[:, :self.num_features].zero_()
        self.set_bandwidth(K0)

    @property
    def id(self):
        return _decoder_id(self.config, self.mlp.state_dict())

    # ----- operations -----
    def pretrain(self, dataset, cfg: TrainConfig) -> TrainReport:
        device = next(self.parameters()).device
        M, G, C = len(dataset), cfg.grid, self.channels
        zs = [nn.Parameter(0.01 * torch.randn(C, G, G, device=device)) for _ in dataset]
        a_s = [nn.Parameter(torch.zeros((), device=device)) for _ in dataset]

        opt = torch.optim.Adam(list(self.mlp.parameters()) + zs + a_s, lr=cfg.lr)
        total = max(1, cfg.epochs * M * cfg.iters_per_image)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)

        losses, step, t0 = [], 0, time.time()
        for _ in range(cfg.epochs):
            order = list(range(M))
            random.shuffle(order)
            for i in order:
                for _ in range(cfg.iters_per_image):
                    coords = torch.rand(cfg.coords_per_step, 2, device=device) * 2 - 1
                    with torch.no_grad():
                        target = dataset[i].sample(coords)
                    loss = F.mse_loss(self.decode(coords, zs[i], a_s[i]), target)
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                    losses.append(loss.item())
                    if cfg.on_step:
                        cfg.on_step(step, loss.item())
                    step += 1
        return TrainReport(losses, losses[-1], time.time() - t0)

    # ----- progressive (coarse-to-fine) training, theory-v2 §2.6 -----
    def _make_ckpt(self, zs, a_s, opt, step, losses):
        """Full, resumable training state (tensors moved to CPU for portability)."""
        return {
            "config": self.config,
            "theta": {k: v.detach().cpu() for k, v in self.mlp.state_dict().items()},
            "zs": [z.detach().cpu() for z in zs],
            "a_s": [a.detach().cpu() for a in a_s],
            "opt": opt.state_dict(),
            "global_step": step,
            "loss_history": list(losses),
            "rng_torch": torch.get_rng_state(),
            "rng_np": np.random.get_state(),
            "rng_py": random.getstate(),
        }

    def _fit_bands(self, dataset, stages, base_lr, coords_per_step, grid,
                   zero_init, on_step, theta_lr_frac=1.0,
                   checkpoint_every=0, checkpoint_fn=None, resume=None):
        """Schedule-based trainer shared by progressive and the joint baseline.
        `stages` is a list of (bandwidth_K, n_steps); each stage runs a fresh cosine
        LR from base_lr down to ~0. The latents (z, a) use base_lr; theta uses
        theta_lr_frac * base_lr (its own param group). Returns (TrainReport, zs, a_s).

        Resumable: if `resume` (a dict from `_make_ckpt`, loaded onto `device`) is given,
        continue from the saved global step with the saved weights/latents/optimizer/RNG.
        If `checkpoint_fn` is given it is called as checkpoint_fn(step, ckpt) every
        `checkpoint_every` steps and once at the end."""
        device = next(self.parameters()).device
        n_img, C = len(dataset), self.channels
        total = sum(n for _, n in stages)
        bounds, s = [], 0                              # global-step -> stage map
        for K, n in stages:
            bounds.append((s, s + n, K, n)); s += n

        if resume is not None:
            self.mlp.load_state_dict(resume["theta"])
            zs = [nn.Parameter(z.to(device)) for z in resume["zs"]]
            a_s = [nn.Parameter(a.to(device)) for a in resume["a_s"]]
            losses, start = list(resume["loss_history"]), resume["global_step"]
        else:
            zs = [nn.Parameter(0.01 * torch.randn(C, grid, grid, device=device)) for _ in dataset]
            a_s = [nn.Parameter(torch.zeros((), device=device)) for _ in dataset]
            if zero_init:
                self.init_progressive(stages[0][0])   # zero feature cols; set first band
            else:
                self.set_bandwidth(self.M)            # full bandwidth (joint baseline)
            losses, start = [], 0

        # two param groups: theta gets a (usually smaller) fraction of the latent LR
        group_base = [base_lr * theta_lr_frac, base_lr]
        opt = torch.optim.Adam([{"params": list(self.mlp.parameters()), "lr": group_base[0]},
                                {"params": zs + a_s, "lr": group_base[1]}])
        if resume is not None:
            opt.load_state_dict(resume["opt"])
            torch.set_rng_state(resume["rng_torch"].to("cpu"))
            np.random.set_state(resume["rng_np"]); random.setstate(resume["rng_py"])

        t0 = time.time()
        for step in range(start, total):
            bs, _, K, n = next(b for b in bounds if b[0] <= step < b[1])
            self.set_bandwidth(K)
            factor = 0.5 * (1.0 + math.cos(math.pi * min(step - bs, n) / max(1, n)))  # cosine restart/stage
            for g, gb in zip(opt.param_groups, group_base):
                g["lr"] = gb * factor
            i = random.randrange(n_img)
            coords = torch.rand(coords_per_step, 2, device=device) * 2 - 1
            with torch.no_grad():
                target = dataset[i].sample(coords)
            loss = F.mse_loss(self.decode(coords, zs[i], a_s[i]), target)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if on_step:
                on_step(step, K, loss.item())
            if checkpoint_fn and checkpoint_every and (step + 1) % checkpoint_every == 0:
                checkpoint_fn(step + 1, self._make_ckpt(zs, a_s, opt, step + 1, losses))
        self.set_bandwidth(self.M)                # leave decoder at full bandwidth
        if checkpoint_fn:
            checkpoint_fn(total, self._make_ckpt(zs, a_s, opt, total, losses))
        return TrainReport(losses, losses[-1], time.time() - t0), zs, a_s

    def pretrain_progressive(self, dataset, cfg: ProgressiveConfig, **kw):
        """Coarse-to-fine: sweep K = k0..M one ring per stage, iters_per_stage steps
        each, warm-started. Returns (TrainReport, zs, a_s). `kw` (checkpoint_every,
        checkpoint_fn, resume) is forwarded to the trainer for resumable runs."""
        stages = [(K, cfg.iters_per_stage) for K in range(cfg.k0, self.M + 1)]
        return self._fit_bands(dataset, stages, cfg.lr, cfg.coords_per_step,
                               cfg.grid, zero_init=True, on_step=cfg.on_step,
                               theta_lr_frac=cfg.theta_lr_frac, **kw)

    def pretrain_joint(self, dataset, total_steps, cfg: ProgressiveConfig, **kw):
        """Equal-budget baseline: full bandwidth from the start, standard init,
        `total_steps` steps, single cosine. Returns (TrainReport, zs, a_s). `kw` is
        forwarded to the trainer (checkpoint_every, checkpoint_fn, resume)."""
        return self._fit_bands(dataset, [(self.M, total_steps)], cfg.lr,
                               cfg.coords_per_step, cfg.grid,
                               zero_init=False, on_step=cfg.on_step,
                               theta_lr_frac=cfg.theta_lr_frac, **kw)

    def reconstruct(self, y, A: ForwardOperator, cfg: ReconConfig) -> ReconResult:
        # Image-fitting case (A = ImageGrid): fit a fresh (z, a) against y. By default
        # theta is frozen (z-only reconstruction). If cfg.adapt_theta, also fine-tune
        # the warm-started theta -- but keep it frozen for the first cfg.theta_warmup
        # steps (let z settle so a cold z does not corrupt the prior) and then adapt it
        # at the gentler cfg.theta_lr. z/a follow a cosine over all steps; theta follows
        # a fresh cosine over the post-warmup window.
        device = next(self.parameters()).device
        N = A.N
        G = cfg.grid or (N // self.P)
        recon = Reconstruction(self, G, init_scale=cfg.init_scale)
        target_field = ArrayField(y.to(device))

        theta = list(self.mlp.parameters())
        adapt = bool(cfg.adapt_theta)
        warm = cfg.theta_warmup if adapt else cfg.steps      # theta frozen for `warm` steps
        tlr = cfg.theta_lr if cfg.theta_lr is not None else cfg.lr

        groups = [{"params": list(recon.parameters()), "lr": cfg.lr}]
        if adapt:
            groups.append({"params": theta, "lr": tlr})
        opt = torch.optim.Adam(groups)

        def theta_trainable(flag):
            for p in theta:
                p.requires_grad_(flag)
        theta_trainable(False)               # frozen during warmup (and always, if not adapting)

        losses = []
        for step in range(cfg.steps):
            if adapt and step == warm:
                theta_trainable(True)        # unfreeze theta after the z warmup
            opt.param_groups[0]["lr"] = cfg.lr * 0.5 * (1 + math.cos(math.pi * step / cfg.steps))
            if adapt:
                if step >= warm:
                    tt, TT = step - warm, max(1, cfg.steps - warm)
                    opt.param_groups[1]["lr"] = tlr * 0.5 * (1 + math.cos(math.pi * tt / TT))
                else:
                    opt.param_groups[1]["lr"] = 0.0
            coords = torch.rand(cfg.coords_per_step, 2, device=device) * 2 - 1
            with torch.no_grad():
                target = target_field.sample(coords)
            loss = F.mse_loss(recon.field(coords), target)
            if cfg.regularizer:
                loss = loss + cfg.regularizer(recon.z)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if cfg.on_step:
                cfg.on_step(step, loss.item())
        theta_trainable(True)                # leave decoder params trainable afterwards
        return ReconResult(recon, losses, losses[-1])

    # ----- persistence (.linrd); B is regenerated from P, so only config+theta saved -----
    def save(self, path=DEFAULT_DECODER):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"id": self.id, "config": self.config,
                    "theta": self.mlp.state_dict()}, path)
        return self.id

    @classmethod
    def load(cls, path=DEFAULT_DECODER, device=None):
        d = torch.load(path, map_location="cpu")
        m = cls(**d["config"], device=device)
        m.mlp.load_state_dict(d["theta"])
        return m


def _decoder_id(config, theta_state):
    h = hashlib.sha256()
    h.update(repr(sorted(config.items())).encode())
    for k in sorted(theta_state):
        h.update(k.encode())
        h.update(theta_state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class Reconstruction:
    """One field's representation: learnable latent z (C,G,G) and log-scale a,
    bound to a decoder. `field` is the differentiable image model."""

    def __init__(self, decoder: LinrDecoder, grid, init_scale=1.0):
        device = next(decoder.parameters()).device
        self.decoder = decoder
        self.z = nn.Parameter(0.01 * torch.randn(decoder.channels, grid, grid, device=device))
        self.a = nn.Parameter(torch.tensor(math.log(init_scale), device=device))

    def field(self, coords):
        return self.decoder.decode(coords, self.z, self.a)

    def parameters(self):
        return [self.z, self.a]

    def set_scale(self, s):
        with torch.no_grad():
            self.a.fill_(math.log(s))

    def get_scale(self):
        return float(self.a.detach().exp())

    @torch.no_grad()
    def render(self, N):
        device = self.z.device
        coords = pixel_grid(N, device).view(-1, 2)
        out = torch.cat([self.field(c) for c in torch.split(coords, 1 << 18)])
        return out.view(N, N)

    def save(self, path, metadata=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"z": self.z.detach().cpu(), "a": self.a.detach().cpu(),
                    "decoder_id": self.decoder.id, "metadata": metadata or {}}, path)
        return path

    @classmethod
    def load(cls, path, decoder: LinrDecoder):
        d = torch.load(path, map_location="cpu")
        if d["decoder_id"] != decoder.id:
            raise ValueError(f"decoder mismatch: {path} expects {d['decoder_id']}, "
                             f"decoder is {decoder.id}")
        recon = cls(decoder, grid=d["z"].shape[-1])
        with torch.no_grad():
            recon.z.copy_(d["z"]); recon.a.copy_(d["a"])
        return recon, d.get("metadata", {})
