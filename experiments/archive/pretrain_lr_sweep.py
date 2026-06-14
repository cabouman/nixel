"""
Pretraining LR sweep with decoupled theta/z LRs. Trains theta + per-image z from
scratch on a small multi-image set, comparing the single-LR baseline (frac=1.0,
lr=1e-3) against frac=0.5 (theta = 0.5 * z LR) at several z LRs. Reports the mean
render PSNR over the training images at a fixed step budget.

Figure -> runs/archive/.
"""

NUM_IMAGES      = 16
P               = 8
CHANNELS        = 8
ITERS_PER_STAGE = 800              # 5 stages (P=8) -> 4000 steps total
ARMS            = [(1.0, 1e-3), (0.5, 2e-3), (0.5, 5e-3), (0.5, 1e-2)]   # (theta_lr_frac, z_lr)
COORDS          = 65536
SEED            = 0

import glob, math, os, random
import numpy as np, torch
from PIL import Image
from linr import LinrDecoder, ArrayField, ProgressiveConfig, pixel_grid, get_device
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (for _paths)
from _paths import NATURAL_DIR, RUNS_DIR


def render(dec, z, a, N, dev):
    coords = pixel_grid(N, dev).view(-1, 2)
    with torch.no_grad():
        out = torch.cat([dec.decode(c, z, a) for c in torch.split(coords, 1 << 18)])
    return out.view(N, N).cpu()


def psnr_of(rec, gt):
    return -10 * math.log10(torch.mean((rec - gt) ** 2).item() + 1e-12)


def smooth(x, w=201):
    x = np.asarray(x, np.float64); w = min(w, len(x) // 2 * 2 + 1)
    return np.convolve(x, np.ones(w) / w, mode="valid"), (w - 1) // 2


def main():
    dev = get_device()
    paths = sorted(glob.glob(os.path.join(NATURAL_DIR, "*.png")))[:NUM_IMAGES]
    imgs = [torch.from_numpy(np.asarray(Image.open(p).convert("L"), np.float32) / 255.0) for p in paths]
    N = imgs[0].shape[0]; G = N // P
    fields = [ArrayField(im.to(dev)) for im in imgs]
    total = (P // 2 + 1) * ITERS_PER_STAGE
    print(f"{NUM_IMAGES} natural img {N}x{N} | P={P} C={CHANNELS} | {total} steps/arm "
          f"(~{total // NUM_IMAGES} updates per z)\n")

    curves, mean_psnr = {}, {}
    for frac, lr in ARMS:
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        dec = LinrDecoder(P, channels=CHANNELS, device=dev)
        cfg = ProgressiveConfig(iters_per_stage=ITERS_PER_STAGE, lr=lr, theta_lr_frac=frac,
                                coords_per_step=COORDS, grid=G, k0=0)
        rep, zs, a_s = dec.pretrain_progressive(fields, cfg)
        ps = [psnr_of(render(dec, zs[i], a_s[i], N, dev), imgs[i]) for i in range(NUM_IMAGES)]
        mean_psnr[(frac, lr)] = float(np.mean(ps))
        curves[(frac, lr)] = -10 * np.log10(np.asarray(rep.loss_history) + 1e-12)
        print(f"  frac={frac} z_lr={lr:.0e} (theta_lr={frac*lr:.0e}): mean render {mean_psnr[(frac,lr)]:.2f} dB "
              f"(min {min(ps):.1f}, max {max(ps):.1f})")

    best = max(mean_psnr, key=mean_psnr.get)
    print(f"\nbest: frac={best[0]} z_lr={best[1]:.0e} -> {mean_psnr[best]:.2f} dB mean render")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for (frac, lr), c in zip(ARMS, ["C0", "C1", "C2", "C3", "C4"]):
        y, off = smooth(curves[(frac, lr)])
        ax.plot(np.arange(len(y)) + off, y, color=c, lw=1.2,
                label=f"frac={frac} z_lr={lr:.0e} ({mean_psnr[(frac,lr)]:.1f} dB)")
    for k in range(1, P // 2 + 1):
        ax.axvline(k * ITERS_PER_STAGE, color="0.8", ls=":", lw=0.6)
    ax.set_xlabel("step"); ax.set_ylabel("PSNR proxy (dB, smoothed)")
    ax.set_title(f"pretraining LR sweep (theta/z decoupled) — {NUM_IMAGES} img, {total} steps")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    out_dir = os.path.join(RUNS_DIR, "archive"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "pretrain_lr_sweep.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
