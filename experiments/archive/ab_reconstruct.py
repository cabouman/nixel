"""
A/B at RECONSTRUCTION time: with a TRAINED, FROZEN decoder (theta fixed), compare
two ways of fitting the latent z -- same image, same z-init, same total budget,
same LR/COORDS. The ONLY difference is the band schedule:

  JOINT : z fit at FULL bandwidth the whole time (what reconstruct.py does now).
  PROG  : z fit COARSE-TO-FINE -- bands K = 0..M added one ring at a time, with a
          fresh cosine-LR restart per band.

Result (nat_0, P=8 C=8, 2000 steps): a tie -- JOINT 24.98 dB vs PROG 25.05 dB
final, and JOINT is faster early. Fitting z against a fixed, well-conditioned theta
is near-convex, so the band schedule buys nothing: reconstruction stays joint.

Plots full-image PSNR (proxy = -10 log10 of the 65536-coord minibatch MSE, a tight
estimate) vs z-fit iteration. Run pretrain.py first to produce the decoder.
"""

# ============================ PARAMETERS ============================
DECODER     = "decoder.linrd"   # in runs/single/ (run: pretrain.py --exp single)
IMAGE       = 0          # nat_<k>.png the decoder was trained on
REFIT_STEPS = 2000       # total z-fit iterations PER ARM
LR          = 1e-3       # base LR (cosine; per-stage restart for PROG)
COORDS      = 65536      # random pixels per step
SEED        = 1          # z-init seed (shared by both arms)
# ====================================================================

import glob, math, os, random
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from linr import (LinrDecoder, ArrayField, pixel_grid, get_device, _set_cosine_lr)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (for _paths)
from _paths import NATURAL_DIR, RUNS_DIR
OUTPUT_DIR = os.path.join(RUNS_DIR, "archive")          # archive figures live under runs/
MODELS_DIR = os.path.join(RUNS_DIR, "single")           # archive scripts load runs/single/decoder.linrd


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def render(dec, z, a, N, dev):
    coords = pixel_grid(N, dev).view(-1, 2)
    with torch.no_grad():
        out = torch.cat([dec.decode(c, z, a) for c in torch.split(coords, 1 << 18)])
    return out.view(N, N).cpu()


def psnr_of(rec, gt):
    return -10 * math.log10(torch.mean((rec - gt) ** 2).item() + 1e-12)


def refit(dec, field, G, dev, stages):
    """Fit a FRESH z (and scale a) against `field`, theta frozen, following the band
    schedule `stages` = [(K, n_steps), ...]. Returns (z, a, per-step loss list)."""
    for p in dec.mlp.parameters():
        p.requires_grad_(False)
    seed_all(SEED)                                   # identical z-init across arms
    z = nn.Parameter(0.01 * torch.randn(dec.channels, G, G, device=dev))
    a = nn.Parameter(torch.zeros((), device=dev))
    opt = torch.optim.Adam([z, a], lr=LR)
    losses = []
    for K, n in stages:
        dec.set_bandwidth(K)
        for t in range(n):
            _set_cosine_lr(opt, LR, t, n)
            coords = torch.rand(COORDS, 2, device=dev) * 2 - 1
            with torch.no_grad():
                target = field.sample(coords)
            loss = F.mse_loss(dec.decode(coords, z, a), target)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
    dec.set_bandwidth(dec.M)
    for p in dec.mlp.parameters():
        p.requires_grad_(True)
    return z.detach(), a.detach(), losses


def milestones(prox, label):
    print(f"  [{label}] PSNR proxy (trailing 25-step mean):")
    for k in [25, 50, 100, 200, 500, 1000, len(prox)]:
        if k <= len(prox):
            print(f"      iter {k:5d}:  {prox[max(0, k-25):k].mean():.2f} dB")


def main():
    dev = get_device()
    dpath = os.path.join(MODELS_DIR, DECODER)
    if not os.path.exists(dpath):
        raise SystemExit(f"Decoder not found: {dpath}\n(run pretrain.py first).")
    dec = LinrDecoder.load(dpath, device=dev)

    path = sorted(glob.glob(os.path.join(NATURAL_DIR, "nat_*.png")))[IMAGE]
    img = torch.from_numpy(np.asarray(Image.open(path).convert("L"), np.float32) / 255.0)
    N = img.shape[0]; G = N // dec.P; M = dec.M; gt = img
    field = ArrayField(img.to(dev))
    print(f"decoder {DECODER} (P={dec.P}, C={dec.channels}) | {os.path.basename(path)} "
          f"{N}x{N} (G={G}) | {REFIT_STEPS} steps/arm")

    # ---- JOINT: full bandwidth throughout ----
    print("\n[JOINT]  full bandwidth")
    zJ, aJ, lossJ = refit(dec, field, G, dev, [(M, REFIT_STEPS)])
    recJ = render(dec, zJ, aJ, N, dev); psJ = psnr_of(recJ, gt)

    # ---- PROG: bands 0..M, equal steps per band ----
    n_bands = M + 1
    per = REFIT_STEPS // n_bands
    stagesP = [(K, per) for K in range(M)] + [(M, REFIT_STEPS - per * M)]
    print(f"[PROG]   bands 0..{M}  ({stagesP})")
    zP, aP, lossP = refit(dec, field, G, dev, stagesP)
    recP = render(dec, zP, aP, N, dev); psP = psnr_of(recP, gt)

    proxJ = -10 * np.log10(np.asarray(lossJ) + 1e-12)
    proxP = -10 * np.log10(np.asarray(lossP) + 1e-12)
    print()
    milestones(proxJ, "JOINT"); milestones(proxP, "PROG")
    print(f"\nFINAL (full-image render):  JOINT {psJ:.2f} dB   PROG {psP:.2f} dB   "
          f"(Δ = {psP - psJ:+.2f} dB)")

    # ---- figure: GT | joint | prog | convergence ----
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    for a_, im, t in [(ax[0], gt, "ground truth"),
                      (ax[1], recJ, f"joint  {psJ:.1f} dB"),
                      (ax[2], recP, f"progressive  {psP:.1f} dB")]:
        a_.imshow(im, cmap="gray", vmin=0, vmax=1); a_.set_title(t); a_.axis("off")
    ax[3].plot(proxJ, color="C3", lw=1.0, label="joint")
    ax[3].plot(proxP, color="C0", lw=1.0, label="progressive")
    for K in range(1, n_bands):                       # progressive band boundaries
        ax[3].axvline(K * per, color="C0", ls=":", lw=0.6, alpha=0.5)
    ax[3].set_xlabel("z-fit iteration"); ax[3].set_ylabel("PSNR proxy (dB)")
    ax[3].set_title("reconstruction convergence"); ax[3].grid(alpha=0.3); ax[3].legend(fontsize=9)
    fig.suptitle(f"z-refit: joint vs progressive  —  {DECODER}, nat{IMAGE}, "
                 f"{REFIT_STEPS} steps", fontsize=12)
    fig.tight_layout(); os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"ab_reconstruct_P{dec.P}_C{dec.channels}_im{IMAGE}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
