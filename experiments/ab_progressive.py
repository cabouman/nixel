"""
A/B: progressive (coarse-to-fine) vs joint training of the SAME final model,
at an EQUAL iteration budget (theory-v2 §2.6). Single image, fit theta + z.

Edit the parameters below and re-run. Both arms are re-seeded identically, so they
share the same latent init, deeper-layer init, and coordinate stream — the only
difference is the training strategy.
"""

# ============================ PARAMETERS ============================
P               = 8       # pixels per nixel; bands K = 0..M with M = P/2
CHANNELS        = 8       # C
IMAGE           = 0       # which nat_<k>.png (sorted index)
ITERS_PER_STAGE = 2000    # T: steps per band-stage (progressive). Joint gets (M+1)*T total.
LR              = 1e-3    # base learning rate (cosine restart each stage)
COORDS          = 32768   # random pixels per step
K0              = 0       # starting band for progressive
SEED            = 0
# ====================================================================

import glob, math, os, random
import numpy as np, torch
from PIL import Image
from linr import LinrDecoder, ArrayField, ProgressiveConfig, pixel_grid, get_device


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def render(dec, z, a, N, dev):
    coords = pixel_grid(N, dev).view(-1, 2)
    with torch.no_grad():
        out = torch.cat([dec.decode(c, z, a) for c in torch.split(coords, 1 << 18)])
    return out.view(N, N).cpu()


def psnr_of(rec, gt):
    return -10 * math.log10(torch.mean((rec - gt) ** 2).item() + 1e-12)


def smooth(x, w=101):
    x = np.asarray(x, np.float64)
    w = min(w, len(x) // 2 * 2 + 1)
    return np.convolve(x, np.ones(w) / w, mode="valid"), (w - 1) // 2


def main():
    dev = get_device()
    path = sorted(glob.glob("img_data/natural/nat_*.png"))[IMAGE]
    img = torch.from_numpy(np.asarray(Image.open(path).convert("L"), np.float32) / 255.0)
    N = img.shape[0]; G = N // P; M = P // 2
    gt = img
    dataset = [ArrayField(img.to(dev))]
    n_stages = M - K0 + 1
    total = n_stages * ITERS_PER_STAGE
    print(f"{os.path.basename(path)}  N={N}  P={P} (G={G})  C={CHANNELS}  "
          f"CR={P*P/CHANNELS:.1f} | bands {K0}..{M} ({n_stages} stages x {ITERS_PER_STAGE}) "
          f"= {total} steps each arm")

    cfg = ProgressiveConfig(iters_per_stage=ITERS_PER_STAGE, lr=LR,
                            coords_per_step=COORDS, grid=G, k0=K0,
                            on_step=lambda s, K, l: (s % 1000 == 0) and
                            print(f"  step {s:5d}  band {K}  mse {l:.3e}"))

    print("\n[progressive]")
    seed_all(SEED)
    decP = LinrDecoder(P, channels=CHANNELS, device=dev)
    repP, zsP, asP = decP.pretrain_progressive(dataset, cfg)
    recP = render(decP, zsP[0], asP[0], N, dev)
    psP = psnr_of(recP, gt)
    print(f"  done {repP.seconds:.1f}s  final full-image PSNR {psP:.2f} dB")

    print("\n[joint]")
    seed_all(SEED)
    decJ = LinrDecoder(P, channels=CHANNELS, device=dev)
    repJ, zsJ, asJ = decJ.pretrain_joint(dataset, total, cfg)
    recJ = render(decJ, zsJ[0], asJ[0], N, dev)
    psJ = psnr_of(recJ, gt)
    print(f"  done {repJ.seconds:.1f}s  final full-image PSNR {psJ:.2f} dB")

    print(f"\nFINAL:  progressive {psP:.2f} dB   vs   joint {psJ:.2f} dB   "
          f"(Δ = {psP - psJ:+.2f} dB)")

    # ----- figure: GT | progressive | joint | quality-vs-iterations -----
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    ax[0].imshow(gt, cmap="gray", vmin=0, vmax=1); ax[0].set_title("ground truth")
    ax[1].imshow(recP, cmap="gray", vmin=0, vmax=1); ax[1].set_title(f"progressive  {psP:.1f} dB")
    ax[2].imshow(recJ, cmap="gray", vmin=0, vmax=1); ax[2].set_title(f"joint  {psJ:.1f} dB")
    for a_ in ax[:3]:
        a_.axis("off")

    pp = -10 * np.log10(np.asarray(repP.loss_history) + 1e-12)
    pj = -10 * np.log10(np.asarray(repJ.loss_history) + 1e-12)
    yp, off = smooth(pp); yj, _ = smooth(pj)
    xs = np.arange(len(yp)) + off
    ax[3].plot(xs, yp, label="progressive", color="C0")
    ax[3].plot(np.arange(len(yj)) + off, yj, label="joint", color="C3")
    for k in range(1, n_stages):       # progressive stage boundaries (warm restarts)
        ax[3].axvline(k * ITERS_PER_STAGE, color="C0", ls=":", lw=0.6, alpha=0.5)
    ax[3].set_xlabel("cumulative iterations"); ax[3].set_ylabel("PSNR proxy (dB)")
    ax[3].set_title("quality vs iterations"); ax[3].grid(alpha=0.3); ax[3].legend(fontsize=8)

    fig.suptitle(f"Progressive vs joint  —  P={P} (G={G}), C={CHANNELS}, "
                 f"{n_stages}x{ITERS_PER_STAGE} steps", fontsize=12)
    fig.tight_layout(); os.makedirs("output", exist_ok=True)
    fig.savefig("output/ab_progressive.png", dpi=120, bbox_inches="tight")
    print("Saved output/ab_progressive.png")


if __name__ == "__main__":
    main()
