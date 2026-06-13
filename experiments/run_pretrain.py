"""
Pretrain a decoder with progressive (coarse-to-fine) training on NUM_IMAGES
images, then reconstruct image #0 with the decoder frozen (a fresh z-fit), and
compare to ground truth.

Run from the repo root:  python experiments/run_pretrain.py
"""

# ============================ PARAMETERS ============================
P               = 8       # pixels per nixel; nixel grid G = N/P
CHANNELS        = 8       # C
NUM_IMAGES      = 1       # images from img_data/natural to pretrain on
ITERS_PER_STAGE = 2000    # progressive steps per band-stage
LR              = 1e-3
COORDS          = 32768   # coords per pretraining step
RECON_STEPS     = 1500    # iterations to reconstruct image #0 (decoder frozen)
K0              = 0       # starting band
SEED            = 0
# ====================================================================

import glob, math, os, random
import numpy as np, torch
from PIL import Image
from linr import (LinrDecoder, ArrayField, ImageGrid, ProgressiveConfig,
                  ReconConfig, get_device)
from _paths import NATURAL_DIR, OUTPUT_DIR


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dev = get_device()
    paths = sorted(glob.glob(os.path.join(NATURAL_DIR, "nat_*.png")))[:NUM_IMAGES]
    if not paths:
        raise SystemExit("No images in img_data/natural -- run build_databases.py first.")
    imgs = [torch.from_numpy(np.asarray(Image.open(p).convert("L"), np.float32) / 255.0)
            for p in paths]
    N = imgs[0].shape[0]; G = N // P; M = P // 2
    n_stages = M - K0 + 1
    print(f"{len(imgs)} image(s) {N}x{N} | P={P} (G={G}) C={CHANNELS} | "
          f"progressive pretrain: bands {K0}..{M} ({n_stages}x{ITERS_PER_STAGE} steps)")

    dec = LinrDecoder(P, channels=CHANNELS, device=dev)
    dataset = [ArrayField(im.to(dev)) for im in imgs]
    cfg = ProgressiveConfig(iters_per_stage=ITERS_PER_STAGE, lr=LR, coords_per_step=COORDS,
                            grid=G, k0=K0,
                            on_step=lambda s, K, l: (s % 1000 == 0) and
                            print(f"  pretrain step {s:5d}  band {K}  mse {l:.3e}"))
    rep, _, _ = dec.pretrain_progressive(dataset, cfg)
    print(f"  pretrain done ({rep.seconds:.1f}s, final mse {rep.final_loss:.3e})")

    print("Reconstructing image #0 (fresh latent, frozen decoder) ...")
    res = dec.reconstruct(imgs[0], ImageGrid(N, dev), ReconConfig(steps=RECON_STEPS, grid=G))
    fhat = res.recon.render(N).cpu(); gt = imgs[0]
    psnr = -10 * math.log10(torch.mean((fhat - gt) ** 2).item() + 1e-12)
    nrmse = 100 * torch.linalg.norm(fhat - gt) / torch.linalg.norm(gt)
    print(f"  reconstruction of {os.path.basename(paths[0])}: PSNR {psnr:.2f} dB, "
          f"NRMSE {nrmse:.2f}%")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14.4, 4.6))
    ax[0].imshow(gt, cmap="gray", vmin=0, vmax=1); ax[0].set_title("ground truth")
    ax[1].imshow(fhat, cmap="gray", vmin=0, vmax=1); ax[1].set_title(f"reconstruction  {psnr:.1f} dB")
    d = (fhat - gt).numpy(); dl = max(abs(d.min()), abs(d.max())) + 1e-9
    im = ax[2].imshow(d, cmap="seismic", vmin=-dl, vmax=dl)
    ax[2].set_title("difference"); fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a_ in ax:
        a_.axis("off")
    fig.suptitle(f"Progressive pretrain + reconstruct  P={P} (G={G}), C={CHANNELS}, "
                 f"{NUM_IMAGES} img")
    fig.tight_layout(); os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = f"P{P}_C{CHANNELS}_n{NUM_IMAGES}_T{ITERS_PER_STAGE}"
    out = os.path.join(OUTPUT_DIR, f"run_pretrain_{tag}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
