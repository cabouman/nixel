"""
Reconstruction LR sweep optimized for a 1000-step budget. Grid over z LR (recon_lr)
x theta fraction (theta_lr = frac * z_lr), warmup=0. Reports the FINAL render PSNR
at 1000 steps (the metric to optimize now) plus convergence curves.

Figure -> runs/archive/.
"""

MODEL    = "nat_img"          # runs/<MODEL>/decoder.linrd
DATABASE = "phantom"         # natural | phantom
IMAGE    = 0                 # phantom 0 = Shepp-Logan (held out)
STEPS    = 1000
WARMUP   = 0
Z_LRS    = [1e-2, 2e-2, 3e-2, 5e-2]
FRACS    = [1.0, 0.3]
COORDS   = 65536
SEED     = 0

import glob, math, os, random
import numpy as np, torch
from PIL import Image
from linr import LinrDecoder, ImageGrid, ReconConfig, get_device
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (for _paths)
from _paths import NATURAL_DIR, PHANTOM_DIR, RUNS_DIR

DB_DIRS = {"natural": NATURAL_DIR, "phantom": PHANTOM_DIR}


def psnr_of(rec, gt):
    return -10 * math.log10(torch.mean((rec - gt) ** 2).item() + 1e-12)


def main():
    dev = get_device()
    dpath = os.path.join(RUNS_DIR, MODEL, "decoder.linrd")
    if not os.path.exists(dpath):
        raise SystemExit(f"No decoder at {dpath}")
    path = sorted(glob.glob(os.path.join(DB_DIRS[DATABASE], "*.png")))[IMAGE]
    img = torch.from_numpy(np.asarray(Image.open(path).convert("L"), np.float32) / 255.0)
    N = img.shape[0]; gt = img
    print(f"model {MODEL} | {DATABASE} {os.path.basename(path)} {N}x{N} | "
          f"{STEPS} steps, warmup={WARMUP}\n")

    render, curve = {}, {}
    for z_lr in Z_LRS:
        for frac in FRACS:
            dec = LinrDecoder.load(dpath, device=dev)
            torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
            res = dec.reconstruct(img, ImageGrid(N, dev),
                                  ReconConfig(steps=STEPS, lr=z_lr, coords_per_step=COORDS,
                                              grid=N // dec.P, adapt_theta=True,
                                              theta_lr=frac * z_lr, theta_warmup=WARMUP))
            render[(z_lr, frac)] = psnr_of(res.recon.render(N).cpu(), gt)
            curve[(z_lr, frac)] = -10 * np.log10(np.asarray(res.loss_history) + 1e-12)

    print("FINAL render PSNR (dB) at 1000 steps:")
    print("  z_lr \\ frac " + "".join(f"{f:>10}" for f in FRACS))
    for z_lr in Z_LRS:
        print(f"  {z_lr:.0e}      " + "".join(f"{render[(z_lr,f)]:>10.2f}" for f in FRACS))
    best = max(render, key=render.get)
    print(f"\nbest: z_lr={best[0]:.0e}, frac={best[1]} -> {render[best]:.2f} dB")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for (z_lr, frac), c in render.items():
        lbl = f"z={z_lr:.0e} frac={frac} ({render[(z_lr,frac)]:.1f} dB)"
        style = "-" if frac == 1.0 else "--"
        ax.plot(curve[(z_lr, frac)], lw=1.0, ls=style, label=lbl)
    ax.set_xlabel("z-fit iteration"); ax.set_ylabel("PSNR proxy (dB)")
    ax.set_title(f"recon sweep @ {STEPS} steps — {MODEL}, {DATABASE}[{IMAGE}], warmup=0")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    out_dir = os.path.join(RUNS_DIR, "archive"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"recon_sweep_1k_{DATABASE}{IMAGE}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
