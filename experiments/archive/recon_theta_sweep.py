"""
Reconstruction theta-LR / warmup sweep. z LR is fixed high (1e-2); we vary theta's
LR independently and toggle the z-only warmup, to test: now that z-LR is high, is
theta-LR (currently tied to it) too high, and is the warmup still needed?

Curves = PSNR proxy; final = render PSNR. Figure -> runs/archive/.
"""

MODEL    = "nat_img"          # runs/<MODEL>/decoder.linrd
DATABASE = "phantom"         # natural | phantom
IMAGE    = 0                 # phantom 0 = Shepp-Logan (held out)
Z_LR     = 1e-2              # fixed z (and a) LR
STEPS    = 2000
ARMS     = [(1e-2, 200), (1e-2, 0), (3e-3, 0), (1e-3, 0)]   # (theta_lr, warmup)
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
    print(f"model {MODEL} | {DATABASE} {os.path.basename(path)} {N}x{N} | z_lr={Z_LR:.0e} | {STEPS} steps/arm")

    curves, renders, labels = {}, {}, {}
    for tlr, warm in ARMS:
        key = (tlr, warm)
        dec = LinrDecoder.load(dpath, device=dev)
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)   # identical z-init
        res = dec.reconstruct(img, ImageGrid(N, dev),
                              ReconConfig(steps=STEPS, lr=Z_LR, coords_per_step=COORDS,
                                          grid=N // dec.P, adapt_theta=True,
                                          theta_lr=tlr, theta_warmup=warm))
        renders[key] = psnr_of(res.recon.render(N).cpu(), gt)
        curves[key] = -10 * np.log10(np.asarray(res.loss_history) + 1e-12)
        labels[key] = f"theta_lr={tlr:.0e} warmup={warm}"
        print(f"  {labels[key]:28s} final render {renders[key]:.2f} dB")

    iters = [25, 50, 100, 200, 500, 1000, 2000]
    print("\nPSNR proxy (trailing-25 mean) at iteration:")
    print("  " + " " * 26 + "".join(f"{k:>8}" for k in iters))
    for key in [(t, w) for t, w in ARMS]:
        p = curves[key]
        print(f"  {labels[key]:26s}" + "".join(f"{p[max(0,k-25):k].mean():>8.2f}" for k in iters if k <= len(p)))

    import matplotlib.pyplot as plt
    cols = ["C3", "C0", "C1", "C2", "C4"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for (key, c) in zip([(t, w) for t, w in ARMS], cols):
        lbl = f"{labels[key]} ({renders[key]:.1f} dB)"
        ax[0].plot(curves[key], color=c, lw=1.0, label=lbl)
        ax[1].plot(curves[key][:500], color=c, lw=1.0, label=labels[key])
    for a_ in ax:
        a_.grid(alpha=0.3); a_.set_xlabel("z-fit iteration"); a_.set_ylabel("PSNR proxy (dB)")
    ax[0].set_title("convergence"); ax[0].legend(fontsize=8)
    ax[1].set_title("early convergence (first 500)"); ax[1].legend(fontsize=8)
    fig.suptitle(f"recon theta-LR / warmup sweep — {MODEL}, {DATABASE}[{IMAGE}], z_lr={Z_LR:.0e}", fontsize=12)
    fig.tight_layout()
    out_dir = os.path.join(RUNS_DIR, "archive"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"recon_theta_sweep_{DATABASE}{IMAGE}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
