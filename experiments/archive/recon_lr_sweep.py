"""
Reconstruction learning-rate sweep. With a pretrained decoder (default the 'many'
run), fit a held-out test image with adapt-theta at several base LRs and see which
converges fastest. Goal: speed up reconstruction.

Curves are the PSNR proxy (-10 log10 of the 65536-coord minibatch MSE); final
numbers are the on-grid render PSNR. Figure -> runs/archive/.
"""

DECODER_RUN = "many"          # runs/<DECODER_RUN>/decoder.linrd
DATABASE    = "phantom"       # natural | phantom
IMAGE       = 0               # phantom 0 = Shepp-Logan (held out: many has num_phantoms=0)
STEPS       = 2000            # z-fit iterations per arm
LRS         = [1e-3, 3e-3, 1e-2]   # base LR (z and theta share it)
WARMUP      = 200
COORDS      = 65536
SEED        = 0

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
    dpath = os.path.join(RUNS_DIR, DECODER_RUN, "decoder.linrd")
    if not os.path.exists(dpath):
        raise SystemExit(f"No decoder at {dpath} (run pretrain.py --exp {DECODER_RUN}).")
    path = sorted(glob.glob(os.path.join(DB_DIRS[DATABASE], "*.png")))[IMAGE]
    img = torch.from_numpy(np.asarray(Image.open(path).convert("L"), np.float32) / 255.0)
    N = img.shape[0]; gt = img
    print(f"decoder {DECODER_RUN} | {DATABASE} {os.path.basename(path)} {N}x{N} | {STEPS} steps/arm")

    curves, renders = {}, {}
    for lr in LRS:
        dec = LinrDecoder.load(dpath, device=dev)
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)   # identical z-init
        res = dec.reconstruct(img, ImageGrid(N, dev),
                              ReconConfig(steps=STEPS, lr=lr, coords_per_step=COORDS,
                                          grid=N // dec.P, adapt_theta=True, theta_lr=None,
                                          theta_warmup=WARMUP))
        renders[lr] = psnr_of(res.recon.render(N).cpu(), gt)
        curves[lr] = -10 * np.log10(np.asarray(res.loss_history) + 1e-12)
        print(f"  lr={lr:.0e}: final render {renders[lr]:.2f} dB")

    iters = [50, 100, 200, 500, 1000, 2000]
    print("\nPSNR proxy (trailing-25 mean) at iteration:")
    print("  lr     " + "".join(f"{k:>8}" for k in iters))
    for lr in LRS:
        p = curves[lr]
        print(f"  {lr:.0e}" + "".join(f"{p[max(0,k-25):k].mean():>8.2f}" for k in iters if k <= len(p)))

    import matplotlib.pyplot as plt
    cols = ["C0", "C1", "C2", "C3"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for lr, c in zip(LRS, cols):
        ax[0].plot(curves[lr], color=c, lw=1.0, label=f"lr={lr:.0e} (final {renders[lr]:.1f} dB)")
        ax[1].plot(curves[lr][:500], color=c, lw=1.0, label=f"lr={lr:.0e}")
    for a_ in ax:
        a_.axvline(WARMUP, color="0.6", ls=":", lw=0.8); a_.grid(alpha=0.3)
        a_.set_xlabel("z-fit iteration"); a_.set_ylabel("PSNR proxy (dB)")
    ax[0].set_title("convergence vs LR"); ax[0].legend(fontsize=8)
    ax[1].set_title("early convergence (first 500)"); ax[1].legend(fontsize=8)
    fig.suptitle(f"recon LR sweep — {DECODER_RUN} decoder, {DATABASE}[{IMAGE}]", fontsize=12)
    fig.tight_layout()
    out_dir = os.path.join(RUNS_DIR, "archive"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"recon_lr_sweep_{DATABASE}{IMAGE}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
