"""
Fit ONE image directly with progressive (coarse-to-fine) training and render the
trained (theta, z). The single-image capacity probe -- no pretrained model.

Run from the repo root:  python experiments/fit_one.py
"""

# ============================ PARAMETERS ============================
P               = 8       # pixels per nixel; nixel grid G = N/P, bands K = 0..P/2
CHANNELS        = 8       # C  ->  CR = P*P / C
IMAGE           = 0       # which nat_<k>.png (sorted index)
ITERS_PER_STAGE = 2000    # steps per band-stage (bands 0..M => (M+1) stages)
LR              = 1e-3
COORDS          = 65536   # random pixels sampled per step
K0              = 0       # starting band
SEED            = 0
# ====================================================================

import glob, math, os, random
import numpy as np, torch
from PIL import Image
from linr import LinrDecoder, ArrayField, ProgressiveConfig, pixel_grid, get_device
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (for _paths)
from _paths import NATURAL_DIR, RUNS_DIR
OUTPUT_DIR = os.path.join(RUNS_DIR, "archive")          # archive figures live under runs/


def render(dec, z, a, N, dev):
    coords = pixel_grid(N, dev).view(-1, 2)
    with torch.no_grad():
        out = torch.cat([dec.decode(c, z, a) for c in torch.split(coords, 1 << 18)])
    return out.view(N, N).cpu()


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dev = get_device()
    path = sorted(glob.glob(os.path.join(NATURAL_DIR, "nat_*.png")))[IMAGE]
    img = torch.from_numpy(np.asarray(Image.open(path).convert("L"), np.float32) / 255.0)
    N = img.shape[0]; G = N // P; M = P // 2
    n_stages = M - K0 + 1
    print(f"{os.path.basename(path)}  N={N}  P={P} (G={G})  C={CHANNELS}  "
          f"CR={P*P/CHANNELS:.2f} | bands {K0}..{M} ({n_stages}x{ITERS_PER_STAGE} steps)")

    dec = LinrDecoder(P, channels=CHANNELS, device=dev)
    dataset = [ArrayField(img.to(dev))]
    cfg = ProgressiveConfig(iters_per_stage=ITERS_PER_STAGE, lr=LR, coords_per_step=COORDS,
                            grid=G, k0=K0,
                            on_step=lambda s, K, l: (s % 1000 == 0) and
                            print(f"  step {s:5d}  band {K}  mse {l:.3e}"))
    rep, zs, a_s = dec.pretrain_progressive(dataset, cfg)

    rec = render(dec, zs[0], a_s[0], N, dev); gt = img
    mse = torch.mean((rec - gt) ** 2).item()
    psnr = -10 * math.log10(mse + 1e-12)
    nrmse = 100 * torch.linalg.norm(rec - gt) / torch.linalg.norm(gt)
    print(f"\nFIT:  PSNR {psnr:.2f} dB   NRMSE {nrmse:.2f}%   ({rep.seconds:.1f}s)")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    ax[0].imshow(gt, cmap="gray", vmin=0, vmax=1); ax[0].set_title("ground truth")
    ax[1].imshow(rec, cmap="gray", vmin=0, vmax=1); ax[1].set_title(f"fit  {psnr:.1f} dB")
    d = (rec - gt).numpy(); dl = max(abs(d.min()), abs(d.max())) + 1e-9
    im = ax[2].imshow(d, cmap="seismic", vmin=-dl, vmax=dl)
    ax[2].set_title("difference"); fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a_ in ax[:3]:
        a_.axis("off")
    ax[3].semilogy(rep.loss_history, lw=0.8)
    for k in range(1, n_stages):
        ax[3].axvline(k * ITERS_PER_STAGE, color="0.7", ls=":", lw=0.6)
    ax[3].set_xlabel("step"); ax[3].set_ylabel("MSE"); ax[3].set_title("training loss")
    ax[3].grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Progressive single-image fit  P={P} (G={G})  C={CHANNELS}")
    fig.tight_layout(); os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = f"P{P}_C{CHANNELS}_T{ITERS_PER_STAGE}_im{IMAGE}"
    out = os.path.join(OUTPUT_DIR, f"fit_one_{tag}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
