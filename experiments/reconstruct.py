"""
Reconstruct an image with a TRAINED (frozen) decoder: fit only the latent z,
render at full resolution, and compare to ground truth. Optionally save the
reconstruction (.linrz).

Run pretrain.py first to produce a decoder (a .linrd in experiments/models/).
"""

# ============================ PARAMETERS ============================
DECODER     = "decoder_P8_C8_n1_prog.linrd"   # .linrd filename in experiments/models/
IMAGE       = 0        # which nat_<k>.png to reconstruct
RECON_STEPS = 10000    # z-fit iterations; = pretrain total (M+1)*2000
LR          = 1e-3     # z (and a) learning rate
COORDS      = 65536    # coords per step
ADAPT_THETA = True     # fine-tune theta (warm-started prior) in addition to z -- validated default
THETA_LR    = None     # theta LR while adapting; None -> LR (equal-rate joint fine-tune)
WARMUP      = 200      # z-only steps before unfreezing theta
SAVE_RECON  = True     # also save the reconstruction as a .linrz
SEED        = 0
# ====================================================================

import glob, math, os, random
import numpy as np, torch
from PIL import Image
from linr import LinrDecoder, ImageGrid, ReconConfig, get_device
from _paths import NATURAL_DIR, OUTPUT_DIR, MODELS_DIR, RECONS_DIR


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dev = get_device()

    dpath = os.path.join(MODELS_DIR, DECODER)
    if not DECODER or not os.path.exists(dpath):
        raise SystemExit(f"Decoder not found: {dpath}\n"
                         "Set DECODER to a .linrd in experiments/models/ "
                         "(run pretrain.py first).")
    dec = LinrDecoder.load(dpath, device=dev)

    paths = sorted(glob.glob(os.path.join(NATURAL_DIR, "nat_*.png")))
    path = paths[IMAGE]
    img = torch.from_numpy(np.asarray(Image.open(path).convert("L"), np.float32) / 255.0)
    N = img.shape[0]; G = N // dec.P
    mode = (f"adapt-theta (theta_lr={(THETA_LR if THETA_LR is not None else LR):.1e}, warmup={WARMUP})"
            if ADAPT_THETA else "frozen theta (z only)")
    print(f"decoder {DECODER} (P={dec.P}, C={dec.channels}) | reconstruct "
          f"{os.path.basename(path)}  {N}x{N} (G={G}) | {mode}")

    res = dec.reconstruct(img, ImageGrid(N, dev),
                          ReconConfig(steps=RECON_STEPS, lr=LR, coords_per_step=COORDS, grid=G,
                                      adapt_theta=ADAPT_THETA, theta_lr=THETA_LR,
                                      theta_warmup=WARMUP))
    fhat = res.recon.render(N).cpu(); gt = img
    psnr = -10 * math.log10(torch.mean((fhat - gt) ** 2).item() + 1e-12)
    nrmse = 100 * torch.linalg.norm(fhat - gt) / torch.linalg.norm(gt)
    print(f"  PSNR {psnr:.2f} dB   NRMSE {nrmse:.2f}%   (final mse {res.final_loss:.3e})")

    tag = f"nat{IMAGE}_P{dec.P}_C{dec.channels}" + ("_adapt" if ADAPT_THETA else "")
    if SAVE_RECON:
        rpath = os.path.join(RECONS_DIR, f"{tag}.linrz")
        res.recon.save(rpath, metadata={"image": os.path.basename(path), "psnr": psnr})
        print(f"  saved reconstruction -> {rpath}")

    # convergence: minibatch MSE -> PSNR proxy (COORDS large => low variance)
    prox = -10 * np.log10(np.asarray(res.loss_history) + 1e-12)
    print("  convergence (PSNR proxy, trailing 25-step mean):")
    for k in [50, 100, 200, 500, 1000, 2000, 5000, len(prox)]:
        if k <= len(prox):
            print(f"    iter {k:6d}:  {prox[max(0, k-25):k].mean():.2f} dB")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    ax[0].imshow(gt, cmap="gray", vmin=0, vmax=1); ax[0].set_title("ground truth")
    ax[1].imshow(fhat, cmap="gray", vmin=0, vmax=1); ax[1].set_title(f"reconstruction  {psnr:.1f} dB")
    d = (fhat - gt).numpy(); dl = max(abs(d.min()), abs(d.max())) + 1e-9
    im = ax[2].imshow(d, cmap="seismic", vmin=-dl, vmax=dl)
    ax[2].set_title("difference"); fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a_ in ax[:3]:
        a_.axis("off")

    w = min(101, len(prox) // 2 * 2 + 1)
    sm = np.convolve(prox, np.ones(w) / w, mode="valid"); off = (w - 1) // 2
    ax[3].plot(np.arange(len(prox)), prox, color="C0", alpha=0.25, lw=0.6)
    ax[3].plot(np.arange(len(sm)) + off, sm, color="C0", lw=1.2)
    ax[3].set_xlabel("z-fit iteration"); ax[3].set_ylabel("PSNR proxy (dB)")
    ax[3].set_title("reconstruction convergence"); ax[3].grid(alpha=0.3)
    fig.suptitle(f"Reconstruct with {DECODER}")
    fig.tight_layout(); os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"reconstruct_{tag}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
