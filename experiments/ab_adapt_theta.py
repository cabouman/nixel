"""
A/B at reconstruction time: FROZEN theta (fit z only) vs ADAPT theta (warm-started
prior, fine-tune theta in addition to z). Same decoder, same image, same z-init,
same coordinate stream, same budget -- the only difference is whether theta adapts.

Shows how much adapting theta buys over the frozen-theta reconstruction, and how
fast it gets there (PSNR proxy = -10 log10 of the 65536-coord minibatch MSE).
"""

# ============================ PARAMETERS ============================
DECODER  = "decoder_P8_C8_n1_prog.linrd"   # frozen prior theta (.linrd in models/)
IMAGE    = 0          # nat_<k>.png the decoder was trained on
STEPS    = 10000      # z-fit iterations per arm
LR       = 1e-3       # z (and a) learning rate
COORDS   = 65536      # random pixels per step
THETA_LR = None       # theta LR while adapting; None -> 0.1*LR
WARMUP   = 500        # z-only steps before unfreezing theta (adapt arm)
SEED     = 0          # shared z-init / coord stream
# ====================================================================

import glob, math, os, random
import numpy as np, torch
from PIL import Image
from linr import LinrDecoder, ImageGrid, ReconConfig, get_device
from _paths import NATURAL_DIR, OUTPUT_DIR, MODELS_DIR


def run(img, adapt, dev):
    # fresh decoder each arm so an adapted theta never leaks into the other arm
    dec = LinrDecoder.load(os.path.join(MODELS_DIR, DECODER), device=dev)
    N = img.shape[0]; G = N // dec.P
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)   # identical z-init/coords
    res = dec.reconstruct(img, ImageGrid(N, dev),
                          ReconConfig(steps=STEPS, lr=LR, coords_per_step=COORDS, grid=G,
                                      adapt_theta=adapt, theta_lr=THETA_LR, theta_warmup=WARMUP))
    fhat = res.recon.render(N).cpu()
    psnr = -10 * math.log10(torch.mean((fhat - img) ** 2).item() + 1e-12)
    return fhat, psnr, np.asarray(res.loss_history)


def milestones(prox, label):
    print(f"  [{label}] PSNR proxy (trailing 25-step mean):")
    for k in [100, 200, 500, 1000, 2000, 5000, len(prox)]:
        if k <= len(prox):
            print(f"      iter {k:6d}:  {prox[max(0, k-25):k].mean():.2f} dB")


def main():
    dev = get_device()
    if not os.path.exists(os.path.join(MODELS_DIR, DECODER)):
        raise SystemExit(f"Decoder not found in models/: {DECODER} (run pretrain.py).")
    path = sorted(glob.glob(os.path.join(NATURAL_DIR, "nat_*.png")))[IMAGE]
    img = torch.from_numpy(np.asarray(Image.open(path).convert("L"), np.float32) / 255.0)
    N = img.shape[0]
    print(f"{DECODER} | {os.path.basename(path)} {N}x{N} | {STEPS} steps/arm "
          f"| theta_lr={THETA_LR or 0.1*LR:.1e} warmup={WARMUP}")

    print("\n[FROZEN]  fit z only")
    fhatF, psF, lossF = run(img, False, dev)
    print(f"  final render {psF:.2f} dB")
    print("[ADAPT]   fit z + theta")
    fhatA, psA, lossA = run(img, True, dev)
    print(f"  final render {psA:.2f} dB")

    proxF = -10 * np.log10(lossF + 1e-12)
    proxA = -10 * np.log10(lossA + 1e-12)
    print()
    milestones(proxF, "FROZEN"); milestones(proxA, "ADAPT")
    print(f"\nFINAL (render):  FROZEN {psF:.2f} dB   ADAPT {psA:.2f} dB   (Δ = {psA - psF:+.2f} dB)")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    for a_, im, t in [(ax[0], img, "ground truth"),
                      (ax[1], fhatF, f"frozen theta  {psF:.1f} dB"),
                      (ax[2], fhatA, f"adapt theta  {psA:.1f} dB")]:
        a_.imshow(im, cmap="gray", vmin=0, vmax=1); a_.set_title(t); a_.axis("off")
    ax[3].plot(proxF, color="C3", lw=1.0, label="frozen theta")
    ax[3].plot(proxA, color="C0", lw=1.0, label="adapt theta")
    ax[3].axvline(WARMUP, color="C0", ls=":", lw=0.8, alpha=0.7, label="theta unfreeze")
    ax[3].set_xlabel("z-fit iteration"); ax[3].set_ylabel("PSNR proxy (dB)")
    ax[3].set_title("reconstruction convergence"); ax[3].grid(alpha=0.3); ax[3].legend(fontsize=9)
    fig.suptitle(f"frozen vs adapt-theta  —  {DECODER}, nat{IMAGE}, {STEPS} steps", fontsize=12)
    fig.tight_layout(); os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"ab_adapt_theta_P8_C8_im{IMAGE}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
