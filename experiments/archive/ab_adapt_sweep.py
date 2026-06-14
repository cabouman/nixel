"""
Sweep THETA_LR for adapt-theta reconstruction and compare to the fit_one ceiling.

Three adapt-theta arms (warm-started prior, fine-tune theta at theta_lr in
THETA_LRS) are run with identical z-init / coords / budget; the only difference is
theta_lr. The dashed reference is fit_one: train theta+z from scratch (progressive)
on the same image -- the best single-image quality this architecture reaches.

Curves are the PSNR proxy (-10 log10 of the 65536-coord minibatch MSE). Absolute
numbers are the on-grid RENDER PSNR, printed in the final table (proxy runs ~2 dB
optimistic, consistently across arms).
"""

# ============================ PARAMETERS ============================
DECODER   = "decoder_P8_C8_n1_prog.linrd"   # warm prior theta (.linrd in models/)
IMAGE     = 0
STEPS     = 10000          # z-fit iterations per adapt arm
LR        = 1e-3           # z (and a) learning rate
COORDS    = 65536
WARMUP    = 200            # z-only steps before unfreezing theta
THETA_LRS = [1e-4, 3e-4, 1e-3]
SEED      = 0
# ====================================================================

import glob, math, os, random
import numpy as np, torch
from PIL import Image
from linr import (LinrDecoder, ArrayField, ImageGrid, ReconConfig, ProgressiveConfig,
                  pixel_grid, get_device)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (for _paths)
from _paths import NATURAL_DIR, OUTPUT_DIR, MODELS_DIR


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def render(dec, z, a, N, dev):
    coords = pixel_grid(N, dev).view(-1, 2)
    with torch.no_grad():
        out = torch.cat([dec.decode(c, z, a) for c in torch.split(coords, 1 << 18)])
    return out.view(N, N).cpu()


def psnr_of(rec, gt):
    return -10 * math.log10(torch.mean((rec - gt) ** 2).item() + 1e-12)


def trail(prox, k=25):
    return prox[max(0, len(prox) - k):].mean()


def main():
    dev = get_device()
    dpath = os.path.join(MODELS_DIR, DECODER)
    if not os.path.exists(dpath):
        raise SystemExit(f"Decoder not found: {dpath} (run pretrain.py first).")
    path = sorted(glob.glob(os.path.join(NATURAL_DIR, "nat_*.png")))[IMAGE]
    img = torch.from_numpy(np.asarray(Image.open(path).convert("L"), np.float32) / 255.0)
    gt = img; N = img.shape[0]
    info = LinrDecoder.load(dpath, device=dev)
    P, C, M = info.P, info.channels, info.M
    G = N // P
    field = ArrayField(img.to(dev))
    print(f"{DECODER} (P={P} C={C}) | {os.path.basename(path)} {N}x{N} (G={G}) | "
          f"{STEPS} steps/arm  warmup={WARMUP}")

    # ---- ceiling: fit_one (train theta+z from scratch, progressive) ----
    print("\n[fit_one ceiling]  progressive theta+z from scratch")
    seed_all(SEED)
    decF = LinrDecoder(P, channels=C, device=dev)
    cfgF = ProgressiveConfig(iters_per_stage=STEPS // (M + 1), lr=LR,
                             coords_per_step=COORDS, grid=G, k0=0)
    repF, zsF, asF = decF.pretrain_progressive([field], cfgF)
    psF = psnr_of(render(decF, zsF[0], asF[0], N, dev), gt)
    proxF_ceiling = trail(-10 * np.log10(np.asarray(repF.loss_history) + 1e-12))
    print(f"  render {psF:.2f} dB   (proxy {proxF_ceiling:.2f} dB)")

    # ---- adapt-theta arms ----
    curves, renders, imgs = {}, {}, {}
    for tlr in THETA_LRS:
        print(f"[adapt theta_lr={tlr:.0e}]")
        dec = LinrDecoder.load(dpath, device=dev)        # fresh warm theta each arm
        seed_all(SEED)                                   # identical z-init / coords
        res = dec.reconstruct(img, ImageGrid(N, dev),
                              ReconConfig(steps=STEPS, lr=LR, coords_per_step=COORDS, grid=G,
                                          adapt_theta=True, theta_lr=tlr, theta_warmup=WARMUP))
        rec = res.recon.render(N).cpu()
        curves[tlr] = -10 * np.log10(np.asarray(res.loss_history) + 1e-12)
        renders[tlr] = psnr_of(rec, gt); imgs[tlr] = rec
        print(f"  render {renders[tlr]:.2f} dB")

    print("\n=== final RENDER PSNR ===")
    print(f"  fit_one (ceiling)     {psF:.2f} dB")
    for tlr in THETA_LRS:
        print(f"  adapt theta_lr={tlr:.0e}  {renders[tlr]:.2f} dB   "
              f"({renders[tlr] - psF:+.2f} vs ceiling)")
    best = max(renders, key=renders.get)

    # ---- figure ----
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    recF = render(decF, zsF[0], asF[0], N, dev)
    for a_, im, t in [(ax[0], gt, "ground truth"),
                      (ax[1], recF, f"fit_one ceiling  {psF:.1f} dB"),
                      (ax[2], imgs[best], f"adapt theta_lr={best:.0e}  {renders[best]:.1f} dB")]:
        a_.imshow(im, cmap="gray", vmin=0, vmax=1); a_.set_title(t); a_.axis("off")
    colors = ["C0", "C1", "C2"]
    for tlr, col in zip(THETA_LRS, colors):
        ax[3].plot(curves[tlr], color=col, lw=1.0, label=f"theta_lr={tlr:.0e}")
    ax[3].axhline(proxF_ceiling, color="0.4", ls="--", lw=1.0, label="fit_one (proxy)")
    ax[3].axvline(WARMUP, color="0.6", ls=":", lw=0.8, alpha=0.7)
    ax[3].set_xlabel("z-fit iteration"); ax[3].set_ylabel("PSNR proxy (dB)")
    ax[3].set_title("adapt-theta convergence"); ax[3].grid(alpha=0.3); ax[3].legend(fontsize=8)
    fig.suptitle(f"adapt-theta theta_lr sweep  —  {DECODER}, nat{IMAGE}, {STEPS} steps "
                 f"(ceiling render {psF:.1f} dB)", fontsize=12)
    fig.tight_layout(); os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"ab_adapt_sweep_P{P}_C{C}_im{IMAGE}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
