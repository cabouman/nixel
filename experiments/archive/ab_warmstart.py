"""
Is the warm start (pretrained theta) and the z-only warmup actually buying
anything? Three arms, all with theta_lr = LR (equal-rate joint fine-tune), same
image, same z-init, same 10000-step budget, full bandwidth:

  warm+warmup : warm-started theta, theta frozen for WARMUP steps, then adapt   (current best)
  warm        : warm-started theta, no warmup (theta adapts from step 0)
  cold        : NO warm start (random theta), no warmup -- same LR for everything

warm vs cold  -> value of the pretrained prior.
warm vs warm+warmup -> whether the z-only warmup helps at all.
"""

# ============================ PARAMETERS ============================
DECODER  = "decoder.linrd"   # in runs/single/ (run: pretrain.py --exp single)
IMAGE    = 0
STEPS    = 10000
LR       = 1e-3            # used for BOTH z and theta (theta_lr = LR)
COORDS   = 65536
WARMUP   = 200            # for the warm+warmup arm only
SEED     = 0
# ====================================================================

import glob, math, os, random
import numpy as np, torch
from PIL import Image
from linr import LinrDecoder, ImageGrid, ReconConfig, pixel_grid, get_device
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/ (for _paths)
from _paths import NATURAL_DIR, RUNS_DIR
OUTPUT_DIR = os.path.join(RUNS_DIR, "archive")          # archive figures live under runs/
MODELS_DIR = os.path.join(RUNS_DIR, "single")           # load runs/single/decoder.linrd


def seed_all(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def psnr_of(rec, gt):
    return -10 * math.log10(torch.mean((rec - gt) ** 2).item() + 1e-12)


def run(dec, warmup, label, img, N, G, dev, gt):
    seed_all(SEED)                                       # identical z-init / coords
    res = dec.reconstruct(img, ImageGrid(N, dev),
                          ReconConfig(steps=STEPS, lr=LR, coords_per_step=COORDS, grid=G,
                                      adapt_theta=True, theta_lr=LR, theta_warmup=warmup))
    rec = res.recon.render(N).cpu()
    ps = psnr_of(rec, gt)
    print(f"  [{label}] render {ps:.2f} dB")
    return ps, -10 * np.log10(np.asarray(res.loss_history) + 1e-12), rec


def main():
    dev = get_device()
    dpath = os.path.join(MODELS_DIR, DECODER)
    if not os.path.exists(dpath):
        raise SystemExit(f"Decoder not found: {dpath} (run pretrain.py first).")
    path = sorted(glob.glob(os.path.join(NATURAL_DIR, "nat_*.png")))[IMAGE]
    img = torch.from_numpy(np.asarray(Image.open(path).convert("L"), np.float32) / 255.0)
    gt = img; N = img.shape[0]
    info = LinrDecoder.load(dpath, device=dev); P, C = info.P, info.channels; G = N // P
    print(f"{DECODER} (P={P} C={C}) | {os.path.basename(path)} {N}x{N} (G={G}) | "
          f"{STEPS} steps, theta_lr=LR={LR:.0e}")

    print("\n[warm + warmup]")
    psWW, cWW, recWW = run(LinrDecoder.load(dpath, device=dev), WARMUP, "warm+warmup", img, N, G, dev, gt)
    print("[warm, no warmup]")
    psW, cW, recW = run(LinrDecoder.load(dpath, device=dev), 0, "warm", img, N, G, dev, gt)
    print("[cold, no warmup]")
    seed_all(SEED); coldF = LinrDecoder(P, channels=C, device=dev)   # reproducible random theta
    psC, cC, recC = run(coldF, 0, "cold", img, N, G, dev, gt)

    print("\n=== final RENDER PSNR ===")
    print(f"  warm + warmup   {psWW:.2f} dB")
    print(f"  warm (no warmup){psW:.2f} dB   ({psW - psWW:+.2f} vs warm+warmup)")
    print(f"  cold            {psC:.2f} dB   ({psC - psWW:+.2f} vs warm+warmup)")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
    for a_, im, t in [(ax[0], gt, "ground truth"),
                      (ax[1], recWW, f"warm+warmup  {psWW:.1f} dB"),
                      (ax[2], recC, f"cold  {psC:.1f} dB")]:
        a_.imshow(im, cmap="gray", vmin=0, vmax=1); a_.set_title(t); a_.axis("off")
    ax[3].plot(cWW, color="C0", lw=1.0, label="warm+warmup")
    ax[3].plot(cW, color="C2", lw=1.0, label="warm (no warmup)")
    ax[3].plot(cC, color="C3", lw=1.0, label="cold")
    ax[3].axvline(WARMUP, color="0.6", ls=":", lw=0.8, alpha=0.7)
    ax[3].set_xlabel("z-fit iteration"); ax[3].set_ylabel("PSNR proxy (dB)")
    ax[3].set_title("convergence"); ax[3].grid(alpha=0.3); ax[3].legend(fontsize=8)
    fig.suptitle(f"warm start & warmup ablation  —  {DECODER}, nat{IMAGE}, {STEPS} steps", fontsize=12)
    fig.tight_layout(); os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"ab_warmstart_P{P}_C{C}_im{IMAGE}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
