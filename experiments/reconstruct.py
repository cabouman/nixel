"""
Reconstruct an image with a TRAINED (frozen) decoder: fit only the latent z,
render at full resolution, and compare to ground truth. Optionally save the
reconstruction (.linrz).

Run run_pretrain.py first to produce a decoder (a .linrd in experiments/models/).
"""

# ============================ PARAMETERS ============================
DECODER     = "decoder_P8_C8_n8_prog.linrd"   # .linrd filename in experiments/models/
IMAGE       = 0        # which nat_<k>.png to reconstruct
RECON_STEPS = 1500     # z-fit iterations (decoder frozen)
LR          = 5e-3
COORDS      = 65536    # coords per step
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
                         "(run run_pretrain.py first).")
    dec = LinrDecoder.load(dpath, device=dev)

    paths = sorted(glob.glob(os.path.join(NATURAL_DIR, "nat_*.png")))
    path = paths[IMAGE]
    img = torch.from_numpy(np.asarray(Image.open(path).convert("L"), np.float32) / 255.0)
    N = img.shape[0]; G = N // dec.P
    print(f"decoder {DECODER} (P={dec.P}, C={dec.channels}) | reconstruct "
          f"{os.path.basename(path)}  {N}x{N} (G={G})")

    res = dec.reconstruct(img, ImageGrid(N, dev),
                          ReconConfig(steps=RECON_STEPS, lr=LR, coords_per_step=COORDS, grid=G))
    fhat = res.recon.render(N).cpu(); gt = img
    psnr = -10 * math.log10(torch.mean((fhat - gt) ** 2).item() + 1e-12)
    nrmse = 100 * torch.linalg.norm(fhat - gt) / torch.linalg.norm(gt)
    print(f"  PSNR {psnr:.2f} dB   NRMSE {nrmse:.2f}%   (final mse {res.final_loss:.3e})")

    tag = f"nat{IMAGE}_P{dec.P}_C{dec.channels}"
    if SAVE_RECON:
        rpath = os.path.join(RECONS_DIR, f"{tag}.linrz")
        res.recon.save(rpath, metadata={"image": os.path.basename(path), "psnr": psnr})
        print(f"  saved reconstruction -> {rpath}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14.4, 4.6))
    ax[0].imshow(gt, cmap="gray", vmin=0, vmax=1); ax[0].set_title("ground truth")
    ax[1].imshow(fhat, cmap="gray", vmin=0, vmax=1); ax[1].set_title(f"reconstruction  {psnr:.1f} dB")
    d = (fhat - gt).numpy(); dl = max(abs(d.min()), abs(d.max())) + 1e-9
    im = ax[2].imshow(d, cmap="seismic", vmin=-dl, vmax=dl)
    ax[2].set_title("difference"); fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a_ in ax:
        a_.axis("off")
    fig.suptitle(f"Reconstruct with {DECODER}")
    fig.tight_layout(); os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"reconstruct_{tag}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
