"""
Pretrain (or continue training) a LINR decoder on a set of images and SAVE it.

- PROGRESSIVE = True : fresh coarse-to-fine training -- builds the Fourier bands
  up one ring at a time. Use for a NEW decoder.
- PROGRESSIVE = False: full-bandwidth (joint) training. Use to CONTINUE a saved
  decoder (set LOAD_DECODER); progressive would zero its learned features.

The trained decoder is saved to experiments/models/. Then use reconstruct.py to
reconstruct an image with it.
"""

# ============================ PARAMETERS ============================
P               = 8       # pixels per nixel  (NEW decoder only; ignored if loading)
CHANNELS        = 8       # C                 (NEW decoder only; ignored if loading)
NUM_IMAGES      = 1       # images from img_data/natural to train on
ITERS_PER_STAGE = 2000    # progressive: steps per band; joint: total = (M+1)*this
LR              = 1e-3
COORDS          = 65536   # coords per step
K0              = 0       # starting band (progressive)
PROGRESSIVE     = True    # True: fresh coarse-to-fine; False: joint (to continue)
LOAD_DECODER    = ""      # "" = start fresh; else a .linrd filename in models/ to continue
SAVE_DECODER    = ""      # "" = auto-name in models/; else a .linrd filename in models/
SEED            = 0
# ====================================================================

import glob, os, random
import numpy as np, torch
from PIL import Image
from linr import LinrDecoder, ArrayField, ProgressiveConfig, get_device
from _paths import NATURAL_DIR, OUTPUT_DIR, MODELS_DIR


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dev = get_device()

    paths = sorted(glob.glob(os.path.join(NATURAL_DIR, "nat_*.png")))[:NUM_IMAGES]
    if not paths:
        raise SystemExit("No images -- run build_databases.py first.")
    imgs = [torch.from_numpy(np.asarray(Image.open(p).convert("L"), np.float32) / 255.0)
            for p in paths]
    N = imgs[0].shape[0]

    # fresh decoder, or load one to continue training
    if LOAD_DECODER:
        dec = LinrDecoder.load(os.path.join(MODELS_DIR, LOAD_DECODER), device=dev)
        print(f"Loaded decoder {LOAD_DECODER} (P={dec.P}, C={dec.channels}, id {dec.id})")
        if PROGRESSIVE:
            print("  WARNING: PROGRESSIVE=True will ZERO the loaded feature columns "
                  "(fresh curriculum). Set PROGRESSIVE=False to continue training.")
    else:
        dec = LinrDecoder(P, channels=CHANNELS, device=dev)
    Pp, C, M = dec.P, dec.channels, dec.M
    G = N // Pp
    dataset = [ArrayField(im.to(dev)) for im in imgs]

    cfg = ProgressiveConfig(iters_per_stage=ITERS_PER_STAGE, lr=LR, coords_per_step=COORDS,
                            grid=G, k0=K0,
                            on_step=lambda s, K, l: (s % 1000 == 0) and
                            print(f"  step {s:5d}  band {K}  mse {l:.3e}"))

    if PROGRESSIVE:
        n_stages = M - K0 + 1
        print(f"{len(imgs)} img {N}x{N} | P={Pp}(G={G}) C={C} | PROGRESSIVE bands "
              f"{K0}..{M} ({n_stages}x{ITERS_PER_STAGE} = {n_stages*ITERS_PER_STAGE} steps)")
        rep, _, _ = dec.pretrain_progressive(dataset, cfg)
        mode = "prog"
    else:
        total = (M - K0 + 1) * ITERS_PER_STAGE
        print(f"{len(imgs)} img {N}x{N} | P={Pp}(G={G}) C={C} | JOINT full-bandwidth "
              f"{total} steps")
        rep, _, _ = dec.pretrain_joint(dataset, total, cfg)
        mode = "joint"
    print(f"  done ({rep.seconds:.1f}s, final mse {rep.final_loss:.3e})")

    # save the decoder
    name = SAVE_DECODER or f"decoder_P{Pp}_C{C}_n{NUM_IMAGES}_{mode}.linrd"
    save_path = os.path.join(MODELS_DIR, name)
    dec.save(save_path)
    print(f"Saved decoder -> {save_path}  (id {dec.id})")

    # loss curve
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.semilogy(rep.loss_history, lw=0.8)
    if PROGRESSIVE:
        for k in range(1, M - K0 + 1):
            ax.axvline(k * ITERS_PER_STAGE, color="0.7", ls=":", lw=0.6)
    ax.set_xlabel("step"); ax.set_ylabel("MSE"); ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"pretrain loss ({mode})  P={Pp}, C={C}, {NUM_IMAGES} img")
    fig.tight_layout(); os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = f"P{Pp}_C{C}_n{NUM_IMAGES}_T{ITERS_PER_STAGE}_{mode}"
    out = os.path.join(OUTPUT_DIR, f"pretrain_{tag}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
