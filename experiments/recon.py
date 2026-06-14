"""
Reconstruct an image with a saved model -- a runs/<model>/ directory produced by
pretrain.py. Point --model at it; the recon settings come from that run's saved
config.json (override any with --set). Uses the validated adapt-theta recipe
(warm-started theta, fine-tune theta + z, theta_lr = recon_lr, warmup).

Run from the experiments/ directory:

    python recon.py --model blend --image 0
    python recon.py --model blend --database phantom --image 0
    python recon.py --model blend --database phantom --image 0 --set recon_steps=2000

Loads runs/<model>/decoder.linrd and writes results to runs/<model>/recon/.
"""

# adapt-theta recipe (validated defaults; see archive/ab_*):
ADAPT_THETA = True       # fine-tune theta in addition to z
THETA_LR    = None       # None -> recon_lr (equal-rate joint fine-tune)
WARMUP      = 200        # z-only steps before unfreezing theta

import argparse, glob, json, math, os, random
import numpy as np, torch
from PIL import Image
from linr import LinrDecoder, ImageGrid, ReconConfig, get_device
from _paths import NATURAL_DIR, PHANTOM_DIR, RUNS_DIR
from configs import config_from_json

DB_DIRS = {"natural": NATURAL_DIR, "phantom": PHANTOM_DIR}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="single", help="run dir under runs/ (the saved model)")
    ap.add_argument("--image", type=int, default=None, help="override recon_image (sorted index)")
    ap.add_argument("--database", choices=["natural", "phantom"], default=None,
                    help="which DB to reconstruct from (default: the model's database)")
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value")
    args = ap.parse_args()

    dev = get_device()
    run_dir = os.path.join(RUNS_DIR, args.model)
    dpath = os.path.join(run_dir, "decoder.linrd")
    if not os.path.exists(dpath):
        raise SystemExit(f"No model at {dpath}\nRun:  python pretrain.py --name {args.model} ...")
    cfg_path = os.path.join(run_dir, "config.json")
    saved = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    cfg = config_from_json(saved, args.set)   # empty -> defaults, then --set overrides

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    dec = LinrDecoder.load(dpath, device=dev)

    image = cfg.recon_image if args.image is None else args.image
    database = args.database or cfg.database
    paths = sorted(glob.glob(os.path.join(DB_DIRS[database], "*.png")))
    path = paths[image]
    img = torch.from_numpy(np.asarray(Image.open(path).convert("L"), np.float32) / 255.0)
    N = img.shape[0]; G = N // dec.P; gt = img
    tlr = THETA_LR if THETA_LR is not None else cfg.recon_lr
    mode = f"adapt-theta (theta_lr={tlr:.1e}, warmup={WARMUP})" if ADAPT_THETA else "frozen theta"
    print(f"[{args.model}] decoder (P={dec.P} C={dec.channels}) | {database} {os.path.basename(path)} "
          f"{N}x{N} (G={G}) | {mode} | {cfg.recon_steps} steps")

    res = dec.reconstruct(img, ImageGrid(N, dev),
                          ReconConfig(steps=cfg.recon_steps, lr=cfg.recon_lr, coords_per_step=cfg.coords,
                                      grid=G, adapt_theta=ADAPT_THETA, theta_lr=THETA_LR,
                                      theta_warmup=WARMUP))
    fhat = res.recon.render(N).cpu()
    psnr = -10 * math.log10(torch.mean((fhat - gt) ** 2).item() + 1e-12)
    nrmse = 100 * torch.linalg.norm(fhat - gt) / torch.linalg.norm(gt)
    print(f"  PSNR {psnr:.2f} dB   NRMSE {nrmse:.2f}%   (final mse {res.final_loss:.3e})")

    prox = -10 * np.log10(np.asarray(res.loss_history) + 1e-12)
    print("  convergence (PSNR proxy, trailing 25-step mean):")
    for k in [50, 100, 200, 500, 1000, 2000, 5000, len(prox)]:
        if k <= len(prox):
            print(f"    iter {k:6d}:  {prox[max(0, k-25):k].mean():.2f} dB")

    out_dir = os.path.join(run_dir, "recon"); os.makedirs(out_dir, exist_ok=True)
    tag = f"{database}_img{image}" + ("_adapt" if ADAPT_THETA else "")
    res.recon.save(os.path.join(out_dir, f"{tag}.linrz"),
                   metadata={"image": os.path.basename(path), "psnr": psnr})

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
    fig.suptitle(f"reconstruct [{args.model}]  {os.path.basename(path)}")
    fig.tight_layout()
    out = os.path.join(out_dir, f"{tag}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
