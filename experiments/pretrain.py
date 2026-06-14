"""
Pretrain a LINR decoder for a named experiment, into its own run directory.
Run from the experiments/ directory:

    python pretrain.py --exp single
    python pretrain.py --exp many                 # long job; checkpoints
    nohup python pretrain.py --exp many > runs/many.log 2>&1 &
    python pretrain.py --exp many --resume        # continue after interruption

Outputs land in runs/<exp>/:  config.json, state.pt (resumable), decoder.linrd
(latest theta), loss.npy, pretrain_loss.png. Edit configs.py to change parameters
or add experiments; override ad hoc with `--set key=value`.
"""

import argparse, dataclasses, glob, json, os, random
import numpy as np, torch
from PIL import Image
from linr import LinrDecoder, ArrayField, ProgressiveConfig, get_device
from _paths import NATURAL_DIR, PHANTOM_DIR, RUNS_DIR
from configs import get_experiment, ExperimentConfig



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="single", help="experiment name in configs.py")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<exp>/state.pt")
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value",
                    help="override config fields, e.g. --set iters_per_stage=50000")
    args = ap.parse_args()
    dev = get_device()
    run_dir = os.path.join(RUNS_DIR, args.exp)
    os.makedirs(run_dir, exist_ok=True)
    state_path = os.path.join(run_dir, "state.pt")
    decoder_path = os.path.join(run_dir, "decoder.linrd")
    cfg_path = os.path.join(run_dir, "config.json")

    if args.resume:
        # continue the SAME experiment: use the run's saved config, not the CLI
        if not os.path.exists(state_path):
            raise SystemExit(f"--resume: no checkpoint at {state_path}")
        if not os.path.exists(cfg_path):
            raise SystemExit(f"--resume: no config.json in {run_dir}")
        if args.set:
            print("  (ignoring --set on --resume; using the run's saved config)")
        with open(cfg_path) as f:
            saved = json.load(f)
        fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
        cfg = ExperimentConfig(**{k: v for k, v in saved.items() if k in fields})
    else:
        cfg = get_experiment(args.exp, args.set)

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

    # ---- data: num_images natural (from image_start) + num_phantoms phantom (from phantom_start) ----
    nat = sorted(glob.glob(os.path.join(NATURAL_DIR, "*.png")))[cfg.image_start: cfg.image_start + cfg.num_images]
    phn = sorted(glob.glob(os.path.join(PHANTOM_DIR, "*.png")))[cfg.phantom_start: cfg.phantom_start + cfg.num_phantoms]
    paths = nat + phn
    if not paths:
        raise SystemExit("No training images -- run build_databases.py first "
                         "(or check num_images / num_phantoms).")
    imgs = [torch.from_numpy(np.asarray(Image.open(p).convert("L"), np.float32) / 255.0)
            for p in paths]
    N = imgs[0].shape[0]; G = N // cfg.P
    dataset = [ArrayField(im.to(dev)) for im in imgs]

    # ---- decoder (built from config; theta restored below if resuming) ----
    dec = LinrDecoder(cfg.P, channels=cfg.channels, hidden=cfg.hidden,
                      layers=cfg.layers, out_act=cfg.out_act, device=dev)
    M = dec.M
    total = (M - cfg.k0 + 1) * cfg.iters_per_stage

    resume_state = None
    if args.resume:
        resume_state = torch.load(state_path, map_location=dev, weights_only=False)
        print(f"Resuming {args.exp} from step {resume_state['global_step']}/{total}")

    mode = "progressive" if cfg.progressive else "joint"
    print(f"[{args.exp}] {len(nat)} natural + {len(phn)} phantom = {len(imgs)} img {N}x{N} | "
          f"P={cfg.P}(G={G}) C={cfg.channels} hidden={cfg.hidden} layers={cfg.layers} | {mode} | "
          f"total {total} steps | checkpoint_every={cfg.checkpoint_every}")

    # write config up front so an interrupted run is still resumable
    def write_config(extra=None):
        meta = {**dataclasses.asdict(cfg), "exp": args.exp, "total_steps": total}
        with open(cfg_path, "w") as f:
            json.dump({**meta, **(extra or {})}, f, indent=2)
    write_config()

    # ---- checkpointing ----
    def save_ckpt(step, ckpt):
        tmp = state_path + ".tmp"
        torch.save(ckpt, tmp); os.replace(tmp, state_path)   # atomic full state for resume
        dec.save(decoder_path)                               # latest theta deliverable
        print(f"  [ckpt] step {step}/{total} -> {os.path.relpath(state_path, RUNS_DIR)}", flush=True)

    log_every = max(1, total // 50)

    def on_step(step, K, loss):
        if step % log_every == 0:
            print(f"  step {step:7d}/{total}  band {K}  mse {loss:.3e}", flush=True)

    pcfg = ProgressiveConfig(iters_per_stage=cfg.iters_per_stage, lr=cfg.lr,
                             coords_per_step=cfg.coords, grid=G, k0=cfg.k0, on_step=on_step)
    kw = dict(checkpoint_every=cfg.checkpoint_every, checkpoint_fn=save_ckpt, resume=resume_state)
    if cfg.progressive:
        rep, _, _ = dec.pretrain_progressive(dataset, pcfg, **kw)
    else:
        rep, _, _ = dec.pretrain_joint(dataset, total, pcfg, **kw)
    print(f"  done ({rep.seconds:.1f}s, final mse {rep.final_loss:.3e})  decoder id {dec.id}")

    # ---- persist final config metadata + loss ----
    write_config({"decoder_id": dec.id, "seconds": rep.seconds, "final_mse": rep.final_loss})
    np.save(os.path.join(run_dir, "loss.npy"), np.asarray(rep.loss_history, np.float32))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.semilogy(rep.loss_history, lw=0.8)
    if cfg.progressive:
        for k in range(1, M - cfg.k0 + 1):
            ax.axvline(k * cfg.iters_per_stage, color="0.7", ls=":", lw=0.6)
    ax.set_xlabel("step"); ax.set_ylabel("MSE"); ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"pretrain [{args.exp}] {mode}  {len(imgs)} img  P={cfg.P} C={cfg.channels}")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "pretrain_loss.png"), dpi=120, bbox_inches="tight")
    print(f"Saved run -> {run_dir}  (decoder.linrd, config.json, loss.npy, pretrain_loss.png)")


if __name__ == "__main__":
    main()
