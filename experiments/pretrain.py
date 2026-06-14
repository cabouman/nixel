"""
Pretrain a LINR decoder from a configs.py template into a runs/<name>/ model dir.
Run from the experiments/ directory:

    python pretrain.py --exp default --name blend        # -> runs/blend/
    python pretrain.py --exp default --name blend --set iters_per_stage=100000
    nohup python pretrain.py --exp default --name blend > runs/blend.log 2>&1 &
    python pretrain.py --exp default --name blend --resume   # continue after interruption

--exp picks the template; --name is the run-dir name (default: --exp) -- pick a
unique --name to keep each model. Outputs land in runs/<name>/: config.json,
state.pt (resumable), decoder.linrd (latest theta), loss.npy, pretrain_loss.png.
Reconstruct with:  python recon.py --model <name> --image <i>.
"""

import argparse, dataclasses, glob, json, os, random
import numpy as np, torch
from PIL import Image
from linr import LinrDecoder, ArrayField, ProgressiveConfig, get_device
from _paths import NATURAL_DIR, PHANTOM_DIR, RUNS_DIR
from configs import get_experiment, ExperimentConfig



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="single", help="template name in configs.py")
    ap.add_argument("--name", default=None, help="run-dir name under runs/ (default: --exp); "
                                                 "pick a unique name to keep the model")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/state.pt")
    ap.add_argument("--init-from", default=None, metavar="MODEL",
                    help="warm-start theta from runs/<MODEL>/decoder.linrd, then train JOINT")
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value",
                    help="override config fields, e.g. --set iters_per_stage=50000")
    args = ap.parse_args()
    dev = get_device()
    run_name = args.name or args.exp
    run_dir = os.path.join(RUNS_DIR, run_name)
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

    if args.init_from and not args.resume and cfg.progressive:
        # progressive zeros the first-layer Fourier columns at each stage, which
        # would wipe the warm-started theta -- so warm-start implies joint training.
        print("  --init-from given: forcing joint training (progressive would zero the warm-started theta)")
        cfg = dataclasses.replace(cfg, progressive=False)

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
    N = imgs[0].shape[0]
    dataset = [ArrayField(im.to(dev)) for im in imgs]

    # ---- decoder: warm-start theta from another model (--init-from) or build fresh ----
    # (when resuming, theta is restored from state.pt below)
    if args.init_from and not args.resume:
        src = os.path.join(RUNS_DIR, args.init_from, "decoder.linrd")
        if not os.path.exists(src):
            raise SystemExit(f"--init-from: no decoder at {src}")
        dec = LinrDecoder.load(src, device=dev)
        print(f"  warm-started theta from {args.init_from} (decoder id {dec.id})")
    else:
        dec = LinrDecoder(cfg.P, channels=cfg.channels, hidden=cfg.hidden,
                          layers=cfg.layers, out_act=cfg.out_act, device=dev)
    M = dec.M
    G = N // dec.P
    total = (M - cfg.k0 + 1) * cfg.iters_per_stage

    resume_state = None
    if args.resume:
        resume_state = torch.load(state_path, map_location=dev, weights_only=False)
        print(f"Resuming {run_name} from step {resume_state['global_step']}/{total}")

    mode = "progressive" if cfg.progressive else "joint"
    print(f"[{run_name}] {len(nat)} natural + {len(phn)} phantom = {len(imgs)} img {N}x{N} | "
          f"P={cfg.P}(G={G}) C={cfg.channels} hidden={cfg.hidden} layers={cfg.layers} | {mode} | "
          f"lr={cfg.lr:.0e} (theta {cfg.theta_lr_frac}x) | total {total} steps | "
          f"checkpoint_every={cfg.checkpoint_every}")

    # write config up front so an interrupted run is still resumable
    def write_config(extra=None):
        meta = {**dataclasses.asdict(cfg), "exp": args.exp, "total_steps": total,
                "init_from": args.init_from}
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
                             theta_lr_frac=cfg.theta_lr_frac, coords_per_step=cfg.coords,
                             grid=G, k0=cfg.k0, on_step=on_step)
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
    ax.set_title(f"pretrain [{run_name}] {mode}  {len(imgs)} img  P={cfg.P} C={cfg.channels}")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "pretrain_loss.png"), dpi=120, bbox_inches="tight")
    print(f"Saved run -> {run_dir}  (decoder.linrd, config.json, loss.npy, pretrain_loss.png)")


if __name__ == "__main__":
    main()
