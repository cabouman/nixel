"""
Named experiment presets -- the single place experiment parameters live.

Pick one with `--exp <name>` in pretrain.py / reconstruct.py; tweak ad hoc with
`--set key=value` (e.g. `--set iters_per_stage=50000 num_images=500`). Add a new
experiment by adding an entry to EXPERIMENTS.
"""

import dataclasses


@dataclasses.dataclass
class ExperimentConfig:
    # ---- data: pretrain on num_images natural + num_phantoms phantom images ----
    num_images: int = 1              # natural images to pretrain on (0 = none)
    image_start: int = 0             # index of the first natural image (sorted); holds out 0..image_start-1
    num_phantoms: int = 0            # phantom images to ALSO include in pretraining (0 = none)
    phantom_start: int = 0           # index of the first phantom image (sorted)
    database: str = "natural"        # which DB the RECONSTRUCTION target (recon_image) comes from
    # ---- decoder (lightweight: evaluated per-voxel per-CT-iteration) ----
    P: int = 8                       # pixels per nixel
    channels: int = 8                # C
    hidden: int = 128
    layers: int = 4
    out_act: str = "linear"          # "linear" | "nonneg"
    # ---- pretraining ----
    progressive: bool = True         # coarse-to-fine (recommended for theta)
    iters_per_stage: int = 2000      # progressive: steps/band; joint: total steps
    lr: float = 1e-3                 # latent (z, a) LR
    theta_lr_frac: float = 1.0       # theta LR = theta_lr_frac * lr (<1 -> theta gentler than the z latents)
    coords: int = 65536              # random pixels per step
    k0: int = 0                      # progressive starting band
    seed: int = 0
    checkpoint_every: int = 0        # 0 = checkpoint only at the end
    # ---- default reconstruction (used by recon.py) ----
    recon_image: int = 0             # which image (sorted index) to reconstruct
    recon_steps: int = 10000         # z-fit iterations
    recon_lr: float = 2e-2           # z (and a) LR (tuned for ~1000-step recon)
    recon_theta_lr_frac: float = 0.3 # theta LR = this fraction of recon_lr (gentler theta lets z run faster)
    recon_warmup: int = 0            # z-only steps before adapting theta (0 = adapt from start)


EXPERIMENTS = {
    # Quick, single-image case to run locally (matches the validated single-image flow).
    "single": ExperimentConfig(
        num_images=1, image_start=0,           # single image: no z undersampling -> lr=1e-3, frac=1.0
        iters_per_stage=2000, checkpoint_every=0,
        recon_image=0, recon_steps=1000,
    ),
    # Main template: a blend of naturals + phantoms, holding out indices 0..9 of each
    # for testing. Edit here or override with --set; keep the resulting model with --name.
    "default": ExperimentConfig(
        num_images=90, image_start=10,         # naturals 10..99 (hold out 0..9)
        num_phantoms=90, phantom_start=10,     # phantoms 10..99 (hold out 0..9 incl. Shepp-Logan)
        iters_per_stage=20000, checkpoint_every=2000,   # 5 stages x 20000 = 100,000 total
        lr=1e-2, theta_lr_frac=0.5,            # decoupled: z fast (1e-2), theta gentler (5e-3); ~+1 dB at fixed budget
        recon_image=0, recon_steps=1000,
    ),
}


def _coerce(cur, value):
    if isinstance(cur, bool):
        return str(value).lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int):
        return int(value)
    if isinstance(cur, float):
        return float(value)
    return value


def _apply_overrides(cfg, overrides):
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        key, value = item.split("=", 1)
        if not hasattr(cfg, key):
            raise SystemExit(f"--set: unknown field '{key}'. "
                             f"Fields: {', '.join(f.name for f in dataclasses.fields(cfg))}")
        setattr(cfg, key, _coerce(getattr(cfg, key), value))
    return cfg


def get_experiment(name, overrides=None):
    """A COPY of the named template with `--set` overrides (used by pretrain.py)."""
    if name not in EXPERIMENTS:
        raise SystemExit(f"Unknown --exp '{name}'. Choices: {', '.join(EXPERIMENTS)}")
    return _apply_overrides(dataclasses.replace(EXPERIMENTS[name]), overrides)


def config_from_json(meta, overrides=None):
    """Rebuild an ExperimentConfig from a run's saved config.json dict + `--set`
    overrides (used by recon.py so a saved model reconstructs with its own settings)."""
    fields = {f.name for f in dataclasses.fields(ExperimentConfig)}
    return _apply_overrides(ExperimentConfig(**{k: v for k, v in meta.items() if k in fields}),
                            overrides)
