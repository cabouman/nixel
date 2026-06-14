"""
Named experiment presets -- the single place experiment parameters live.

Pick one with `--exp <name>` in pretrain.py / reconstruct.py; tweak ad hoc with
`--set key=value` (e.g. `--set iters_per_stage=50000 num_images=500`). Add a new
experiment by adding an entry to EXPERIMENTS.
"""

import dataclasses
from typing import Optional


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
    lr: float = 1e-3
    coords: int = 65536              # random pixels per step
    k0: int = 0                      # progressive starting band
    seed: int = 0
    checkpoint_every: int = 0        # 0 = checkpoint only at the end
    # ---- default reconstruction (used by reconstruct.py) ----
    recon_image: int = 0             # which image (sorted index) to reconstruct
    recon_steps: int = 10000         # z-fit iterations


EXPERIMENTS = {
    # Quick, single-image case to run locally (matches the validated single-image flow).
    "single": ExperimentConfig(
        num_images=1, image_start=0,
        iters_per_stage=2000, checkpoint_every=0,
        recon_image=0, recon_steps=10000,
    ),
    # Many-image pretraining -- a long job. Kick off and let it run; checkpoints every
    # 2000 steps so it survives interruption (resume with `--resume`).
    "many": ExperimentConfig(
        num_images=200, image_start=10,        # train on natural 10.. ; hold out 0..9 for testing
        num_phantoms=0, phantom_start=0,       # set num_phantoms>0 to mix in phantom images
        iters_per_stage=20000, checkpoint_every=2000,
        recon_image=0, recon_steps=10000,      # recon_image=0 is held out (image_start=10)
    ),
    # Like 'many' but its own run dir (won't clobber 'many'); 100,000 iterations per
    # stage -> P=8's 5 progressive stages give 500,000 total.
    "many-v2": ExperimentConfig(
        num_images=200, image_start=10,
        num_phantoms=0, phantom_start=0,
        iters_per_stage=100000, checkpoint_every=2000,  # 5 stages x 100000 = 500,000 total
        recon_image=0, recon_steps=10000,
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


def get_experiment(name, overrides=None):
    """Return a COPY of the named ExperimentConfig with `--set key=value` overrides applied."""
    if name not in EXPERIMENTS:
        raise SystemExit(f"Unknown --exp '{name}'. Choices: {', '.join(EXPERIMENTS)}")
    cfg = dataclasses.replace(EXPERIMENTS[name])
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got '{item}'")
        key, value = item.split("=", 1)
        if not hasattr(cfg, key):
            raise SystemExit(f"--set: unknown field '{key}'. "
                             f"Fields: {', '.join(f.name for f in dataclasses.fields(cfg))}")
        setattr(cfg, key, _coerce(getattr(cfg, key), value))
    return cfg
