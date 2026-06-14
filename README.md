# nixel — LINR

A memory-efficient, continuous representation for images and volumes (aimed at
tomographic reconstruction): a coarse grid of small local latent vectors
("**nixels**") decoded by a small shared MLP.

## Documentation

- [`docs/theory-v1.md`](docs/theory-v1.md) — the representation: coordinates,
  nixel interpolation, periodic Fourier basis, decoder, compression.
- [`docs/theory-v2.md`](docs/theory-v2.md) — nested Fourier bands and progressive
  (coarse-to-fine) training.
- [`docs/API_def.md`](docs/API_def.md) — the user-facing API.

## Install

```bash
pip install -e ".[dev]"     # editable install of the `linr` package + pytest
```

## Layout

```
linr.py          the package (import linr)
experiments/     the pipeline: build_databases, configs, pretrain, reconstruct
  configs.py     named experiment presets (the one place params live)
  img_data/      built image databases (gitignored)
  runs/          ALL generated outputs: config.json, decoder.linrd, state.pt, recon/, figures (gitignored)
  archive/       settled A/B studies + fit_one probe + notes
tests/           pytest correctness checks
dev_scripts/     conda-env helpers
docs/            theory + API
```

## Use

Run everything from the `experiments/` directory. Experiments are named in
[`configs.py`](experiments/configs.py) (`single`, `many`, ...). Select one with
`--exp`; override fields ad hoc with `--set key=value`.

```bash
cd experiments              # all commands below are run from here

python build_databases.py                 # build img_data/{natural,phantom}

python pretrain.py --exp single           # pretrain -> runs/single/decoder.linrd
python reconstruct.py --exp single        # reconstruct (warm-start adapt-theta)

# long job: kick off, leave it, resume after an interruption
nohup python pretrain.py --exp many > runs/many.log 2>&1 &
tail -f runs/many.log                     # watch progress
python pretrain.py --exp many --resume    # continue from runs/many/state.pt

cd .. && pytest                           # tests run from the repo root
```

Reference studies (kept frozen, not part of the pipeline) live in
`experiments/archive/` — the progressive and adapt-theta A/B comparisons and the
`fit_one.py` single-image probe; see [`archive/README.md`](experiments/archive/README.md).

The four typical commands are collected in `run_exp.sh` (run from `experiments/`).

Typical workflow: `build_databases` → `pretrain --exp <name>` (writes
`runs/<name>/`) → `reconstruct --exp <name>` (loads `runs/<name>/decoder.linrd`).
Long `--exp many` runs checkpoint every `checkpoint_every` steps to
`state.pt`; `--resume` continues from the run's *saved* config (CLI overrides are
ignored on resume).

Data and output paths are anchored to `experiments/` (so a script still works if
launched from elsewhere), but the convention is to run from `experiments/`. Runs on
CUDA, Apple MPS, or CPU automatically. All generated data lives in `img_data/` and
`runs/` (under `experiments/`), both gitignored.
