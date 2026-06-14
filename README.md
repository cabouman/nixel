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
experiments/     runnable scripts; data + run outputs live here (gitignored)
  configs.py     named experiment presets (the one place params live)
  runs/          per-experiment outputs: config.json, decoder.linrd, state.pt, figures
  archive/       settled A/B studies (progressive, adapt-theta) + notes
tests/           pytest correctness checks
dev_scripts/     conda-env helpers
docs/            theory + API
```

## Use

Experiments are named in [`experiments/configs.py`](experiments/configs.py)
(`single`, `many`, ...). Select one with `--exp`; override fields ad hoc with
`--set key=value`.

```bash
python experiments/build_databases.py             # build img_data/{natural,phantom}

python experiments/pretrain.py --exp single       # pretrain -> runs/single/decoder.linrd
python experiments/reconstruct.py --exp single    # reconstruct (warm-start adapt-theta)

# long job: kick off, leave it, resume after an interruption
nohup python experiments/pretrain.py --exp many > runs/many.log 2>&1 &
python experiments/pretrain.py --exp many --resume   # continue from runs/many/state.pt

python experiments/fit_one.py                     # single-image progressive fit (probe)
pytest                                             # run the tests
```

Typical workflow: `build_databases` → `pretrain --exp <name>` (writes
`runs/<name>/`) → `reconstruct --exp <name>` (loads `runs/<name>/decoder.linrd`).
Long `--exp many` runs checkpoint every `checkpoint_every` steps to
`state.pt`; `--resume` continues from the run's *saved* config (CLI overrides are
ignored on resume).

Scripts run from anywhere — paths are anchored to `experiments/`. Runs on CUDA,
Apple MPS, or CPU automatically. Generated data (`img_data/`, `output/`, `runs/`,
`*.linrd`, `*.linrz`) is gitignored.
