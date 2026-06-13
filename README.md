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
experiments/     runnable scripts; data + figures live here (gitignored)
tests/           pytest correctness checks
dev_scripts/     conda-env helpers
docs/            theory + API
```

## Use

```bash
python experiments/build_databases.py     # build experiments/img_data/{natural,phantom}
python experiments/fit_one.py             # single-image progressive fit (probe)
python experiments/pretrain.py            # train a decoder (progressive or continue) and SAVE it
python experiments/reconstruct.py         # reconstruct an image with a saved decoder
python experiments/ab_progressive.py      # progressive vs joint A/B (comparison)
pytest                                     # run the tests
```

Typical workflow: `build_databases` → `pretrain` (saves a `.linrd` decoder in
`experiments/models/`) → `reconstruct` (loads it, fits an image).

Scripts can be run from anywhere — data and output paths are anchored to
`experiments/` (`experiments/img_data/` and `experiments/output/`). Edit the
parameters at the top of each script. Runs on CUDA, Apple MPS, or CPU
automatically. Generated data (`img_data/`, `output/`, `*.linrd`, `*.linrz`) is
gitignored.
