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
experiments/     runnable scripts (run from the repo root)
tests/           pytest correctness checks
dev_scripts/     conda-env helpers
docs/            theory + API
```

## Use (run from the repo root)

```bash
python experiments/build_databases.py     # build img_data/{natural,phantom}
python experiments/fit_one.py             # single-image progressive fit
python experiments/run_pretrain.py        # progressive pretrain + reconstruct
python experiments/ab_progressive.py      # progressive vs joint A/B
pytest                                     # run the tests
```

Figures are written to `output/`. Runs on CUDA, Apple MPS, or CPU automatically.
Generated data (`img_data/`, `output/`, `*.linrd`, `*.linrz`) is gitignored.
