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
experiments/     the pipeline: build_databases, configs, pretrain, recon
  configs.py     experiment templates (single, default) -- the one place params live
  img_data/      built image databases (gitignored)
  runs/          model library: one runs/<name>/ per pretrained model (gitignored)
  archive/       settled A/B studies + fit_one probe + notes
tests/           pytest correctness checks
dev_scripts/     conda-env helpers
docs/            theory + API
```

Each `runs/<name>/` holds a self-contained model: `decoder.linrd`, the `config.json`
that produced it, `state.pt` (resume), `loss.npy`, figures, and a `recon/` of its
reconstructions.

## Use

Run everything from the `experiments/` directory. `configs.py` holds a couple of
**templates** (`single`, `default`); tweak with `--set key=value`, and use `--name`
to keep each pretrained model under its own `runs/<name>/`.

```bash
cd experiments              # all commands below are run from here
python build_databases.py                 # build img_data/{natural,phantom}

# pretrain a model into runs/<name>/  (--exp = template, --name = kept model dir)
python pretrain.py --exp default --name blend
nohup python pretrain.py --exp default --name blend > runs/blend.log 2>&1 &   # long, in background
python pretrain.py --exp default --name blend --resume                       # continue if interrupted

# reconstruct with a saved model: point --model at its runs/<name>/ dir
python recon.py --model blend --image 0                      # natural image 0
python recon.py --model blend --database phantom --image 0   # phantom 0 = Shepp-Logan
python recon.py --model blend --database phantom --image 0 --set recon_steps=2000

cd .. && pytest                           # tests run from the repo root
```

`experiments/run_pretrain.sh` and `experiments/run_recon.sh` collect these commands.
Reference studies (frozen) live in `experiments/archive/` — see
[`archive/README.md`](experiments/archive/README.md).

Typical workflow: `build_databases` → `pretrain --exp <template> --name <model>`
(writes `runs/<model>/`) → `recon --model <model> --image <i>`. To keep a model,
just give it a unique `--name` (or rename its `runs/` dir). Long runs checkpoint
every `checkpoint_every` steps to `state.pt`; `--resume` continues from the run's
*saved* config.

Data and output paths are anchored to `experiments/` (so a script still works if
launched from elsewhere), but the convention is to run from `experiments/`. Runs on
CUDA, Apple MPS, or CPU automatically. All generated data lives in `img_data/` and
`runs/` (under `experiments/`), both gitignored.
