# archive — settled comparison studies

A/B scripts that answered a design question once and for all. They are kept for
reference (and to re-convince a skeptic) but are **not** part of the active
workflow. They run from anywhere; a `sys.path` shim lets them import
`experiments/_paths.py` from this subdirectory.

## The "progressive" question — two separate answers

- **Progressive PRETRAINING helps.** `ab_progressive.py` compares progressive
  (coarse-to-fine, bands `K=0..M` added one ring at a time) vs joint
  full-bandwidth training of θ at equal iteration budget. Progressive wins
  (~+0.8 dB single image). This is why `pretrain.py` uses progressive training.

- **Progressive RECONSTRUCTION does not help.** `ab_reconstruct.py` fits `z`
  against a frozen θ two ways — joint full-bandwidth vs a progressive band sweep.
  They tie (joint ~24.98 dB vs prog ~25.05 dB final) and joint is *faster* early.
  Fitting `z` against a fixed, well-conditioned θ is near-convex, so the band
  schedule buys nothing. **Reconstruction stays joint.**

## Why this matters

Coarse-to-fine is a tool for the hard, non-convex *training* of θ — not for the
much easier job of fitting `z` to an already-trained decoder.

## The adapt-θ question — fine-tune θ during reconstruction

Reconstruction is now `adapt_theta=True` by default: warm-start from the
pretrained θ, then fine-tune θ alongside `z`. (θ stays the same lightweight MLP —
it is evaluated per-voxel per-CT-iteration, so it can't grow.) All on nat_0,
P=8 C=8, 10000 steps, render PSNR:

- **Adapting θ beats frozen-θ (z-only).** `ab_adapt_theta.py`: frozen 26.71 dB vs
  adapt (theta_lr=0.1·LR) 28.49 dB — **+1.78 dB**, and no instability at unfreeze.

- **Use `theta_lr = LR`.** `ab_adapt_sweep.py` sweeps theta_lr: 1e-4→28.49,
  3e-4→28.85, **1e-3 (=LR)→29.24** — monotone; the original 0.1·LR was too timid.
  Equal-rate joint fine-tune wins. (All beat from-scratch `fit_one` 28.20, aided by
  the pre-paid prior.)

- **The warm start is the big lever; the warmup is minor.** `ab_warmstart.py`
  (theta_lr=LR): warm+warmup 29.23 dB, warm-no-warmup 29.04 (−0.20),
  **cold-from-scratch 27.38 (−1.86)**. The pretrained prior is worth ~1.9 dB; the
  z-only warmup only ~0.2 dB (kept at 200 as a cheap nicety).

**Settled defaults:** `adapt_theta=True`, `theta_lr=LR`, `theta_warmup=200`
(baked into `reconstruct.py` and the `linr.reconstruct` default).
