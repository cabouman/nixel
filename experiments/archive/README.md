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
