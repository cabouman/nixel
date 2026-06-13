# LINR Theory v2 — Nested Fourier Bands and Progressive Training

This document extends [theory-v1.md](theory-v1.md). The foundation is
**unchanged**: coordinates (v1 §2), latent interpolation (v1 §3), the real 2-D DFT
Fourier basis (v1 §4), the decoder MLP (v1 §5), the core operations (v1 §6), and
the rest. v2 adds two things that let *one fixed-architecture decoder* be trained
**coarse-to-fine in frequency**:

1. a fixed **ordering** of the `F = P²` Fourier basis functions into **nested
   bands** (this document), and
2. a **progressive training schedule** that grows the active band with warm
   starts (§v2.6, to be written next).

The final model is *identical* to the v1 decoder — v2 changes only **how it is
trained**.

---

## v2.1 Why

Coordinate networks have a spectral bias: trained jointly, they fit low
frequencies first, plateau, then slowly discover the high frequencies (we observed
exactly this — a plateau followed by a later drop). Rather than hope the optimizer
crosses that plateau, we add frequency content in **stages**, each warm-started
from the converged previous stage. Because the integer-frequency basis is
**nested** (low frequencies are a subset of high), this needs *no architectural
change* — only a **mask**.

---

## v2.2 Nested bands: ordering the basis (L∞ / square rings)

From v1 §4 the basis is the `F = P²` real functions, each `cos(2π b·u)` or
`sin(2π b·u)` for an integer frequency `b = (m, n)`. Give each function a **band
index** equal to the L∞ (max) norm of its frequency:

$$K(b) = \max(|m|, |n|), \qquad K(b) \in \{0, 1, \dots, M\}, \quad M = P/2.$$

Order all $F$ functions by $K(b)$ ascending (DC, $K=0$, first), breaking ties by a
fixed rule (e.g. $(K,\ |m|+|n|,\ m,\ n,\ \cos \text{ before } \sin)$). This order is
fixed given $P$ and stored with the decoder. The **band-$K$ prefix** is

$$S_K = \{\ \text{basis functions with } K(b) \le K\ \}, \qquad
S_0 \subset S_1 \subset \dots \subset S_M = \text{(full basis)}.$$

Each step $K{-}1 \to K$ adds the **square ring** $\max(|m|,|n|) = K$: the highest
horizontal $(K,0)$, the highest vertical $(0,K)$, **and** the corner $(K,K)$,
together.

**Why L∞.** It matches the "max frequency $K$" picture — integer stages
$K = 1, \dots, P/2$ ending at the full basis — and it brings the corner in *with*
the highest axis frequencies, which is the natural grouping on a square lattice (a
lattice is not isotropic, so isotropic ordering has no special claim here).
Euclidean `√(m²+n²)` and L1/zig-zag `|m|+|n|` orderings also nest; they only change
the order in which corner frequencies arrive, which matters little. (L1 would add
one diagonal at a time if finer stages are ever wanted.)

**Why a prefix of the *one* full basis (not a basis rebuilt per $K$).** The real
DFT has four self-conjugate frequencies — DC and the three Nyquist corners — whose
sine vanishes; these are simply absent from the v1 basis. Ordering that single,
correct set of $P^2$ functions by $K(b)$ and taking prefixes never folds or rebuilds
anything; we only **reveal more of a fixed list**. Rebuilding a fresh basis at each
$K$ would mishandle the Nyquist folding at $K = M$.

**Why DFT and not DCT.** The DCT is natural for *independent* blocks (JPEG avoids
the DFT's periodic-wraparound edge discontinuity, getting better per-block energy
compaction). But our nixels are **not** independent blocks — they are one
continuous field whose within-nixel basis must be **identical in every nixel**
(translation invariance, v1 §8), i.e. periodic with period 1, `γ(u+1) = γ(u)`.
Integer-frequency cosines/sines are period-1; DCT modes are not —
$\cos(\pi k(u+1)) = (-1)^k \cos(\pi k u)$, which flips sign every nixel for odd $k$.
So a DCT would break translation invariance and reintroduce block-boundary
behavior. Moreover we store **latents** $z$, not transform coefficients, so the DCT's
compaction advantage does not even apply. DFT it is.

---

## v2.3 The masked decoder (one architecture, all bands)

$\gamma(u) \in \mathbb{R}^F$ is always computed in full. A **bandwidth mask**
$m_K \in \{0,1\}^F$ is $1$ for the functions in $S_K$ (a prefix, by the ordering
above) and $0$ for the rest. The decoder forward map (cf. v1 §5) becomes

$$f(r) = e^{a}\ \text{act}\Big(\text{MLP}_\theta\big(\ m_K \odot \gamma(u),\ \ z'(u)\ \big)\Big).$$

Partition the first layer $W_1 \in \mathbb{R}^{H \times (F + C)}$ by input column:

```
input :  [   γ_active (|S_K|)   |   γ_masked (F − |S_K|)   |    z'  (C)   ]
mask  :  [      1 … 1           |          0 … 0           |    1 … 1     ]
W1    :  [   Wγ_active          |       Wγ_masked          |    Wz        ]   (H × (F+C))

       h  =  Wγ_active · γ_active   +   0   (masked input)   +   Wz · z'   +   b1
```

The latent columns `Wz` are **always active**. The masked feature columns receive
zero input, so they contribute nothing: **the band-$K$ decoder is exactly a v1
decoder built with only the $|S_K|$ lowest-frequency features**. At $K = M$ the
mask is all ones and this is precisely the full v1 model.

---

## v2.4 Growth and warm start

We mask **inputs, not weights**: a weight column times a zero input is already
zero, so the first-layer feature columns need no masking of their own. The only
requirement is that a feature's column be **zero at the instant it is unmasked**,
or the output would jump. We get that for free by **zero-initializing all
first-layer feature columns** $W^\gamma$ at the start (the latent columns $W^z$,
biases, and deeper layers use the usual initialization). A masked feature has zero
input, hence *exactly zero gradient*, so its column never moves from zero until the
band reaches it.

Train in stages $K = K_0, \dots, M$. Going from stage $K{-}1$ to stage $K$:

1. **Grow the input mask** to include the next ring ($\max(|m|,|n|) = K$), so those
   features now pass through.
2. The newly activated columns are **already zero** (zero-init, never touched), so
   the output is *unchanged* from converged stage $K{-}1$ — the new features
   contribute nothing yet.
3. **Fine-tune everything** (warm-started): the active feature columns, **all
   deeper MLP layers**, the latent $z$, and the scale $a$ are optimized jointly;
   the newly activated columns train up from zero. The previous band's weights are
   an *initialization*, **not** frozen — they re-adapt to make room for the new
   band. (A strict-wavelet variant would freeze the old columns; not recommended —
   it costs final quality.)

So no weights are ever masked or explicitly re-zeroed at a growth step — the
seamless warm start falls out of zero-init plus zero-gradient on masked inputs. The
lower-frequency components' *contribution* starts identical at each growth, then
re-adapts for best overall fit.

Because stage `M` is the unmasked full model, a **progressively-trained** decoder
is architecturally identical to a **directly-trained** one — enabling a clean A/B
comparison: the *same* final model, reached by two different training paths.

---

## v2.5 What this changes and what it does not

- **Architecture / capacity:** unchanged. The mask only zeros inputs; at full
  bandwidth it *is* the v1 decoder. v2 is a **training organization**, not a bigger
  model. The quality ceiling at a given `P` is still set by `G`, `C`, and the
  architecture (v1).
- **Storage (`.linrd`):** the decoder still saves `{config, θ}`; `B` and its fixed
  L∞ ordering are regenerated from `P`. The active band `K` is a *training-stage*
  variable; a finished, saved decoder is at full bandwidth `K = M`.
- **Reconstruction / render / forward operators:** unchanged — they use the full
  decoder. (One *could* reconstruct at reduced bandwidth by masking, but the
  default is full bandwidth.)

---

## v2.6 Progressive (coarse-to-fine) training

The point is to reach the **full** $K = M$ model along a good path — adding
frequency content band by band, each warm-started from the converged previous
band — rather than training all frequencies jointly and stalling on the
spectral-bias plateau.

### The schedule

Fix $P$ (hence $M = P/2$ and the grid $G$). Initialize with a zeroed first-layer
feature block and band $K_0$ (`init_progressive(K_0)`, §v2.4). Then sweep the band
**one ring at a time**, $K = K_0, K_0{+}1, \dots, M$. At each stage $K$:

1. **Grow the input mask** to band $K$ (`set_bandwidth(K)`). The newly unmasked
   columns are already zero, so the output is unchanged — a seamless warm start.
2. **Train a stage of $T$ steps** at bandwidth $K$ with a **fresh cosine LR
   schedule** (a warm restart over the $T$ steps), fine-tuning **everything**: the
   active feature columns, all deeper layers, the latent $z$, and the scale $a$.
3. Advance to $K{+}1$.

The final stage $K = M$ is exactly the full v1 model.

### Choices (fixed)

- **Granularity:** one ring per stage ($K {+}{=} 1$). Rings may be grouped later if
  fewer stages are wanted.
- **Start band:** $K_0 = 0$ — begin from "latent + per-nixel DC," i.e. the smooth
  latent-grid model (the latent carries the cross-nixel structure), then add
  within-nixel frequencies band by band.
- **LR restart:** a fresh **cosine** decay each stage (warm restart). Without it, a
  single global schedule would decay the learning rate to ~0 before the late bands
  ever train.
- **Target:** the **full-resolution** image at every stage. A low-$K$ model simply
  cannot represent the high frequencies and converges to a band-limited fit;
  low-pass-filtering the target per stage (a truer Laplacian pyramid) is a
  documented fallback if coarse stages ever misbehave.
- **What is staged:** only the within-nixel **bandwidth**. The latent $z$ (full
  $G \times G$) and the cross-nixel structure are trained from stage 1 and keep
  adapting; the grid is **not** staged.

This schedule applies unchanged whether the dataset is a single image (the
direct-fit case) or many images (auto-decoder pretraining) — a stage is just $T$
optimizer steps over the dataset at the current bandwidth.

### A/B protocol (evaluation)

Two training paths to the **same** final model ($K = M$):

- **Progressive** — the schedule above; total budget $\sum_{K=K_0}^{M} T$ steps.
- **Joint baseline** — full bandwidth from the start, standard init, trained for
  the **same total** number of steps.

The mask does not change the per-step cost (we still compute the full $\gamma$ and
the full first-layer matmul), so **equal iteration budget = equal compute** — a
clean comparison. Report **quality (PSNR / NRMSE) vs. cumulative iterations** for
both paths. Hypothesis: progressive reaches a given quality in fewer iterations
and/or settles in a better final minimum by sidestepping the spectral-bias
plateau.

> Implementation note: the mask currently zeros inactive features *after*
> computing them, so low bandwidth gives no speedup. A future optimization could
> compute only the active prefix of $\gamma$ (and use the corresponding columns of
> $W_1$) for a real per-step saving at coarse stages; it does not change the math.
