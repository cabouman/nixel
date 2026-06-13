# LINR ("nixel") — Theory and Notation

This document describes, in mathematical detail, the algorithm currently
implemented in `linr_phantom.py` (`class LINR`) and `linr_pretrain.py`. It is the
roadmap; the code follows it line for line.

---

## 1. Goal

Represent a continuous 2-D field (an image, ultimately a 3-D volume) as

* a coarse grid of small local latent vectors ("**nixels**"), `z`, plus
* a single **shared decoder** `θ` (and a fixed Fourier matrix `B`),

so that the field can be evaluated at **any** coordinate, is **continuous**, is
**differentiable** w.r.t. `z` and `θ`, and uses far fewer numbers than a dense
array. For tomography, the per-point dependence on only a few local nixels keeps
the forward/back-projection Jacobian **sparse and local** (Section 6).

---

## 2. Coordinates

The spatial coordinate is global and normalized:

$$r = (x, y) \in [-1, 1]^2 .$$

The representation is continuous in `r`; an array size `N` only enters when we
*sample* it onto an `N×N` grid for display or error measurement.

### Nixel grid

The latent is a tensor

$$z \in \mathbb{R}^{C \times G \times G},$$

i.e. a $G \times G$ grid of nixels, each holding a length-$C$ latent vector.
Write the nixel at integer grid position $(p, q)$ — with $p$ the row and $q$ the
column, $p, q \in \{0, \dots, G-1\}$ — as $z_{:,\,p,\,q} \in \mathbb{R}^{C}$.

### Nixel-unit coordinate `u` (the key step)

We convert `r` to a **nixel-unit** coordinate `u`, using the **cell-centered**
(`align_corners=False`) convention:

$$u_x = \tfrac{x+1}{2}\,G - \tfrac12, \qquad u_y = \tfrac{y+1}{2}\,G - \tfrac12 .$$

Properties (this is the part to internalize):

* **Integer `u` = nixel centers.** `u = k` lands exactly on nixel `k`. So nixel 0
  is centered at `x = -1 + 1/G`, nixel `G-1` at `x = 1 - 1/G`. The nixels tile
  `[-1,1]` as `G` cells of width `2/G`, each nixel at its cell's center.
* **Pixel pitch.** If the field is sampled at `N×N`, one nixel spans
  `P = N/G` pixels. `P` is the only thing that sets the compression ratio
  (Section 9).
* **`u` is a global, monotonic, affine function of `r`.** It is *not* a per-nixel
  coordinate that resets at boundaries — that distinction is what preserves
  continuity (Section 8).

Both the latent interpolation (Section 3) and the Fourier features (Section 4)
are computed from `u`, **not** from the raw global `r`. This is what makes the
decoder grid-relative and transferable (Section 8).

> Code: `u = (coords + 1)*0.5*G - 0.5` in `LINR.decode`; the same mapping is
> recomputed inside `LINR._bilinear` (`fx = (x+1)*0.5*W - 0.5`).

---

## 3. Latent interpolation `z'(r)`

$z'(r) \in \mathbb{R}^{C}$ is the bilinear interpolation of the four nixels
surrounding $u$. With

$$q_0 = \lfloor u_x \rfloor,\ q_1 = q_0+1,\qquad p_0 = \lfloor u_y \rfloor,\ p_1 = p_0+1,$$
$$w_x = u_x - q_0,\qquad w_y = u_y - p_0,$$

the interpolant is

$$z'(r) = (1-w_y)\big[(1-w_x)\,z_{:,p_0,q_0} + w_x\,z_{:,p_0,q_1}\big]
        + w_y\big[(1-w_x)\,z_{:,p_1,q_0} + w_x\,z_{:,p_1,q_1}\big].$$

**Edge handling — replicate (clamp) padding.** Indices are clamped to
`[0, G-1]` and the weights to `[0,1]`. So a query in the outer half-cell
(`u < 0` or `u > G-1`) collapses onto the boundary nixel: the outer ring takes
the boundary nixel's value (constant extrapolation). `z'(r)` is therefore
continuous everywhere, with zero gradient just outside the last nixel center.

> Code: `LINR._bilinear`. Implemented by hand (not `grid_sample`) because
> `grid_sample`'s backward pass is unimplemented on Apple MPS.

---

## 4. Fourier features `γ(u)` — a periodic, shared per-nixel basis

`γ(u)` must be **identical in every nixel** — one shared local model, with nixels
distinguished only through `z'`. So `γ` is **periodic with period 1 per nixel**,
which forces **integer frequencies**: for integer $b=(m,n)$, $\cos/\sin(2\pi b\cdot u)$
depend only on the within-nixel position, repeat every nixel, and are smooth at
the nixel boundaries (so $f$ stays continuous, §8).

Concretely, `γ(u)` is the **real 2-D Fourier basis of the $P\times P$ within-nixel
patch**. Let $M = P/2$ (Nyquist; $P$ even) and $b\cdot u = m\,u_x + n\,u_y$. The
basis functions are:

* **Cosines** $\cos(2\pi\, b\cdot u)$ for $b=(m,n)$ in
  * $n = 1,\dots,M{-}1$, $\ m = -M{+}1,\dots,M$  (interior columns), and
  * $n \in \{0,\,M\}$, $\ m = 0,\dots,M$  (the two self-symmetric columns);
* **Sines** $\sin(2\pi\, b\cdot u)$ for the **same** $b$ **except** the four
  self-conjugate corners $(0,0),(M,0),(0,M),(M,M)$, where $\sin(2\pi b\cdot u)\equiv 0$
  on the pixel grid and is dropped.

Equivalently: take one representative of each conjugate pair $\{b,-b\}$; every
representative gets a cosine, and all but those four self-conjugate frequencies
(DC and the three Nyquist corners) also get a sine. The totals are
$\tfrac{P^2}{2}+2$ cosines and $\tfrac{P^2}{2}-2$ sines, so

$$\boxed{\ \gamma(u)\in\mathbb{R}^{F},\qquad F = P^2\ }\qquad (P=16 \Rightarrow F=256),$$

exactly the $P^2$ real degrees of freedom of the patch — a **lossless** basis.
$P$ is the single design parameter: it fixes $M=P/2$ and $F=P^2$. (The naive full
grid $|m|,|n|\le M$ would give $(2M+1)^2=(P+1)^2$, double-counting the $\pm$Nyquist
line; the correct count is $P^2$.)

> Code: `Decoder.fourier(u)` stacks these $\cos$/$\sin$ over a fixed integer
> frequency table built once from $P$. Integer $b$ makes $b\cdot u$ automatically
> periodic, so the global nixel coordinate $u$ is used directly (no `frac` needed).

---

## 5. Decoder MLP and output

The decoder input is the concatenation of the periodic encoding and the
interpolated latent:

$$h = [\,\gamma(u),\ z'(r)\,] \in \mathbb{R}^{F + C}, \qquad F = P^2 .$$

**Architecture.** An MLP of width $H$ with $L$ ReLU hidden layers, then a linear
map to one scalar:

| layer | map | activation |
|-------|-----|------------|
| 1 | $\mathbb{R}^{F+C} \to \mathbb{R}^{H}$ | ReLU |
| $2, \dots, L$ | $\mathbb{R}^{H} \to \mathbb{R}^{H}$ | ReLU |
| out | $\mathbb{R}^{H} \to \mathbb{R}$ | (none) |

$$\text{MLP}_\theta(h) = W_{\text{out}}\,\sigma\big(W_L \cdots \sigma(W_1 h + b_1)
\cdots + b_L\big) + b_{\text{out}}, \qquad \sigma = \text{ReLU}.$$

**Output.** Apply an activation `act`, then the global scale $a$:

$$f(r) = e^{a}\cdot \text{act}\big(\text{MLP}_\theta(h)\big),$$

with `act = identity` (`out_act="linear"`, signed/unbounded) or `softplus`
(`out_act="nonneg"`, for non-negative quantities such as CT attenuation). Since
$e^{a} > 0$, the scale preserves non-negativity.

**Parameter count** (example $P=16 \Rightarrow F=256$, $C=8$, $H=128$, $L=4$):

| block | params | value |
|-------|--------|-------|
| Linear 1 | $(F+C)\,H + H = 264\cdot128 + 128$ | 33,920 |
| Linear $2\text{–}4$ | $3\,(H^2 + H) = 3(128^2 + 128)$ | 49,536 |
| Linear out | $H + 1$ | 129 |
| **total $\theta$** | | **83,585** |

$F = P^2$ sets only the **first-layer width** — a decoder compute/size cost
(shared and amortized across all reconstructions), *not* per-reconstruction
storage.

**Global scale $a$.** A single scalar **per reconstruction** (it scales the whole
field uniformly — nothing to interpolate), stored in log space so optimizer steps
are multiplicative and can span orders of magnitude in a few iterations; default
$a = 0$ (scale 1), and $e^{a}$ may be initialized to a known data scale.

> Code: `Decoder.decode(coords, z, a)`; `a` is the log value-scale.

### Full forward map

$$\boxed{\,f(r) \;=\; e^{a}\cdot \text{act}\Big(\text{MLP}_\theta\big(\;
\gamma\!\big(\underbrace{\tfrac{r+1}{2}G-\tfrac12}_{u}\big),\;
\underbrace{\text{interp}(z, u)}_{z'(r)}\;\big)\Big)\,}$$

Trainable: `z` (the reconstruction), `a` (per reconstruction), `θ` (shared).
Fixed: `B`.

---

## 6. Core operations: pretrain, reconstruct, render

Three operations: **pretrain** learns the shared decoder; **reconstruct** fits a
field's latent with the decoder fixed; **render** turns a reconstruction into an
$N\times N$ image.

### Pretrain — learn the shared decoder

Given a set of fields $\{x_i\}_{i=1}^{M}$, jointly estimate the shared $\theta$ and
a **separate** latent/scale $(z_i, a_i)$ per field:

$$(\hat\theta,\ \{\hat z_i, \hat a_i\})
= \arg\min_{\theta,\ \{z_i, a_i\}}\ \sum_{i=1}^{M}\ \sum_{r \in R}\big(
x_i(r) - f(r;\, z_i, a_i, \theta)\big)^2,$$

where $R$ is a **finite set of sample coordinates** in $[-1,1]^2$ (re-drawn
randomly each step — stochastic optimization), not a true expectation: we can
only ever sum over a sampled subset of coordinates.

$\theta$ appears in every term, so its gradient averages over all images and it
becomes a common decoder; each $(z_i, a_i)$ receives gradient only when image $i$
is sampled and specializes to that image. $B$ is fixed. In practice we sample a
minibatch of images × random coordinates per step (Adam, cosine decay). The
$(z_i, a_i)$ are byproducts; only $(\theta, B)$ is saved.

### Reconstruct — fit latent field with fixed decoder

With $(\theta, B)$ frozen, estimate the latent and scale of a new field from
measurements $y$:

$$(\hat z,\ \hat a) = \arg\min_{z,\ a}\ \big\| y - \mathcal{A}\,f(\cdot\,; z, a, \theta) \big\|^2 + R(z),$$

where $\mathcal{A}$ is the identity for plain image fitting, or the **tomographic
forward operator** for CT, and $R(z)$ is a possible regularizing term for the
reconstruction. The estimate $(\hat z, \hat a)$ *is* the reconstruction (saved as
`.linrz`).

### Render — form pixels from nixels

Rendering evaluates the continuous field on an $N\times N$ pixel grid to produce
the displayed image $\hat f \in \mathbb{R}^{N\times N}$:

$$\hat f_{ij} = f(r_{ij};\, \hat z, \hat a, \theta), \qquad i, j = 0, \dots, N-1,$$

where $r_{ij}$ are the cell-centered pixel coordinates. $N$ is chosen at render
time — the representation is continuous, so it can be rendered at **any**
resolution from the same $(\hat z, \hat a)$.

### Locality (the CT payoff)

A projection is a line integral $p = \sum_k w_k\, f(r_k)$. Each $f(r_k)$ depends
on only the **4 nixels** surrounding $r_k$ (via $z'$) plus the shared $\theta$.
Hence $\partial f(r_k)/\partial z$ is nonzero for just those 4 nixels, and a
nixel receives gradient
only from rays passing through its support — the same sparse, local structure as
the conventional CT system matrix, but over a compressed `z`.

---

## 7. What is "the reconstruction" vs "the decoder"

* **Decoder** (shared, reused across many reconstructions): `(θ, B)`. This is the pretrained decoder model.
* **Reconstruction** (per image/volume): `(z, a)`. This is the compressed nixel field.

The shared decoder is saved to a `.linrd` file:

| field | description |
|-------|-------------|
| `id` | content hash of the decoder; a reconstruction stores it to verify the pairing |
| `P` | pixels per nixel — the design parameter (sets `K=P/2`, `F=P²`) |
| `C` | latent channels per nixel |
| `H` | decoder MLP hidden width |
| `L` | number of ReLU hidden layers in the MLP |
| `out_act` | output activation: `linear` (signed) or `nonneg` (softplus) |
| `B` | fixed integer Fourier frequency table (built deterministically from `P`) |
| `θ` | trained MLP weights — the shared decoder itself |

The reconstruction is saved to a `.linrz` file:

| field | description |
|-------|-------------|
| `z` | latent nixel grid, shape `(C, G, G)` — the compressed field |
| `a` | log global value scale (so `exp(a)` scales the output) |
| `decoder_id` | `id` of the decoder required to decode this `z`; checked on load |
| `metadata` | array size `N`, orientation, value units, date, etc. |

Because `K`, `F`, and the architecture all derive from `(P, C, H, L, out_act)`,
those five fields plus `B` and `θ` are everything needed to rebuild the decoder;
the `decoder_id` stamp on each `.linrz` makes a wrong decoder pairing fail loudly.

---

## 8. Continuity and grid-invariance

* **Continuity of `f`.** `u` is a continuous (affine) function of `r`; `z'(r)`
  is continuous (bilinear); `γ(u)` is continuous. A composition of continuous
  maps is continuous, so `f` has no seams at nixel boundaries. This is why we use
  a *global* `u` and never a per-nixel coordinate that resets to 0 in each cell
  (that would make `f` discontinuous).
* **Grid-invariance / transfer of `θ`.** Both inputs to the MLP are expressed in
  nixel units. Changing `G` changes how many nixels tile the image and the map
  `r → u`, but the decoder's learned function of `(γ(u), z')` is unchanged in
  meaning. Hence the *same* `θ`, `B` can decode latents at any grid size, and a
  decoder trained at one `G` warm-starts another.
* **Translation-invariance across nixels.** Because `γ` is periodic with period 1
  (integer frequencies, §4), the within-nixel basis is *identical* in every nixel;
  the decoder applies one shared local model and distinguishes nixels only via
  `z'`.

The fully grid-*invariant* quantity is the fractional within-nixel position, but
using only that breaks continuity. Absolute `u` keeps both continuity and
transfer.

---

## 9. Compression ratio

With pixels-per-nixel $P = N/G$ and latent size $|z| = C\,G^2$,

$$\text{CR} = \frac{N^2}{|z|} = \frac{N^2}{C\,G^2} = \frac{(N/G)^2}{C}
            = \frac{P^2}{C}.$$

`θ` and `B` are **not** counted: they are shared and amortize to ~0 per
reconstruction at 3-D scale. CR depends only on `P` and `C`, not on the absolute
array size `N`.

---

## 10. Hyperparameters

| symbol | code | meaning |
|--------|------|---------|
| `G` | `grid` | nixels per side (`z` is `C×G×G`) |
| `C` | `channels` | latent length per nixel |
| `P` | `pixels_per_nixel` | **decoder design parameter** `= N/G`; sets `K=P/2`, `F=P²`, and CR `= P²/C` |
| `K` | (derived) | within-nixel Nyquist `= P/2` (max integer frequency) |
| `F` | (derived) | number of Fourier features `= P²` (complete per-nixel basis) |
| `H, L` | `hidden, layers` | decoder MLP width / depth |
| `act` | `out_act` | `linear` (signed) or `nonneg` (softplus) |
| `a` | `log_scale` | per-reconstruction global value scale (log) |

---

The user-facing API is specified separately in [`API_def.md`](API_def.md).
