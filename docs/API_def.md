# LINR — User-facing API (proposed)

This is a **proposal** for how the LINR framework should be structured for use in
an application. It is the user's point of view: what objects exist, what you call,
and what you get back. The math is in [`theory.md`](theory.md); this document is
about software structure. (It does not yet match the current exploratory code in
`linr_phantom.py`; migrating to it is the next implementation step.)

Target: **PyTorch**. Forward operators are **pluggable** (image-fitting today, a
CT projector later). The reconstruction loop is available **both** as a ready-made
call *and* as a low-level differentiable model you can drive from your own
optimizer / reconstruction framework.

---

## 1. Design in one picture

Three objects, three operations.

| object | what it is | owns |
|--------|-----------|------|
| `LinrDecoder` | the **shared** decoder, built for a target `P` | `θ` (MLP), `B` (Fourier table) |
| `Reconstruction` | **one field's** representation, bound to a decoder | `z` (latent grid), `a` (log scale) |
| `ForwardOperator` | a pluggable measurement model `A` | maps a continuous field → measurements |

| operation | call | produces |
|-----------|------|----------|
| **pretrain** | `decoder.pretrain(dataset, cfg)` | trains `θ`; returns a `TrainReport` |
| **reconstruct** | `decoder.reconstruct(y, A, cfg)` | a `Reconstruction` + `ReconReport` |
| **render** | `recon.render(N)` | an `N×N` array |

The separation matters: `LinrDecoder` is created/trained **once** and shared;
each scan produces its own small `Reconstruction`. A reconstruction is only
meaningful paired with the decoder that made it, so it is stamped with that
decoder's id (see [`theory.md` §7](theory.md)).

---

## 2. `LinrDecoder` — the shared decoder

```python
class LinrDecoder(torch.nn.Module):
    def __init__(self, pixels_per_nixel: int, channels: int = 8,
                 hidden: int = 128, layers: int = 4,
                 out_act: str = "linear", device=None):
        # pixels_per_nixel = P  ->  derives K = P//2, F = P*P, and the fixed
        #                          integer Fourier table B (real 2-D DFT basis).
        # out_act in {"linear", "nonneg"}.

    # --- identity / introspection ---
    P: int; channels: int; num_features: int   # F = P*P
    id: str                                     # content hash; stamped into reconstructions

    # --- low-level differentiable field evaluation ---
    def decode(self, coords, z, a=0.0) -> Tensor:
        # coords: (..., 2) in [-1,1]^2;  z: (C,G,G);  a: scalar (log scale)
        # returns f(coords) = exp(a) * act(MLP(gamma(u), interp(z,u)))   (differentiable)

    # --- operations ---
    def pretrain(self, dataset, cfg: TrainConfig) -> TrainReport: ...
    def reconstruct(self, y, A: ForwardOperator, cfg: ReconConfig) -> ReconResult: ...

    # --- persistence (.linrd) ---
    def save(self, path: str = DEFAULT_DECODER): ...
    @classmethod
    def load(cls, path: str = DEFAULT_DECODER, device=None) -> "LinrDecoder": ...
```

`decode` is the single primitive everything else is built on. `pretrain` mutates
`θ` in place; `reconstruct` leaves `θ` frozen.

---

## 3. `Reconstruction` — one field's representation

Holds the learnable `z` and `a` for a single field, bound to a decoder. This is
**the differentiable image model** — pass `recon.field` to any forward operator
and backprop into `recon.parameters()`.

```python
class Reconstruction:
    def __init__(self, decoder: LinrDecoder, grid: int, init_scale: float = 1.0):
        # grid = G; choose G = N // P so the within-nixel density matches the decoder.
        # creates fresh learnable z (C,G,G) and a (scalar log-scale).

    # --- the continuous field (differentiable in z, a) ---
    def field(self, coords) -> Tensor:          # = decoder.decode(coords, self.z, self.a)
    def parameters(self) -> list[Tensor]:        # [z, a] -- hand these to an optimizer
    def set_scale(self, s: float); def get_scale(self) -> float

    # --- render to a raster ---
    def render(self, N: int) -> Tensor:          # (N, N) array; N is free (continuous)

    # --- persistence (.linrz) ---
    def save(self, path: str, metadata: dict | None = None): ...
    @classmethod
    def load(cls, path: str, decoder: LinrDecoder) -> "Reconstruction": ...
                                                 # raises if decoder.id != stored decoder_id
```

---

## 4. `ForwardOperator` — pluggable measurement model `A`

`A` maps the continuous field to measurements. It owns *which* coordinates to
query and *how* to combine them, so the same `Reconstruction` plugs into any
reconstruction problem.

```python
class ForwardOperator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, field: Callable[[Tensor], Tensor]) -> Tensor:
        # field: coords (...,2) -> values; returns predicted measurements (differentiable)

class ImageGrid(ForwardOperator):       # image fitting: A = identity sampling
    def __init__(self, N: int): ...      # queries field at the N×N pixel grid -> (N,N)

# Future (CT): a ray projector with the same interface.
class RayProjector(ForwardOperator):     # line integrals — NOT grid sampling
    def __init__(self, geometry): ...
    # for each ray: build quadrature points r_k in [-1,1]^2 along the ray,
    # evaluate field(r_k), and return  p = sum_k w_k * f(r_k)  -> sinogram
```

Reconstruction then solves `(ẑ, â) = argmin_z,a || y − A(recon.field) ||² + R(z)`.

**Off-grid evaluation is built in.** `field` (hence `decode`) accepts an
*arbitrary* batch of coordinates `(...,2)` — there is no fixed-grid assumption
anywhere in the core. Each operator generates whatever points it needs and calls
`field` there: `ImageGrid` uses the pixel lattice; `RayProjector` uses the
points **along each projection ray**. Two contract points for any operator: it
must (1) emit coordinates in the LINR's normalized `[-1,1]²` frame, and (2) own
its quadrature weights `w_k`. The per-point locality (4 nixels + `θ`, hence the
sparse Jacobian) holds for arbitrary ray points exactly as for grid points.

---

## 5. Operations: controls in, feedback out

Controls and feedback are explicit objects so calls are self-documenting and the
loss convergence is returned (not just printed).

```python
@dataclass
class TrainConfig:
    steps: int = 4000
    lr: float = 1e-3
    coords_per_step: int = 4096      # |R|, the sampled coordinate set per step
    images_per_step: int = 8         # minibatch of fields
    grids: list[int] | int = 16      # latent grid(s) assigned to training fields
    on_step: Callable | None = None  # optional callback(step, loss) for live monitoring

@dataclass
class ReconConfig:
    steps: int = 1500
    lr: float = 5e-3
    grid: int | None = None          # G; default N // P
    regularizer: Callable | None = None   # R(z); None = no regularization
    on_step: Callable | None = None

@dataclass
class TrainReport:
    loss_history: list[float]        # per-step loss -> convergence curve
    final_loss: float
    seconds: float

@dataclass
class ReconResult:
    recon: Reconstruction            # the product
    loss_history: list[float]
    final_loss: float
```

`pretrain` takes a **dataset** = a collection of fields, where a "field" is
anything samplable at coordinates (so analytic phantoms and rasters share one
interface):

```python
class Field(Protocol):
    def sample(self, coords: Tensor) -> Tensor: ...   # coords (...,2) -> values

class ArrayField(Field):    ...   # bilinear-samples a stored N×N raster
class EllipseField(Field):  ...   # analytic Shepp-Logan-style phantom
```

---

## 6. Typical usage

### Pretrain a decoder, with feedback

```python
decoder = LinrDecoder(pixels_per_nixel=16, channels=8)
dataset = [ArrayField(img) for img in load_training_images()]

report = decoder.pretrain(dataset, TrainConfig(steps=4000,
                          on_step=lambda s, l: print(s, l)))
plot(report.loss_history)        # convergence
decoder.save()                   # -> ./linr_params/decoder.linrd
```

### Reconstruct an image (A = identity), then render

```python
decoder = LinrDecoder.load()
y = load_image(N=512)                       # the measurement
res = decoder.reconstruct(y, ImageGrid(N=512), ReconConfig(steps=1500))
fhat = res.recon.render(512)                # (512,512) reconstruction
res.recon.save("recons/scanA.linrz", metadata={"units": "1/cm"})
```

### Reconstruct from CT data (future projector — same shape of call)

```python
res = decoder.reconstruct(sinogram, RayProjector(geometry), ReconConfig(steps=15))
volume_slice = res.recon.render(1024)
```

### Drive the low-level model from your own optimizer / framework

```python
recon = Reconstruction(decoder, grid=512 // decoder.P)
opt = torch.optim.Adam(recon.parameters(), lr=5e-3)
for _ in range(n_iter):
    y_pred = A(recon.field)                 # your A; or A∘recon.field inside your code
    loss = (y - y_pred).pow(2).sum() + R(recon.z)
    opt.zero_grad(); loss.backward(); opt.step()
```

The last pattern is the integration point for an external MBIR framework: LINR
supplies the differentiable field model and its parameters; your code owns the
solver.

---

## 7. Design rationale (why this shape)

* **Decoder vs. Reconstruction split** mirrors the math (shared `θ` vs. per-field
  `z,a`) and the storage (`.linrd` vs. `.linrz`). It makes the "pretrain once,
  reconstruct many" workflow the natural one.
* **`decode` as the one primitive** keeps a single differentiable path; `render`,
  `field`, and every operator are thin layers over it — easy to test and to trust.
* **Pluggable `ForwardOperator`** is what lets the same representation serve image
  fitting and CT (and anything else) without touching the core.
* **Both loop levels**: `reconstruct(...)` for the common case; `Reconstruction.field`
  + `parameters()` for embedding inside a larger reconstruction algorithm.
* **Config / Report objects** make controls explicit and return the loss
  convergence as data, rather than burying it in prints.
