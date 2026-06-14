"""Checkpoint/resume of pretraining must reproduce an uninterrupted run (on CPU,
where the torch RNG state is fully captured)."""

import io
import numpy as np, torch, random
from linr import LinrDecoder, ArrayField


def _fit(dec, img, resume=None, checkpoint_fn=None):
    return dec._fit_bands(
        [ArrayField(img)], [(0, 6), (1, 6)], 1e-3, 64, 8,
        zero_init=(resume is None), on_step=None,
        checkpoint_every=(6 if checkpoint_fn is not None else 0),
        checkpoint_fn=checkpoint_fn, resume=resume)


def test_resume_matches_uninterrupted():
    img = torch.rand(16, 16)                      # shared target

    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    decA = LinrDecoder(4, channels=4)
    blob = {}

    def save_step6(step, ckpt):                   # serialize at step 6, like pretrain.py
        if step == 6:
            buf = io.BytesIO(); torch.save(ckpt, buf); blob["s6"] = buf.getvalue()

    repA, _, _ = _fit(decA, img, checkpoint_fn=save_step6)    # full run

    ckpt6 = torch.load(io.BytesIO(blob["s6"]), weights_only=False)
    decC = LinrDecoder(4, channels=4)             # init overwritten by resume
    repC, _, _ = _fit(decC, img, resume=ckpt6)    # resume from step 6

    assert len(repA.loss_history) == 12 and len(repC.loss_history) == 12
    assert np.allclose(repC.loss_history, repA.loss_history, atol=1e-6)
