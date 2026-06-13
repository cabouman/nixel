"""Correctness checks for the theory-v2 nested-band masking (run: pytest)."""

import torch
import torch.nn.functional as F

from linr import LinrDecoder

P, C, G = 4, 8, 16


def test_basis_structure():
    d = LinrDecoder(P, channels=C)
    fb = d.feature_band
    assert d.num_features == P * P
    assert d.M == P // 2
    assert float(fb.min()) == 0.0 and float(fb.max()) == float(d.M)
    assert torch.equal(fb, torch.cat([d.Bcos.abs().amax(0), d.Bsin.abs().amax(0)]))


def test_band_nesting():
    d = LinrDecoder(P, channels=C)
    fb = d.feature_band

    def active(K):
        return set((fb <= K).nonzero().flatten().tolist())

    assert all(active(K).issubset(active(K + 1)) for K in range(d.M))
    assert len(active(d.M)) == d.num_features      # K=M activates everything
    assert len(active(0)) == 1                     # K=0 is DC only


def test_default_is_full_bandwidth_noop():
    d = LinrDecoder(P, channels=C)
    assert d.bandwidth == d.M
    assert bool((d.feature_band <= d.M).all())     # full mask is all ones


def test_init_progressive_zeros_only_features():
    d = LinrDecoder(P, channels=C)
    d.init_progressive(0)
    W0 = d.mlp[0].weight
    assert int(torch.count_nonzero(W0[:, :d.num_features])) == 0   # feature cols zeroed
    assert int(torch.count_nonzero(W0[:, d.num_features:])) > 0    # latent cols kept
    assert int(torch.count_nonzero(d.mlp[0].bias)) > 0
    assert int(torch.count_nonzero(d.mlp[2].weight)) > 0           # deeper layer kept


def test_zero_features_make_output_bandwidth_independent():
    d = LinrDecoder(P, channels=C)
    d.init_progressive(0)
    c = torch.rand(7, 2) * 2 - 1
    z = torch.randn(C, G, G)
    o0 = d.decode(c, z).detach().clone()
    d.set_bandwidth(d.M)
    assert torch.allclose(o0, d.decode(c, z), atol=1e-6)


def test_masked_columns_stay_exactly_zero_then_move_after_growth():
    torch.manual_seed(1)
    d = LinrDecoder(P, channels=C)
    d.init_progressive(1)                          # bandwidth K=1
    z = torch.nn.Parameter(0.01 * torch.randn(C, G, G))
    opt = torch.optim.Adam(list(d.mlp.parameters()) + [z], lr=1e-2)
    target = torch.rand(64)

    def step():
        loss = F.mse_loss(d.decode(torch.rand(64, 2) * 2 - 1, z), target)
        opt.zero_grad(); loss.backward(); opt.step()

    for _ in range(5):
        step()
    Wf = d.mlp[0].weight[:, :d.num_features]
    masked = d.feature_band > 1                     # band-2 features (masked at K=1)
    active = d.feature_band <= 1
    assert int(torch.count_nonzero(Wf[:, masked])) == 0   # masked stay EXACTLY zero
    assert int(torch.count_nonzero(Wf[:, active])) > 0    # active moved

    d.set_bandwidth(2)                             # grow the band
    for _ in range(5):
        step()
    grown = d.mlp[0].weight[:, :d.num_features][:, masked]
    assert int(torch.count_nonzero(grown)) > 0     # previously-masked now move
